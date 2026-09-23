from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from freetoken.distributed import DistributedInfo
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs
    from .supervisor import BackendHandle


def _report_startup_error(ack_queue: mp.Queue, exc: BaseException) -> None:
    """Tell the parent WHY this worker is dying — push an ("error", reason) ack before it exits,
    so the supervisor reports the real cause (e.g. a config ValueError) instead of the generic
    "backend worker … exited during load". Best-effort and flushed (close + join_thread), since
    the process is about to terminate; a failure to report must never mask the original error."""
    try:
        ack_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        ack_queue.close()
        ack_queue.join_thread()
    except Exception:  # noqa: BLE001 -- reporting is a nicety; never shadow the real exception
        pass


def _detach_process_group() -> None:
    """Shell mode only: move this worker out of the terminal's foreground process group.

    The shell binds ^C to "cancel this turn", but a terminal delivers SIGINT to the whole
    foreground group — which, with the engine running in this same process, includes the
    workers. They would take the same ^C and exit (``_run_scheduler`` below stops gracefully on
    KeyboardInterrupt), leaving the shell chatting with a dead engine. Nothing depends on the
    signal reaching them: the parent tears the workers down explicitly on every stop path
    (uvicorn's lifespan, plus the SIGTERM/SIGHUP handler and the reap backstop in api_server).

    ``ft serve`` keeps the default — uvicorn owns ^C there, and the group-wide delivery is part
    of how it stops."""
    try:
        os.setpgrp()
    except OSError:  # no job control (already a group leader / unusual environment)
        pass


def apply_mtp_env_gate(server_args: "ServerArgs", logger: logging.Logger) -> None:
    """``--spec-mtp > 0`` needs ``FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1`` in the spawned
    scheduler (the verify step's accept decision is host-side and must land before the next
    forward launches -- see ``scheduler.py``'s raise). Auto-set it here, in the parent,
    before the scheduler subprocess spawns (a spawned child re-execs and inherits
    ``os.environ``) instead of making the user export it by hand. A user who explicitly set
    it to "0" is left alone -- the scheduler still raises in that case."""
    if server_args.spec_mtp <= 0:
        return
    if "FREETOKEN_DISABLE_OVERLAP_SCHEDULING" in os.environ:
        return
    os.environ["FREETOKEN_DISABLE_OVERLAP_SCHEDULING"] = "1"
    logger.info(
        "--spec-mtp %d: auto-setting FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 for the "
        "spawned scheduler",
        server_args.spec_mtp,
    )


def apply_tuning_profile_env_gate(server_args: "ServerArgs", logger: logging.Logger) -> None:
    """Env-backed v1 tunables (``FREETOKEN_SPEC_DEFER_REPLAY``, ``FREETOKEN_DRAFT_GRAPH``)
    from a persisted ``ft tune`` profile, setdefault'd into the parent's environ before the
    scheduler subprocess spawns -- same inheritance mechanism as ``apply_mtp_env_gate``, and
    never touches a key the user (or an earlier gate) already set.

    Only the two env-backed tunables are applied here; ``spec_mtp`` and ``moe_strategy`` are
    CLI-flag/config-level tunables that need arg-resolution-time application (args.py /
    engine.py, not this spawn-time hook), and are deliberately left to a later pass -- see
    ``freetoken.tuning.profile.apply_env_defaults`` for why the "only if unset" precedence is
    the same either way.

    GPU identification uses NVML directly (``gpu_select._nvml_uuids``), not
    ``torch.cuda`` -- this runs before ``mp.Process(..., start_method="spawn")``, and
    initializing a CUDA context in the parent here would be the exact kind of side effect
    that mechanism has to avoid. No NVML, or no ``--max-seq-len-override`` (the key's
    context bucket needs a concrete value), skips the profile lookup entirely; a serving
    request never depends on it.
    """
    if server_args.max_seq_len_override is None:
        return
    try:
        from freetoken.gpu_select import _nvml_uuids

        uuids = _nvml_uuids()
        gpu_uuid = uuids[0] if uuids else None
    except Exception:  # noqa: BLE001 -- best-effort; a broken NVML load must never block boot
        gpu_uuid = None
    if not gpu_uuid:
        return
    from freetoken.tuning.profile import apply_env_defaults, compute_key, load

    key = compute_key(
        gpu_uuid=gpu_uuid,
        model_path=server_args.model_path,
        kv_format=server_args.kv_format,
        max_seq_len=server_args.max_seq_len_override,
    )
    profile = load(key)
    if profile is None:
        return
    applied = apply_env_defaults(profile.chosen, os.environ)
    if applied:
        logger.info(
            "ft tune profile %s: setting %s from measured evidence (%s)",
            key,
            ", ".join(applied),
            profile.evidence.date,
        )


def _run_tokenize_worker(detach: bool, **kwargs) -> None:
    """Module-level so it survives the spawn pickle; exists only to detach the group first."""
    if detach:
        _detach_process_group()
    from freetoken.tokenizer import tokenize_worker

    tokenize_worker(**kwargs)


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    if args.shell_mode:
        _detach_process_group()

    # published (not bound) here: the engine binds it after the allocator setup
    from freetoken.gpu_select import set_assigned_gpu

    # resolved UUIDs when we have them, the raw --gpu entries when NVML could not resolve them, else one CUDA ordinal per rank
    targets = args.gpu_assigned or args.gpu or tuple(str(r) for r in range(args.tp_info.size))
    set_assigned_gpu(targets[args.tp_info.rank])

    import torch
    from freetoken.scheduler import Scheduler

    if args.tp_info.is_primary():
        from freetoken.utils.progress import set_progress_sink

        set_progress_sink(lambda desc, done, total: ack_queue.put(("progress", desc, done, total)))

    with torch.inference_mode():
        try:
            scheduler = Scheduler(args)
            scheduler.sync_all_ranks()
        except Exception as exc:  # noqa: BLE001 -- surface the reason, then let it propagate
            # A startup failure (bad config, OOM, corrupt weights) would otherwise reach the
            # parent only as a dead process -> a generic "exited during load". Push the real
            # reason first so the supervisor (and the desktop failure modal) can surface it;
            # the traceback still prints and the process still exits non-zero.
            _report_startup_error(ack_queue, exc)
            raise

        if args.tp_info.is_primary():
            # Report the real per-unit cache VRAM costs (KV/expert/mamba), the device-wide free
            # VRAM, and the per-pool rebuild floors before the ready ack, so the supervisor has
            # them (and the desktop's slider bounds) by the time the gate flips. Optional +
            # best-effort: a failure here must never keep the model from serving, and older
            # consumers ignore ("meta", …).
            try:
                from freetoken.kvcache.cache_status import compute_cache_status_meta

                meta = compute_cache_status_meta(scheduler.engine)
                # the parent must not touch CUDA to learn this
                meta["gpus"] = scheduler.gpus
                ack_queue.put(("meta", meta))
            except Exception:  # noqa: BLE001 -- metadata is a nicety; readiness is not
                pass
            ack_queue.put("Scheduler is ready")
            # The supervisor stops draining ack_queue once ready, so uninstall the sink:
            # runtime cache rebuilds re-run the graph capture (which emits progress) and
            # would otherwise push onto a queue nobody reads for the server's lifetime.
            set_progress_sink(None)

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(
    run_shell: bool = False,
    argv: list[str] | None = None,
    prog: str | None = None,
) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(
        sys.argv[1:] if argv is None else argv,
        run_shell,
        prog=prog,
    )
    logger = init_logger(__name__, "initializer")

    if server_args.gpu:
        # resolve here so a typo is one clear error before any worker spawns
        from freetoken.gpu_select import resolve_gpu_uuids

        try:
            server_args = replace(server_args, gpu_assigned=resolve_gpu_uuids(server_args.gpu))
        except ValueError as exc:
            raise SystemExit(f"{prog or 'ft serve'}: error: {exc}") from exc
        logger.info(
            f"--gpu {','.join(server_args.gpu)} -> "
            f"{', '.join(server_args.gpu_assigned) if server_args.gpu_assigned else 'resolved at CUDA init (no NVML)'}"
        )

    def start_subprocess() -> "BackendHandle":
        import multiprocessing as mp

        from .supervisor import BackendHandle

        mp.set_start_method("spawn", force=True)
        detach = server_args.shell_mode  # see _detach_process_group
        apply_tuning_profile_env_gate(server_args, logger)
        apply_mtp_env_gate(server_args, logger)

        world_size = server_args.tp_info.size
        ack_queue: mp.Queue = mp.Queue()
        processes: list[mp.Process] = []

        for i in range(world_size):
            new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
            p = mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"freetoken-TP{i}-scheduler",
            )
            p.start()
            processes.append(p)

        num_tokenizers = server_args.num_tokenizer
        p = mp.Process(
            target=_run_tokenize_worker,
            kwargs={
                "detach": detach,
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "mm": server_args.mm,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="freetoken-detokenizer-0",
        )
        p.start()
        processes.append(p)
        for i in range(num_tokenizers):
            p = mp.Process(
                target=_run_tokenize_worker,
                kwargs={
                    "detach": detach,
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "mm": server_args.mm,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"freetoken-tokenizer-{i}",
            )
            p.start()
            processes.append(p)

        # Expected ready acks: 1 primary scheduler + num_tokenizers + 1 detokenizer.
        return BackendHandle(
            ack_queue=ack_queue,
            processes=processes,
            expected_acks=num_tokenizers + 2,
        )

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
