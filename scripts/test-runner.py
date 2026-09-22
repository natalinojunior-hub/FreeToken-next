#!/usr/bin/env python3
import argparse
import ctypes
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TMPDIR = "/models/desenvolvimento/tmp"
PIDFD_SEND_SIGNAL = 424


class RunnerInterrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def gpu_pids():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip() or "nvidia-smi failed")
    pids = []
    for line in result.stdout.decode(errors="replace").splitlines():
        value = line.strip()
        if value:
            pids.append(int(value))
    return sorted(set(pids))


def same_user(pid):
    return os.stat(f"/proc/{pid}").st_uid == os.getuid()


def send_pidfd(pid, sig):
    fd = os.pidfd_open(pid)
    try:
        if hasattr(os, "pidfd_send_signal"):
            os.pidfd_send_signal(fd, sig)
            return
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.syscall(PIDFD_SEND_SIGNAL, fd, sig, 0, 0)
        if result:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    finally:
        os.close(fd)


def wait_pidfd(pid):
    fd = os.pidfd_open(pid)
    try:
        select.select([fd], [], [])
        return True
    finally:
        os.close(fd)


def clean_gpu():
    pids = gpu_pids()
    if not pids:
        return
    print(
        f"[WARN] terminating same-user NVIDIA compute PIDs: {', '.join(map(str, pids))}", flush=True
    )
    for pid in pids:
        if not same_user(pid):
            raise RuntimeError(f"refusing foreign-user GPU PID {pid}")
    for pid in pids:
        try:
            send_pidfd(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        wait_pidfd(pid)
    remaining = gpu_pids()
    if remaining:
        raise RuntimeError(f"GPU compute PIDs remain: {', '.join(map(str, remaining))}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required")
    return args


def main():
    args = parse_args()
    tmpdir = Path(os.environ.get("TMPDIR", DEFAULT_TMPDIR))
    if str(tmpdir) == "/tmp" or str(tmpdir).startswith("/tmp/"):
        print("[FAIL] TMPDIR must be persistent disk, not /tmp", file=sys.stderr)
        return 2
    tmpdir.mkdir(parents=True, exist_ok=True)
    log_path = (
        args.log or tmpdir / f"test-runner-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"
    )
    process = None
    interrupted = None
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(
            signum, lambda value, _frame: (_ for _ in ()).throw(RunnerInterrupted(value))
        )
    try:
        clean_gpu()
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                args.command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
            process.wait()
            return process.returncode
    except RunnerInterrupted as error:
        interrupted = error.signum
        return 128 + error.signum
    except Exception as error:
        print(f"[FAIL] {error}; log: {log_path}", file=sys.stderr, flush=True)
        return 2
    finally:
        if process is not None and process.poll() is None:
            sig = interrupted or signal.SIGTERM
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            wait_pidfd(process.pid)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        print(f"[status] log: {log_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
