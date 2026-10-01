"""Diagnostic benchmark launcher: compare one MTP verify with sequential RAW.

Run with the usual bench_pp_tg arguments. FREETOKEN_VERIFY_ORACLE_CYCLE selects
verify cycles, comma-separated (default 1); all run in one boot. This launcher never changes the normal CLI path.
Diagnostic throughput includes the oracle and must not be used as a baseline.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def check_verify(scheduler: Any, batch: Any, args: Any, forward: Any) -> Any:
    import torch
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.core import Batch
    from freetoken.layers.linear import _LinearTPImpl

    engine = scheduler.engine
    req = batch.reqs[0]
    model = engine.model.model
    state = scheduler._verify_state(req)
    initial = [value.cpu().clone() for value in state]
    residual = model._last_residual.clone()
    lengths = req.cached_len, req.device_len
    rng = torch.cuda.get_rng_state(engine.device)
    runner = engine.graph_runner
    graph_limit, graphs = runner.max_graph_bs, runner.verify_graphs
    runner.max_graph_bs, runner.verify_graphs = 0, {}
    captured: dict[str, dict[str, list[torch.Tensor]]] = {"raw": {}, "verify": {}}
    mode = "raw"
    hooks: list[tuple[Any, Any]] = []
    methods: dict[str, str] = {}
    fp8_rowwise_differences: list[dict[str, Any]] = []
    completed = False

    def capture(op: Any, name: str) -> None:
        original = op.forward

        def wrapped(*inputs: Any, **kwargs: Any) -> Any:
            if inputs and isinstance(inputs[0], torch.Tensor):
                captured[mode].setdefault(name + ".input", []).append(inputs[0].cpu().clone())
            if mode == "verify" and (
                os.getenv("FREETOKEN_VERIFY_ORACLE_ROWWISE_ALL") == "1"
                or (
                    os.getenv("FREETOKEN_VERIFY_ORACLE_ROWWISE_FP8") == "1"
                    and type(op.quant_method).__name__ == "Fp8TensorLinearMethod"
                    and (
                        os.getenv("FREETOKEN_VERIFY_ORACLE_BATCHED_FP8") != "1"
                        or type(op.quant_method.kernel).__name__ == "TorchFp8TensorLinearKernel"
                    )
                )
            ):
                output = torch.cat([original(row.unsqueeze(0)) for row in inputs[0]])
            else:
                output = original(*inputs, **kwargs)
            if (
                mode == "verify"
                and os.getenv("FREETOKEN_VERIFY_ORACLE_CHECK_FP8") == "1"
                and type(op.quant_method).__name__ == "Fp8TensorLinearMethod"
                and len(fp8_rowwise_differences) < 10
            ):
                actual = output.cpu().clone()
                sequential = torch.cat(
                    [original(row.unsqueeze(0)).cpu() for row in inputs[0]], dim=0
                )
                if not torch.equal(actual, sequential):
                    same_shape = actual.shape == sequential.shape
                    fp8_rowwise_differences.append(
                        {
                            "tensor": name,
                            "kernel_type": type(op.quant_method.kernel).__name__,
                            "max_abs": (
                                float((actual.float() - sequential.float()).abs().max())
                                if same_shape
                                else None
                            ),
                            "shape": [list(actual.shape), list(sequential.shape)],
                        }
                    )
            tensors = output if isinstance(output, tuple) else (output,)
            for index, value in enumerate(tensors):
                if isinstance(value, torch.Tensor):
                    captured[mode].setdefault(f"{name}.output{index}", []).append(
                        value.cpu().clone()
                    )
            return output

        hooks.append((op, original))
        op.forward = wrapped

    def visit(op: Any, name: str) -> None:
        from freetoken.layers.base import BaseOP, OPList

        if isinstance(op, _LinearTPImpl):
            methods[name] = type(op.quant_method).__name__
            capture(op, name)
        for key, child in vars(op).items():
            if key.startswith("_"):
                continue
            if isinstance(child, OPList):
                for index, layer in enumerate(child.op_list):
                    visit(layer, f"{name}.{key}.{index}")
            elif isinstance(child, BaseOP):
                visit(child, f"{name}.{key}")

    light = os.getenv("FREETOKEN_VERIFY_ORACLE_LIGHT") == "1"
    if not light:
        visit(model, "model")
    qsa_layer = int(os.getenv("FREETOKEN_VERIFY_ORACLE_QSA_LAYER", "3"))
    backend = engine.attn_backend
    qsa_orig = (backend.qsa_forward, backend._select)
    qsa_now = {"layer": -1}
    qsa_cap: dict[str, dict[str, list[torch.Tensor]]] = {"raw": {}, "verify": {}}

    def qsa_select(*args: Any, **kwargs: Any) -> Any:
        indices = qsa_orig[1](*args, **kwargs)
        if qsa_now["layer"] == qsa_layer:
            qsa_cap[mode].setdefault("indices", []).append(indices.detach().cpu().clone())
        return indices

    def qsa_forward(q: Any, k: Any, v: Any, index: Any, layer_id: int, b: Any) -> Any:
        qsa_now["layer"] = layer_id
        out = qsa_orig[0](q, k, v, index, layer_id, b)
        if layer_id == qsa_layer:
            for key, value in (("q", q), ("k", k), ("v", v), ("out", out)):
                qsa_cap[mode].setdefault(key, []).append(value.detach().cpu().clone())
        return out

    if not light:
        backend.qsa_forward = qsa_forward
        backend._select = qsa_select
    raw_logits = []
    original_input_ids = req.input_ids
    # Row-wise replay advances beyond the prompt; PLE needs the complete token
    # history while the diagnostic request is walked forward.
    req.input_ids = torch.cat(
        (req.input_ids, batch.input_ids[1:].to(req.input_ids.device))
    ).contiguous()
    try:
        for row in range(batch.input_ids.shape[0]):
            req.cached_len, req.device_len = lengths[0] + row, lengths[0] + row + 1
            raw = Batch(reqs=[req], phase="decode")
            raw.padded_reqs = [req]
            raw.input_ids = batch.input_ids[row : row + 1]
            raw.positions = batch.positions[row : row + 1]
            if batch.mrope_positions is not None:
                raw.mrope_positions = batch.mrope_positions[:, row : row + 1]
            raw.out_loc = batch.out_loc[row : row + 1]
            raw.active_table_idx = torch.tensor([req.table_idx], device=engine.device)
            raw.linear_table_idx = torch.tensor(
                [scheduler._linear_slot(req)], dtype=torch.int32, device=engine.device
            )
            raw.fla_metadata = build_fla_metadata(raw, engine.device)
            engine.attn_backend.prepare_metadata(raw)
            out = forward(raw, engine.sampler.prepare(raw))
            out.copy_done_event.synchronize()
            raw_logits.append(engine.last_batch_logits.cpu().clone())
        raw_state = [value.cpu().clone() for value in state]
        for value, saved in zip(state, initial):
            value.copy_(saved)
        req.cached_len, req.device_len = lengths
        model._last_residual = residual
        torch.cuda.set_rng_state(rng, engine.device)
        engine.attn_backend.prepare_metadata(batch)
        mode = "verify"
        out = forward(batch, args)
        out.copy_done_event.synchronize()
        expected = torch.cat(raw_logits)
        actual = engine.last_batch_logits.cpu()
        if light:
            top = expected.float().topk(2, dim=-1).values
            margins = (top[:, 0] - top[:, 1]).tolist()
            bad = [
                {"row": i, "margin": round(margins[i], 4)}
                for i, (r, v) in enumerate(
                    zip(expected.argmax(-1).tolist(), actual.argmax(-1).tolist())
                )
                if r != v
            ]
            print(
                "[oracle-light] "
                + json.dumps(
                    {"cycle": getattr(scheduler, "_raw_oracle_cycles", 0), "mismatch": bad}
                ),
                flush=True,
            )
            completed = True
            return out
        first = None
        for name, values in captured["verify"].items():
            raw_values = captured["raw"].get(name)
            if raw_values is None:
                continue
            reference, got = torch.cat(raw_values), torch.cat(values)
            if not torch.equal(reference, got):
                first = {
                    "tensor": name,
                    "quant_method": methods.get(name.rsplit(".", 1)[0]),
                    "max_abs": float((reference.float() - got.float()).abs().max()),
                    "unequal": int((reference != got).sum()),
                }
                break
        print(
            "[verify-raw-oracle] "
            + json.dumps(
                {
                    "rows": expected.shape[0],
                    "raw_tokens": expected.argmax(-1).tolist(),
                    "verify_tokens": actual.argmax(-1).tolist(),
                    "logits_equal": torch.equal(expected, actual),
                    "logits_max_abs": float((expected.float() - actual.float()).abs().max()),
                    "state_unequal_indices": [
                        index
                        for index, (value, saved) in enumerate(zip(state, raw_state))
                        if not torch.equal(value.cpu(), saved)
                    ],
                    "state_max_abs_deltas": [
                        float((value.cpu().float() - saved.float()).abs().max())
                        for value, saved in zip(state, raw_state)
                    ],
                    "first_projection_difference": first,
                    "qsa_layer": qsa_layer,
                    "qsa_compare": {
                        key: (
                            None
                            if key not in qsa_cap["verify"]
                            else {
                                "equal": torch.equal(
                                    torch.cat(qsa_cap["raw"][key]),
                                    torch.cat(qsa_cap["verify"][key]),
                                ),
                                "max_abs": float(
                                    (
                                        torch.cat(qsa_cap["raw"][key]).float()
                                        - torch.cat(qsa_cap["verify"][key]).float()
                                    )
                                    .abs()
                                    .max()
                                ),
                            }
                        )
                        for key in ("q", "k", "v", "indices", "out")
                    },
                    "fp8_rowwise_differences": fp8_rowwise_differences,
                }
            ),
            flush=True,
        )
        completed = True
        return out
    finally:
        backend.qsa_forward, backend._select = qsa_orig
        if not completed:
            for value, saved in zip(state, initial):
                value.copy_(saved)
            model._last_residual = residual
            torch.cuda.set_rng_state(rng, engine.device)
        req.cached_len, req.device_len = lengths
        req.input_ids = original_input_ids
        runner.max_graph_bs, runner.verify_graphs = graph_limit, graphs
        for op, original in hooks:
            op.forward = original


def install() -> None:
    from freetoken.scheduler.spec import SchedulerSpecMixin

    original_step = SchedulerSpecMixin.run_spec_step
    if getattr(original_step, "_raw_oracle", False):
        return
    if os.getenv("FREETOKEN_VERIFY_ORACLE_ROWWISE_ALL") == "1":
        import torch
        from freetoken.core import get_global_ctx
        from freetoken.layers.quantization.linear.base import LinearMethod

        original_linear = LinearMethod.apply

        def linear(self: Any, layer: Any, x: Any) -> Any:
            batch = get_global_ctx()._batch
            if batch is not None and batch.spec_logits_indices is not None and x.shape[0] > 1:
                return torch.cat([original_linear(self, layer, row.unsqueeze(0)) for row in x])
            return original_linear(self, layer, x)

        LinearMethod.apply = linear
    if os.getenv("FREETOKEN_VERIFY_ORACLE_TORCH_HC") == "1":
        from freetoken.core import get_global_ctx
        from freetoken.models.qwen4_exp.hc import GatedResidual

        original_mix = GatedResidual.mix
        original_combine = GatedResidual.combine

        def mix(self: Any, residual: Any) -> Any:
            batch = get_global_ctx()._batch
            if batch is not None and batch.spec_logits_indices is not None:
                return self._mix_torch(residual)
            return original_mix(self, residual)

        def combine(self: Any, residual: Any, output: Any, inject: Any) -> Any:
            batch = get_global_ctx()._batch
            if batch is not None and batch.spec_logits_indices is not None:
                return self._combine_torch(residual, output, inject)
            return original_combine(self, residual, output, inject)

        GatedResidual.mix = mix
        GatedResidual.combine = combine
    if os.getenv("FREETOKEN_VERIFY_ORACLE_BATCHED_FP8") == "1":
        from freetoken.core import get_global_ctx
        from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
        from freetoken.kernel.triton.fp8_pertensor_linear import _gemv_rows as fp8_gemv_rows
        from freetoken.layers.quantization.linear.fp8_tensor import TritonFp8TensorLinearKernel

        original_fp8_apply = TritonFp8TensorLinearKernel.apply

        def apply(self: Any, layer: Any, x: Any) -> Any:
            if get_global_ctx().batch.spec_logits_indices is None:
                return original_fp8_apply(self, layer, x)
            *lead, width = x.shape
            result = fp8_gemv_rows(
                x.reshape(-1, width), e4m3_kernel_view(layer.weight), layer.weight_scale, x.dtype
            ).reshape(*lead, layer.weight.shape[0])
            return result + layer.bias.to(result.dtype) if layer.bias is not None else result

        TritonFp8TensorLinearKernel.apply = apply
    if os.getenv("FREETOKEN_VERIFY_ORACLE_BATCHED_NVFP4") == "1":
        from freetoken.core import get_global_ctx
        from freetoken.kernel.triton.nvfp4_linear import _gemv_rows as nvfp4_gemv_rows
        from freetoken.layers.quantization.linear.nvfp4 import TritonNvfp4LinearKernel

        original_nvfp4_apply = TritonNvfp4LinearKernel.apply

        def apply(self: Any, layer: Any, x: Any) -> Any:
            if get_global_ctx().batch.spec_logits_indices is None or x.shape[0] == 1:
                return original_nvfp4_apply(self, layer, x)
            *lead, width = x.shape
            result = nvfp4_gemv_rows(
                x.reshape(-1, width),
                layer.weight.t(),
                layer.weight_scale.t(),
                layer.weight_global,
                x.dtype,
                True,
            ).reshape(*lead, layer.weight.shape[1])
            return result + layer.bias.to(result.dtype) if layer.bias is not None else result

        TritonNvfp4LinearKernel.apply = apply
    cycles = os.getenv("FREETOKEN_VERIFY_ORACLE_CYCLE", "1")
    selected = None if cycles == "all" else {int(c) for c in cycles.split(",")}

    def step(self: Any) -> bool:
        forward = self.engine.forward_batch

        def checked(batch: Any, args: Any) -> Any:
            if batch.spec_logits_indices is not None and len(batch.reqs) == 1:
                count = getattr(self, "_raw_oracle_cycles", 0) + 1
                self._raw_oracle_cycles = count
                if selected is None or count in selected:
                    return check_verify(self, batch, args, forward)
            return forward(batch, args)

        self.engine.forward_batch = checked
        try:
            return original_step(self)
        finally:
            self.engine.forward_batch = forward

    step._raw_oracle = True
    SchedulerSpecMixin.run_spec_step = step


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        from freetoken.cli import main as serve_main

        return serve_main()
    from benchmarks import bench_pp_tg

    serve_cmd = bench_pp_tg.serve_cmd

    def command(args: Any, port: int) -> list[str]:
        cmd = serve_cmd(args, port)
        cmd[cmd.index("freetoken.cli")] = "scripts.mtp_verify_oracle"
        return cmd

    bench_pp_tg.serve_cmd = command
    return bench_pp_tg.main()


if sys.argv[1:2] == ["serve"]:
    install()

if __name__ == "__main__":
    raise SystemExit(main())
