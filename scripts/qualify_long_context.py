#!/usr/bin/env python3
"""Run the archived long-context needle and short-task usage gates on this worktree."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path


WORKTREE = Path(__file__).resolve().parents[1]
DEFAULT_NEEDLE = Path(
    "/models/desenvolvimento/old/freetoken-next/external/ft-campaign2/campaign15/prompt256k-needle.txt"
)
DEFAULT_QUALITY_SOURCE = Path(
    "/models/desenvolvimento/old/freetoken-next/external/ft-campaign2/campaign13/quality_eval.py"
)
NEEDLE = "7391-KESTREL"
QUALITY_TASK_NAMES = (
    "math1",
    "math2",
    "math3",
    "math4",
    "math5",
    "math6",
    "code1",
    "code2",
    "code3",
    "code4",
    "code5",
    "json1",
    "json2",
    "needle",
    "pt1",
    "pt2",
    "fact1",
    "fact2",
    "logic1",
    "sort1",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _runtime_source_sha256(worktree: Path) -> tuple[str, list[str]]:
    listed = subprocess.run(
        ["git", "-C", str(worktree), "ls-files", "-co", "--exclude-standard", "python"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    digest = hashlib.sha256()
    for relative in sorted(listed):
        path = worktree / relative
        if path.is_file():
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    dirty = subprocess.run(
        ["git", "-C", str(worktree), "status", "--short", "--", "python"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return digest.hexdigest(), dirty


def _model_identity(model_path: str, worktree: Path) -> tuple[str, tuple[str, ...]]:
    """Read model identity through the worktree's normal HF/GGUF config loader."""
    python_root = str((worktree / "python").resolve())
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    from freetoken.utils import cached_load_hf_config

    config = cached_load_hf_config(model_path)
    model_type = str(getattr(config, "model_type", "") or "")
    architectures = tuple(getattr(config, "architectures", ()) or ())
    return model_type, architectures


def _load_quality_tasks(path: Path):
    """Load only the archived task data and stdlib validators, never its launcher."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    wanted = {"NEEDLE_FILLER", "needle", "needle_ans", "TASKS"}
    functions = {"_needle_prompt", "_code_check", "_num", "_json_eq"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports = [
                alias
                for alias in node.names
                if alias.name in {"json", "os", "re", "subprocess", "sys", "tempfile"}
            ]
            if imports:
                selected.append(ast.copy_location(ast.Import(names=imports), node))
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            names = {
                child.id
                for target in node.targets
                for child in ast.walk(target)
                if isinstance(child, ast.Name)
            }
            if names & wanted:
                selected.append(node)
    namespace: dict = {"__file__": str(path), "__name__": "_archived_quality_data"}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)

    def numeric_check(answer: str, expected: float) -> bool:
        numbers = namespace["re"].findall(r"-?\d+(?:[.,]\d+)?", answer)
        return bool(numbers) and abs(float(numbers[-1].replace(",", ".")) - expected) < 1e-6

    namespace["_num"] = numeric_check
    tasks = namespace.get("TASKS")
    if not isinstance(tasks, list) or tuple(row[0] for row in tasks) != QUALITY_TASK_NAMES:
        raise ValueError(f"{path} does not contain the expected 20 archived quality tasks")
    return tasks, source


def _self_test(quality_source: Path) -> None:
    tasks, _ = _load_quality_tasks(quality_source)
    good = {
        "math1": "36",
        "math2": "346",
        "math3": "72",
        "math4": "29",
        "math5": "2",
        "math6": "3",
        "code1": "```python\nimport re\ndef is_palindrome(s):\n x=re.sub(r'[^a-z0-9]','',s.lower()); return x==x[::-1]\n```",
        "code2": "```python\ndef merge_intervals(xs):\n out=[]\n for a,b in sorted(xs):\n  if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)\n  else: out.append([a,b])\n return out\n```",
        "code3": "```python\ndef roman_to_int(s):\n v={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}; n=0\n for i,c in enumerate(s): n += -v[c] if i+1<len(s) and v[c]<v[s[i+1]] else v[c]\n return n\n```",
        "code4": "```python\nfrom collections import Counter\ndef top_k_words(text,k): return [word for word,count in sorted(Counter(text.lower().split()).items(),key=lambda x:(-x[1],x[0]))[:k]]\n```",
        "code5": "```python\ndef fib(n):\n a,b=0,1\n for _ in range(n): a,b=b,a+b\n return a\n```",
        "json1": '{"name":"Ana Souza","age":34,"city":"Curitiba"}',
        "json2": '{"primes":[2,3,5,7,11,13,17,19]}',
        "needle": "The code is 7314-KX.",
        "pt1": "Canberra",
        "pt2": "vermelho, verde, azul",
        "fact1": "iron",
        "fact2": "1989",
        "logic1": "yes",
        "sort1": "3, 7, 19, 21, 42, 88",
    }
    assert len(tasks) == 20
    for name, _prompt, check in tasks:
        assert check(good[name]), f"positive validator fixture failed: {name}"
        assert not check("incorrect"), f"negative validator fixture passed: {name}"
    checks = {name: check for name, _prompt, check in tasks}
    assert checks["math4"]("Maria: 29. 29")
    assert not checks["math4"]("29.29")
    assert NEEDLE in "answer: " + NEEDLE
    assert NEEDLE not in "wrong sentinel"


def _serve_helpers(worktree: Path):
    bench_dir = worktree / "benchmarks"
    if not (bench_dir / "bench_pp_tg.py").is_file():
        raise FileNotFoundError(f"bench_pp_tg.py not found under worktree {worktree}")
    sys.path.insert(0, str(bench_dir))
    import bench_pp_tg  # noqa: PLC0415

    return bench_pp_tg


def _chat(origin: str, model_id: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
    }
    request = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if proc.poll() is None:
        proc.wait()


def _serve_command(model: str, context: int, port: int, extra: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "freetoken.cli",
        "serve",
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-seq-len-override",
        str(context),
        *extra,
    ]


def _run_suite(args, suite: str, bench, records: list[dict], log_path: Path) -> None:
    port = bench.free_port()
    origin = f"http://127.0.0.1:{port}"
    max_tokens = args.needle_max_tokens if suite == "needle" else args.max_tokens
    command = _serve_command(args.model, args.context, port, args.serve_extra)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=args.worktree,
            env={**os.environ, "PYTHONPATH": args.pythonpath},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pump = threading.Thread(target=bench.pump_output, args=(proc.stdout, log_file), daemon=True)
        pump.start()
        try:
            bench.wait_ready(origin, proc, str(log_path), args.server_timeout)
            model_id = bench.get_json(f"{origin}/v1/models")["data"][0]["id"]
            if suite == "needle":
                prompt = args.needle_file.read_text(errors="replace")
                record = {
                    "kind": "needle",
                    "model": args.model,
                    "context": args.context,
                    "prompt_file": str(args.needle_file),
                    "prompt_sha256": _sha256(prompt),
                    "needle": NEEDLE,
                    "max_tokens": max_tokens,
                    "log_path": str(log_path),
                    "started_at": time.time(),
                }
                started = time.monotonic()
                try:
                    response = _chat(origin, model_id, prompt, max_tokens, args.request_timeout)
                    message = response["choices"][0]["message"]
                    text = (
                        (message.get("reasoning_content") or "")
                        + "\n"
                        + (message.get("content") or "")
                    )
                    usage = response.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    record.update(
                        {
                            "pass": NEEDLE in text,
                            "completion": text,
                            "finish_reason": response["choices"][0].get("finish_reason"),
                            "usage": usage,
                            "context_fit": (
                                prompt_tokens is not None
                                and completion_tokens is not None
                                and prompt_tokens + completion_tokens <= args.context
                            ),
                            "latency_seconds": time.monotonic() - started,
                        }
                    )
                except (
                    Exception
                ) as exc:  # Capture request failures as a failed machine-readable gate.
                    record.update({"pass": False, "context_fit": False, "error": repr(exc)})
                records.append(record)
            else:
                tasks, _source = _load_quality_tasks(args.quality_source)
                for name, prompt, check in tasks:
                    started = time.monotonic()
                    row = {
                        "kind": "usage_task",
                        "task": name,
                        "model": args.model,
                        "context": args.context,
                        "max_tokens": max_tokens,
                        "log_path": str(log_path),
                    }
                    try:
                        response = _chat(origin, model_id, prompt, max_tokens, args.request_timeout)
                        message = response["choices"][0]["message"]
                        text = message.get("content") or ""
                        usage = response.get("usage", {})
                        row.update(
                            {
                                "pass": bool(check(text)),
                                "completion": text,
                                "finish_reason": response["choices"][0].get("finish_reason"),
                                "usage": usage,
                                "context_fit": (
                                    usage.get("prompt_tokens") is not None
                                    and usage.get("completion_tokens") is not None
                                    and usage["prompt_tokens"] + usage["completion_tokens"]
                                    <= args.context
                                ),
                                "latency_seconds": time.monotonic() - started,
                            }
                        )
                    except Exception as exc:  # Keep running the remaining quality checks.
                        row.update({"pass": False, "context_fit": False, "error": repr(exc)})
                    records.append(row)
        finally:
            _stop_server(proc)
            pump.join(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--mode", choices=("all", "needle", "usage"), default="all")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--needle-file", type=Path, default=DEFAULT_NEEDLE)
    parser.add_argument("--quality-source", type=Path, default=DEFAULT_QUALITY_SOURCE)
    parser.add_argument("--worktree", type=Path, default=WORKTREE)
    parser.add_argument(
        "--pythonpath", help="server PYTHONPATH; default is worktree/python:worktree/benchmarks"
    )
    parser.add_argument("--needle-max-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--server-timeout", type=float, default=1200)
    parser.add_argument(
        "--serve-extra",
        action="append",
        default=[],
        help="extra serve args, e.g. --serve-extra=--kv-format --serve-extra=turbo3",
    )
    parser.add_argument("--request-timeout", type=float, default=900)
    args = parser.parse_args()

    if args.self_test:
        _self_test(args.quality_source)
        print("long-context quality self-test passed (20 archived validators)")
        return 0
    if not args.model or not args.out:
        parser.error("--model and --out are required unless --self-test is used")
    args.worktree = args.worktree.resolve()
    try:
        model_type, architectures = _model_identity(args.model, args.worktree)
    except Exception as exc:
        parser.error(f"could not read model config for {args.model}: {exc}")
    if model_type not in {"qwen4_exp", "qwen4exp", "qwen3_5_moe", "qwen35moe"} and not {
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpGGUFForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    }.intersection(architectures):
        parser.error(
            "model config is outside the supported Qwen3.8 family "
            f"(model_type={model_type!r}, architectures={architectures!r})"
        )
    if (
        args.context <= 64
        or not 0 < args.needle_max_tokens <= args.context - 64
        or not 0 < args.max_tokens <= args.context - 64
    ):
        parser.error("context and completion limits must be positive and fit the selected context")
    args.needle_file = args.needle_file.resolve()
    args.quality_source = args.quality_source.resolve()
    if args.pythonpath is None:
        args.pythonpath = os.pathsep.join(
            (str(args.worktree / "python"), str(args.worktree / "benchmarks"))
        )
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tasks, quality_text = _load_quality_tasks(args.quality_source)
    bench = _serve_helpers(WORKTREE)
    runtime_source_sha256, runtime_dirty_paths = _runtime_source_sha256(args.worktree)
    records = [
        {
            "kind": "metadata",
            "schema": "freetoken-long-quality-v1",
            "model": args.model,
            "context": args.context,
            "mode": args.mode,
            "worktree": str(args.worktree),
            "runtime_git_sha": subprocess.run(
                ["git", "-C", str(args.worktree), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or "unknown",
            "runtime_source_sha256": runtime_source_sha256,
            "runtime_dirty_paths": runtime_dirty_paths,
            "harness_source_sha256": _sha256(Path(__file__).read_text(encoding="utf-8")),
            "pythonpath": args.pythonpath,
            "serve_args": [f"--max-seq-len-override={args.context}", *args.serve_extra],
            "needle_file": str(args.needle_file),
            "quality_source": str(args.quality_source),
            "quality_source_sha256": _sha256(quality_text),
            "quality_task_names": [row[0] for row in tasks],
            "artifact_path": str(args.out),
        }
    ]
    suites = ("needle", "usage") if args.mode == "all" else (args.mode,)
    for suite in suites:
        log_path = args.out.with_name(f"{args.out.stem}-{suite}.server.log")
        try:
            _run_suite(args, suite, bench, records, log_path)
        except (Exception, SystemExit) as exc:
            records.append(
                {
                    "kind": "runner_error",
                    "suite": suite,
                    "model": args.model,
                    "context": args.context,
                    "log_path": str(log_path),
                    "error": repr(exc),
                }
            )

    needle_rows = [row for row in records if row["kind"] == "needle"]
    usage_rows = [row for row in records if row["kind"] == "usage_task"]
    gates = {
        "needle_pass": bool(needle_rows)
        and all(row.get("pass") and row.get("context_fit") for row in needle_rows),
        "usage_20_of_20": len(usage_rows) == 20
        and all(row.get("pass") and row.get("context_fit") for row in usage_rows),
    }
    if args.mode != "all":
        gates = {
            k: v
            for k, v in gates.items()
            if k == ("needle_pass" if args.mode == "needle" else "usage_20_of_20")
        }
    passed = all(gates.values())
    records.append(
        {
            "kind": "summary",
            "model": args.model,
            "context": args.context,
            "needle_pass": sum(bool(row.get("pass")) for row in needle_rows),
            "usage_pass": sum(bool(row.get("pass")) for row in usage_rows),
            "usage_total": len(usage_rows),
            "needle_usage_total": len(needle_rows),
            "needle_completion_tokens": [
                row.get("usage", {}).get("completion_tokens") for row in needle_rows
            ],
            "gates": gates,
            "passed": passed,
            "runner_errors": [row for row in records if row["kind"] == "runner_error"],
            "artifact_path": str(args.out),
        }
    )
    with args.out.open("w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(records[-1], ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
