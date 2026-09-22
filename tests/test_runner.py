import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("test_runner", ROOT / "scripts/test-runner.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_runner_waits_for_process_group(tmp_path):
    log = tmp_path / "runner.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
    (fake_bin / "nvidia-smi").chmod(0o755)
    env = {**runner.os.environ, "PATH": f"{fake_bin}:{runner.os.environ['PATH']}"}
    result = runner.subprocess.run(
        [sys.executable, str(ROOT / "scripts/test-runner.py"), "--log", str(log), "--", "true"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert log.exists()


def test_gpu_query_is_event_driven(monkeypatch):
    seen = []

    def fake_run(*args, **kwargs):
        seen.append("timeout" in kwargs)
        return runner.subprocess.CompletedProcess(args[0], 0, b"", b"")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.gpu_pids() == []
    assert seen == [False]
