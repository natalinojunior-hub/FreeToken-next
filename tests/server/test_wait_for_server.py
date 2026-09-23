import os
import subprocess


def test_wait_for_server_accepts_health_ok(tmp_path):
    curl = tmp_path / "curl"
    curl.write_text('#!/bin/sh\nprintf \'{"status":"ok"}\\n\'\n')
    curl.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}

    result = subprocess.run(
        ["bash", "scripts/wait-for-server.sh", "8123", str(os.getpid())],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
