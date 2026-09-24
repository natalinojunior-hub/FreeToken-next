import json
import os
import subprocess
import sys


def test_token_trace_flushes_on_sigterm(tmp_path):
    path = tmp_path / "trace.jsonl"
    code = (
        "import os, signal; "
        "from freetoken.debug.token_trace import record; "
        "record(kind='smoke', token_id=7); "
        "os.kill(os.getpid(), signal.SIGTERM)"
    )
    env = dict(os.environ, FREETOKEN_TOKEN_TRACE=str(path))
    proc = subprocess.Popen([sys.executable, "-c", code], env=env)
    assert proc.wait(timeout=5) == 0
    assert json.loads(path.read_text()) == {"kind": "smoke", "token_id": 7}
