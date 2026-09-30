from scripts.qualify_long_context import _serve_command


def test_quality_harness_forwards_kv_format_to_server():
    command = _serve_command("model", 262144, 1234, ["--kv-format", "turbo3"])
    assert command[-2:] == ["--kv-format", "turbo3"]
    assert command[command.index("--max-seq-len-override") + 1] == "262144"
