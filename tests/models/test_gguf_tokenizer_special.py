"""GGUF CONTROL / USER_DEFINED vocab entries must tokenize as one id, as llama.cpp does.

Regression: <think> (USER_DEFINED) was split into "<th" "ink" ">", so every rendered chat
prompt ending in the template's <think> opener reached the model corrupted.
"""

import freetoken.models.gguf.tokenizer as gt


def _byte_chars() -> list[str]:
    """GPT-2 byte-level alphabet (one printable char per byte value)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


def _meta() -> dict:
    byte_chars = _byte_chars()
    specials = ["<|endoftext|>", "<|im_end|>", "<think>", "</think>", "<tool_call>"]
    tokens = byte_chars + ["ab"] + specials
    types = [1] * (len(byte_chars) + 1) + [3, 3, 4, 4, 4]
    return {
        "general.architecture": "qwen4exp",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.eos_token_id": len(byte_chars) + 1,
    }


def test_control_and_user_defined_tokens_are_atomic(monkeypatch):
    meta = _meta()
    monkeypatch.setattr(gt, "load_gguf_metadata", lambda _: meta)
    monkeypatch.setattr(gt, "gguf_architecture", lambda _: "qwen4exp")
    tok = gt.load_gguf_tokenizer("unused.gguf")

    tokens, types = meta["tokenizer.ggml.tokens"], meta["tokenizer.ggml.token_type"]
    for i, t in enumerate(types):
        if t in (3, 4):
            assert tok.encode(tokens[i], add_special_tokens=False) == [i], tokens[i]

    think = tokens.index("<think>")
    ids = tok.encode("a<think>\nb", add_special_tokens=False)
    assert think in ids
    # USER_DEFINED markers stay visible text (the reasoning parser needs them); CONTROL goes.
    assert "<think>" in tok.decode(ids, skip_special_tokens=True)
