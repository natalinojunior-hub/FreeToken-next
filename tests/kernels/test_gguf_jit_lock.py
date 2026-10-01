"""A JIT build killed mid-compile must not hang every later boot on torch's file baton."""

from __future__ import annotations

import os
import time

import pytest

from freetoken.kernel import gguf


@pytest.fixture
def lock(tmp_path, monkeypatch):
    path = tmp_path / "lock"
    path.write_text("")
    monkeypatch.setattr(
        "torch.utils.cpp_extension._get_build_directory", lambda name, verbose: str(tmp_path)
    )
    return path


def _age(path, seconds):
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_stale_lock_without_a_compiler_is_removed(lock, monkeypatch):
    monkeypatch.setattr(gguf, "_compiler_running", lambda: False)
    _age(lock, 300)
    gguf._clear_stale_build_lock("x")
    assert not lock.exists()


def test_lock_kept_while_a_compiler_runs(lock, monkeypatch):
    monkeypatch.setattr(gguf, "_compiler_running", lambda: True)
    _age(lock, 300)
    gguf._clear_stale_build_lock("x")
    assert lock.exists()


def test_fresh_lock_is_kept(lock, monkeypatch):
    monkeypatch.setattr(gguf, "_compiler_running", lambda: False)
    gguf._clear_stale_build_lock("x")
    assert lock.exists()
