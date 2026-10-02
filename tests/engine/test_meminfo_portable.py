import builtins
import ctypes
from types import SimpleNamespace


def test_meminfo_falls_back_to_global_memory_status_without_proc(monkeypatch):
    """Native Windows has no /proc/meminfo; the RAM-tier budget reads GlobalMemoryStatusEx."""
    from freetoken.engine import engine

    real_open = builtins.open

    def no_proc(path, *a, **k):
        if str(path) == "/proc/meminfo":
            raise FileNotFoundError(path)
        return real_open(path, *a, **k)

    def fill(ref):
        ref._obj.ullTotalPhys = 32 << 30
        ref._obj.ullAvailPhys = 20 << 30
        return 1

    monkeypatch.setattr(builtins, "open", no_proc)
    monkeypatch.setattr(
        ctypes,
        "windll",
        SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=fill)),
        raising=False,
    )
    assert engine._meminfo() == {"MemTotal": 32 << 30, "MemAvailable": 20 << 30}
