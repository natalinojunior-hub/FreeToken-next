from __future__ import annotations

import os
import socket

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


def _pick_free_port() -> int:
    """Ephemeral port from the OS (bind port 0, let it choose, then release)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _zmq_endpoint(n: int, suffix: str, ports: dict[int, int]) -> str:
    """ZeroMQ endpoint for slot ``n``. POSIX keeps the ``ipc:///tmp`` transport; native
    Windows has no ``ipc`` transport, so each slot gets ``tcp://127.0.0.1:<port>`` with a
    free port picked once (bind port 0) and cached in ``ports`` so bind/connect agree."""
    if os.name != "nt":
        return f"ipc:///tmp/freetoken_{n}" + suffix
    port = ports.get(n)
    if port is None:
        port = _pick_free_port()
        ports[n] = port
    return f"tcp://127.0.0.1:{port}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)
    # Windows-only: per-slot tcp ports picked at construction, so the copy pickled into
    # every process carries the same endpoints (POSIX keeps ipc:// and leaves it empty).
    _zmq_tcp_ports: dict[int, int] = field(
        default_factory=lambda: {n: _pick_free_port() for n in range(5)} if os.name == "nt" else {},
        repr=False,
    )

    @property
    def zmq_backend_addr(self) -> str:
        return _zmq_endpoint(0, self._unique_suffix, self._zmq_tcp_ports)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return _zmq_endpoint(1, self._unique_suffix, self._zmq_tcp_ports)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return _zmq_endpoint(2, self._unique_suffix, self._zmq_tcp_ports)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
