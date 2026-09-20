from freetoken.scheduler.adaptive_mtp import resolve_adaptive_k
from unittest.mock import MagicMock


def test_resolve_adaptive_k_normal_context():
    req = MagicMock()
    req.device_len = 2048
    req.remain_len = 100
    k = resolve_adaptive_k(req, max_k=2)
    assert k == 2


def test_resolve_adaptive_k_long_context():
    req = MagicMock()
    req.device_len = 80000
    req.remain_len = 100
    k = resolve_adaptive_k(req, max_k=4)
    # At 80K context, max_k=4 scales down to min(max(2, 2), 4) = 2 to protect TG
    assert k == 2


def test_resolve_adaptive_k_ultra_long_context():
    req = MagicMock()
    req.device_len = 150000
    req.remain_len = 100
    k = resolve_adaptive_k(req, max_k=4)
    # At >131K ultra long context, scales to k=1
    assert k == 1


def test_resolve_adaptive_k_remain_len_clamp():
    req = MagicMock()
    req.device_len = 1000
    req.remain_len = 2  # remain_len - 1 = 1
    k = resolve_adaptive_k(req, max_k=2)
    assert k == 1
