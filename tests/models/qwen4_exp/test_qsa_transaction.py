from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.qsa_sparse import QSASparseAttnBackend


def test_qsa_transaction_restores_index_kv_and_rope_rows():
    class Pool:
        compressed = True

        def _dense(self, layer_id):
            return layer_id

    layers, tokens, dim = 2, 16, 4
    pool = Pool()
    pool._k_codes = torch.arange(layers * tokens * dim, dtype=torch.int8).reshape(
        layers, tokens, dim
    )
    pool._k_norm = torch.arange(layers * tokens, dtype=torch.float32).reshape(layers, tokens)
    pool._v_codes = pool._k_codes.clone()
    pool._v_norm = pool._k_norm.clone()
    cache = SimpleNamespace(
        _pool=pool,
        num_device_pages=2,
        cmp_k_cache=lambda slot: cmp[slot],
        k_slab=lambda layer: (pool._k_codes[layer], pool._k_norm[layer]),
        v_slab=lambda layer: (pool._v_codes[layer], pool._v_norm[layer]),
        host_turbo=lambda layer: None,
        host_kv=lambda layer: None,
        _rope_positions=torch.arange(tokens * 3, dtype=torch.int32).reshape(tokens, 3),
    )
    cmp = torch.arange(layers * tokens * dim, dtype=torch.float32).reshape(layers, tokens, dim)
    backend = object.__new__(QSASparseAttnBackend)
    backend.kvcache = cache
    backend.page_size = 4
    backend._spec_txns = {}

    before_codes = pool._k_codes.clone()
    before_cmp = cmp.clone()
    before_rope = cache._rope_positions.clone()
    backend.begin_spec_txn(7)
    loc = torch.tensor([2, 3], dtype=torch.int32)
    backend._txn_save_rows(7, 1, loc)
    backend._txn_save_cmp(7, 1, torch.tensor([2], dtype=torch.int32))
    backend._txn_save_rope(7, loc)
    pool._k_codes[1, loc] = -1
    cmp[1, 2] = -1
    cache._rope_positions[loc] = -1
    backend.rollback_spec_txn(7)

    assert torch.equal(pool._k_codes, before_codes)
    assert torch.equal(cmp, before_cmp)
    assert torch.equal(cache._rope_positions, before_rope)


def test_qsa_transaction_restores_page_major_bf16_device_and_host_rows():
    class Pool:
        compressed = False

        def _dense(self, layer_id):
            return layer_id

    pool = Pool()
    pool._k = torch.arange(2 * 2 * 4 * 1 * 3, dtype=torch.float32).reshape(2, 2, 4, 1, 3)
    pool._v = pool._k + 100
    host_k = torch.arange(2 * 2 * 4 * 1 * 3, dtype=torch.float32).reshape(2, 2, 4, 1, 3) + 200
    host_v = host_k + 100
    cache = SimpleNamespace(
        _pool=pool,
        num_device_pages=2,
        k_cache=lambda layer: pool._k[layer],
        v_cache=lambda layer: pool._v[layer],
        host_turbo=lambda layer: None,
        host_kv=lambda layer: (host_k[layer], host_v[layer]),
        cmp_k_cache=lambda slot: torch.zeros(16, 3),
        _rope_positions=None,
    )
    backend = object.__new__(QSASparseAttnBackend)
    backend.kvcache = cache
    backend.page_size = 4
    backend._spec_txns = {}
    dev_before, host_before = pool._k.clone(), host_k.clone()
    backend.begin_spec_txn(3)
    backend._txn_save_rows(3, 0, torch.tensor([1, 2 * 4 + 1]))
    backend._txn_save_rows(3, 0, torch.tensor([8 + 1]))
    pool._k[0, 0, 1].fill_(-1)
    host_k[0, 0, 1].fill_(-2)
    backend.rollback_spec_txn(3)
    assert torch.equal(pool._k, dev_before)
    assert torch.equal(host_k, host_before)

    # A pre-draft journal has no active transaction yet, including its host KV rows.
    backend._spec_preimages = {}
    backend._idx_slot = {0: 0}
    backend._physical_loc = lambda loc: loc
    backend.ratio = 4
    cache.cmp_scratch_base = 8
    backend.snapshot_spec_preimage(3, torch.tensor([1, 9]))
    pool._k[0, 0, 1].fill_(-3)
    host_k[0, 0, 1].fill_(-4)
    backend.abort_spec_txn(3)
    assert torch.equal(pool._k, dev_before)
    assert torch.equal(host_k, host_before)
    assert not backend._spec_txns and not backend._spec_preimages


def test_qsa_transaction_commit_clears_slot_for_reuse():
    backend = object.__new__(QSASparseAttnBackend)
    backend._spec_txns = {}
    backend._spec_preimages = {4: {"cmp": {}, "kv": {}, "rope": {}}}
    backend.begin_spec_txn(4)
    backend.commit_spec_txn(4)
    assert not backend.spec_txn_active(4)
    assert not backend._spec_preimages
    backend.begin_spec_txn(4)
    backend.commit_spec_txn(4)
    assert not backend.spec_txn_active(4)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("activate", [False, True])
def test_qsa_abort_restores_first_pre_draft_preimage(compressed, activate):
    """Draft and verify failures restore the same original KV/index/RoPE image."""
    pool = SimpleNamespace(compressed=compressed, _dense=lambda layer: layer)
    keys = torch.arange(16, dtype=torch.float32).view(1, 4, 4, 1, 1)
    values = keys + 100
    cmp = torch.arange(16, dtype=torch.float32).view(1, 16, 1)
    rope = torch.arange(16, dtype=torch.int32).view(16, 1)
    if compressed:
        pool._k_codes = keys.view(1, 16, 1).clone()
        pool._v_codes = values.view(1, 16, 1).clone()
        pool._k_norm = torch.ones(1, 16)
        pool._v_norm = torch.ones(1, 16)
        state = (pool._k_codes, pool._v_codes, pool._k_norm, pool._v_norm, cmp, rope)
    else:
        state = (keys, values, cmp, rope)
    backend = object.__new__(QSASparseAttnBackend)
    backend._spec_txns = {}
    backend._spec_preimages = {}
    backend._idx_slot = {0: 0}
    backend.page_size = backend.ratio = 4
    backend._physical_loc = lambda loc: loc
    backend._fast_layer_index = lambda device: (torch.tensor([0]), torch.tensor([0]))
    backend.kvcache = SimpleNamespace(
        _pool=pool,
        num_device_pages=4,
        cmp_scratch_base=8,
        _cmp_k_buffer=cmp,
        cmp_k_cache=lambda slot: cmp[slot],
        k_cache=lambda layer: keys[layer],
        v_cache=lambda layer: values[layer],
        host_kv=lambda layer: None,
        _rope_positions=rope,
    )
    before = tuple(t.clone() for t in state)
    loc = torch.tensor([2, 3])
    backend.snapshot_spec_preimage(0, loc)
    first = backend._spec_preimages[0]
    for t in state:
        # Only journaled rows are allowed to change during a speculative window.
        if t is cmp:
            t[:, [0, 8]] = -1
        elif t is rope:
            t[loc] = -1
        elif compressed:
            t[:, loc] = -1
        else:
            t[:, 0, loc] = -1
    backend.snapshot_spec_preimage(0, loc)
    assert backend._spec_preimages[0] is first
    if activate:
        # Ordinary rollback between draft and verify must leave the preimage available.
        backend.rollback_spec_txn(0)
        assert backend._spec_preimages[0] is first
        backend.prepare_spec_txn(0, loc)
        assert all(torch.equal(t, expected) for t, expected in zip(state, before))
        for t in state:
            if t is cmp:
                t[:, [0, 8]] = -2
            elif t is rope:
                t[loc] = -2
            elif compressed:
                t[:, loc] = -2
            else:
                t[:, 0, loc] = -2
    backend.abort_spec_txn(0)
    assert all(torch.equal(t, expected) for t, expected in zip(state, before))
    assert not backend._spec_txns and not backend._spec_preimages
    backend.abort_spec_txn(0)  # idempotent cleanup before table reuse


@pytest.mark.parametrize("active", [False, True])
def test_qsa_restore_oom_preserves_journal_for_retry(active):
    backend = object.__new__(QSASparseAttnBackend)
    journal = {"cmp": {}, "kv": {}, "rope": {}}
    backend._spec_txns = {0: journal} if active else {}
    backend._spec_preimages = {} if active else {0: journal}
    attempts = []

    def restore(txn):
        assert txn is journal
        attempts.append(txn)
        if len(attempts) == 1:
            raise torch.OutOfMemoryError("restore injection")

    backend._restore_txn = restore
    with pytest.raises(torch.OutOfMemoryError, match="restore injection"):
        backend.abort_spec_txn(0)
    assert (backend._spec_txns if active else backend._spec_preimages)[0] is journal
    backend.abort_spec_txn(0)
    assert len(attempts) == 2
    assert not backend._spec_txns and not backend._spec_preimages
