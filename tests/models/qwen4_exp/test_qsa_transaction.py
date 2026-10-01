from types import SimpleNamespace

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


def test_qsa_transaction_commit_clears_slot_for_reuse():
    backend = object.__new__(QSASparseAttnBackend)
    backend._spec_txns = {}
    backend.begin_spec_txn(4)
    backend.commit_spec_txn(4)
    assert not backend.spec_txn_active(4)
    backend.begin_spec_txn(4)
    backend.commit_spec_txn(4)
    assert not backend.spec_txn_active(4)
