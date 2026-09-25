import pytest
import torch

from freetoken.kvcache.qsa_staging import QSAStagingAdapter


def test_qsa_page_and_group_mapping():
    adapter = QSAStagingAdapter(page_size=64, index_ratio=8, ring_capacity=16)
    entry = adapter.entry(
        page_id=3,
        request_id=2,
        logical_position=128,
        generation=7,
        device_slot=11,
    )
    assert entry.index_group == 16
    assert entry.rope_position == 128
    assert adapter.block_table([entry]) == (11,)
    assert adapter.ring_slot(2, 130) == 2


def test_qsa_mapping_rejects_stale_generation_and_conflicts():
    adapter = QSAStagingAdapter(64, 8, 8)
    entry = adapter.entry(page_id=0, request_id=0, logical_position=0, generation=2, device_slot=4)
    with pytest.raises(RuntimeError, match="stale"):
        adapter.validate_generation(entry, 1)
    with pytest.raises(ValueError, match="conflicting"):
        adapter.block_table(
            [
                entry,
                adapter.entry(
                    page_id=0, request_id=0, logical_position=0, generation=3, device_slot=5
                ),
            ]
        )


@pytest.mark.parametrize("position", [1, 63, 65])
def test_qsa_pages_start_on_boundary(position):
    with pytest.raises(ValueError, match="page boundary"):
        QSAStagingAdapter(64, 8, 8).entry(
            page_id=0,
            request_id=0,
            logical_position=position,
            generation=0,
            device_slot=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_remap_block_table_is_device_resident_and_rejects_missing_page():
    adapter = QSAStagingAdapter(64, 8, 8)
    entry = adapter.entry(
        page_id=7, request_id=0, logical_position=0, generation=2, device_slot=3
    )
    logical = torch.tensor([[7, -1]], dtype=torch.int32, device="cuda")
    mapped = adapter.remap_block_table(logical, [entry], generations={7: 2})
    assert mapped.device.type == "cuda"
    assert mapped.tolist() == [[3, -1]]
    with pytest.raises(RuntimeError, match="no staged slot"):
        adapter.remap_block_table(
            torch.tensor([[8]], dtype=torch.int32, device="cuda"), [entry]
        )
