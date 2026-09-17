"""FTW replay: dropping entries by name before their bytes are read, and the vision-tower presence check."""

import os

import torch

from freetoken.checkpoint.ftw import FTWReader, FTWWriter, ftw_tensor_names, iter_ftw_weights
from freetoken.models.weight import ftw_lacks_vision, load_weight


def _write_ftw(out_dir, names):
    writer = FTWWriter(str(out_dir))
    tensors = {name: torch.full((4, 8), float(i), dtype=torch.bfloat16) for i, name in enumerate(names)}
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor)
    writer.finalize({})
    return tensors


def test_keep_drops_entries_before_their_bytes_are_read(tmp_path, monkeypatch):
    tensors = _write_ftw(tmp_path, ["model.a.weight", "visual.b.weight", "model.c.weight"])
    read = []
    original = FTWReader.read_into

    def spy(self, dest, entry, **kwargs):
        read.append(entry["name"])
        return original(self, dest, entry, **kwargs)

    monkeypatch.setattr(FTWReader, "read_into", spy)
    got = dict(iter_ftw_weights(str(tmp_path), keep=lambda name: not name.startswith("visual.")))
    assert list(got) == ["model.a.weight", "model.c.weight"]
    assert read == ["model.a.weight", "model.c.weight"]
    for name, tensor in got.items():
        assert torch.equal(tensor, tensors[name])


def test_no_keep_replays_every_entry(tmp_path):
    tensors = _write_ftw(tmp_path, ["model.a.weight", "visual.b.weight"])
    got = dict(iter_ftw_weights(str(tmp_path)))
    assert list(got) == list(tensors)
    assert torch.equal(got["visual.b.weight"], tensors["visual.b.weight"])


def test_load_weight_text_only_skips_the_tower(tmp_path):
    names = ["model.a.weight", "visual.b.weight", "vision_tower.c.weight"]
    _write_ftw(tmp_path, names)
    cpu = torch.device("cpu")
    assert [n for n, _ in load_weight(str(tmp_path), cpu, include_vision=False)] == ["model.a.weight"]
    assert [n for n, _ in load_weight(str(tmp_path), cpu)] == names


def test_ftw_lacks_vision(tmp_path):
    _write_ftw(tmp_path / "text", ["model.a.weight"])
    _write_ftw(tmp_path / "vl", ["model.a.weight", "visual.b.weight"])
    assert ftw_lacks_vision(str(tmp_path / "text"))
    assert not ftw_lacks_vision(str(tmp_path / "vl"))
    assert not ftw_lacks_vision(str(tmp_path))
    assert ftw_tensor_names(str(tmp_path / "vl"), "weight") == ["model.a.weight", "visual.b.weight"]


def test_ftw_writer_drops_completed_shards_from_page_cache(tmp_path, monkeypatch):
    calls = []

    def fadvise(fd, offset, length, advice):
        calls.append((fd, offset, length, advice))

    monkeypatch.setattr(os, "posix_fadvise", fadvise)
    writer = FTWWriter(str(tmp_path), shard_limit=4096)
    writer.add_tensor("a", torch.zeros(2048, dtype=torch.uint8))
    writer.add_tensor("b", torch.zeros(2048, dtype=torch.uint8))
    writer.finalize({})

    assert len(calls) == 4
    assert all(offset == 0 and length == 0 for _, offset, length, _ in calls)


def test_ftw_writer_resumes_after_an_interrupted_shard(tmp_path):
    first = FTWWriter(str(tmp_path), shard_limit=4096)
    first.add_tensor("a", torch.ones(2048, dtype=torch.uint8))
    first._f.flush()
    first._f.close()
    with open(tmp_path / "freetoken-00000.ftw", "ab") as f:
        f.write(b"interrupted")

    resumed = FTWWriter(str(tmp_path), shard_limit=4096)
    resumed.add_tensor("b", torch.full((2048,), 2, dtype=torch.uint8))
    resumed.finalize({})

    assert ftw_tensor_names(str(tmp_path)) == ["a", "b"]
    got = dict(iter_ftw_weights(str(tmp_path)))
    assert torch.equal(got["a"], torch.ones(2048, dtype=torch.uint8))
    assert torch.equal(got["b"], torch.full((2048,), 2, dtype=torch.uint8))
