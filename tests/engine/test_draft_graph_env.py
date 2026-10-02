from types import SimpleNamespace

from freetoken.engine.graph import draft_graph_enabled


def test_draft_graph_defaults_on_for_mtp_models(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DRAFT_GRAPH", raising=False)

    assert draft_graph_enabled(SimpleNamespace(mtp=object()))
    assert not draft_graph_enabled(SimpleNamespace())


def test_draft_graph_zero_override(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DRAFT_GRAPH", "0")

    assert not draft_graph_enabled(SimpleNamespace(mtp=object()))


def test_draft_logits_slice_per_row_quantized_head(monkeypatch):
    import torch

    from freetoken.engine.graph import mtp_draft_logits

    monkeypatch.setenv("FREETOKEN_MTP_DRAFT_VOCAB", "5")
    method = SimpleNamespace(apply=lambda layer, x: (x @ layer.weight.t()) * layer.weight_scale)
    head = SimpleNamespace(
        weight=torch.randn(9, 4),
        weight_scale=torch.rand(9),
        bias=None,
        tied_embedding=None,
        quant_method=method,
        _fp8_scale_segments=[(0, 3), (3, 9)],
    )
    head.forward = lambda x: method.apply(head, x)
    model = SimpleNamespace(mtp=SimpleNamespace(to_head=lambda r: r), lm_head=head)
    x = torch.randn(1, 4)

    logits = mtp_draft_logits(model, x)
    assert logits.shape == (1, 5)
    torch.testing.assert_close(logits, head.forward(x)[:, :5])
    assert head._draft_head_rows[1]._fp8_scale_segments == [(0, 3), (3, 5)]

    head.weight_scale = torch.rand(9, 2)  # block-scaled layouts are not row-sliceable
    del head._draft_head_rows
    from freetoken.engine.graph import _draft_head_rows

    assert _draft_head_rows(head, 5) is None
