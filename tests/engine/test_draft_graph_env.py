from types import SimpleNamespace

from freetoken.engine.graph import draft_graph_enabled


def test_draft_graph_defaults_on_for_mtp_models(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DRAFT_GRAPH", raising=False)

    assert draft_graph_enabled(SimpleNamespace(mtp=object()))
    assert not draft_graph_enabled(SimpleNamespace())


def test_draft_graph_zero_override(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DRAFT_GRAPH", "0")

    assert not draft_graph_enabled(SimpleNamespace(mtp=object()))
