"""The committed recording of a real local-model run (Qwen2.5 3B / Qwen2.5-VL 3B) must
replay to the intended outcome. Machine-independent assertions only."""
import pytest

from procureiq.agent import graph

pytestmark = pytest.mark.recorded


def test_recorded_model_run_outcome(monkeypatch):
    monkeypatch.setenv("OCR", "off")             # recorded on a machine without Tesseract
    st = graph.run()
    if not any(s.startswith("recorded") for s in st["sources"].values()):
        pytest.skip("no recorded model run committed yet")
    status = {q["vendor_id"]: q for q in st["evaluation"]["quotes"]}
    assert status["A"]["status"] == "COMPLIANT" and status["A"]["rank"] == 1
    assert status["B"]["status"] == "EXCLUDED"
    assert any(c["key"] == "conductor_material" and c["label"] == "SUBSTITUTED" for c in status["B"]["claims"])
    assert status["C"]["status"] != "EXCLUDED"
    assert len(st["requirements"]["requirements"]) == 12


def test_recorded_agent_decision_replays(monkeypatch):
    """The local model's own clarification decision replays exactly, and C ends compliant."""
    monkeypatch.setenv("OCR", "off")
    st = graph.run()
    if not st["sources"].get("agent_r1", "").startswith("recorded"):
        pytest.skip("no recorded agent run committed yet")
    first = st["agent_trace"][0]
    assert first["tool"] == "send_clarification" and first["args"]["vendor_id"] == "C"
    assert st["clarifications"][0]["author"] == "agent"
    status = {q["vendor_id"]: q for q in st["evaluation"]["quotes"]}
    assert status["C"]["status"] == "COMPLIANT" and status["C"]["rank"] == 2
