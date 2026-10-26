"""Tool-using planner: driven by a scripted stand-in model (no network needed)."""
import json
import shutil

import pytest

from procureiq import audit, config, llm
from procureiq.agent import graph


def fake_model(script):
    """Return a provider that answers agent prompts from `script(prompt)` and
    declines extraction prompts (so those fall back to the seed outputs)."""
    def call(prompt, image_path, system):
        if not prompt.startswith("TOOLS"):
            raise llm.LLMUnavailable("fake model only plans")
        return json.dumps(script(prompt)), "fake:planner-3b"
    return call


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    shutil.copytree(config.CACHE_DIR, cache)
    monkeypatch.setattr(llm, "CACHE_DIR", cache)
    monkeypatch.setattr(audit, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    yield


def use(monkeypatch, script):
    monkeypatch.setitem(llm.PROVIDERS, "fake", fake_model(script))


def good_agent(prompt):
    hist = prompt.split("STEPS SO FAR", 1)[1]
    if "(none yet)" in hist:
        return {"thought": "Licence looks misread; check the registry first.",
                "tool": "check_licence", "args": {"licence_no": "CM/L-4100067390"}}
    if "send_clarification" not in hist:
        return {"thought": "Not in registry, likely OCR error. Ask C to confirm and send FR proof.",
                "tool": "send_clarification",
                "args": {"vendor_id": "C", "message": "Dear Madam, thank you for your handwritten quote. "
                                                      "We could not read a few values clearly."}}
    return {"thought": "All conditional vendors handled.", "tool": "finish", "args": {"summary": "done"}}


def test_agent_investigates_and_messages(monkeypatch):
    use(monkeypatch, good_agent)
    st = graph.run()
    tools = [t["tool"] for t in st["agent_trace"]]
    assert tools[:3] == ["check_licence", "send_clarification", "finish"]
    assert "NOT in registry" in st["agent_trace"][0]["observation"]
    msg = st["clarifications"][0]["sent"]
    assert msg.startswith("Dear Madam") and "Items to confirm" in msg and "4100067390" in msg
    assert st["clarifications"][0]["author"] == "agent"
    assert {q["vendor_id"]: q["status"] for q in st["evaluation"]["quotes"]}["C"] == "COMPLIANT"
    assert st["sources"]["agent_r1"] == "fake:planner-3b"


def test_agent_cannot_contact_excluded_vendor(monkeypatch):
    def pushy(prompt):
        hist = prompt.split("STEPS SO FAR", 1)[1]
        if "(none yet)" in hist:
            return {"thought": "B is cheapest, ask them to fix it", "tool": "send_clarification",
                    "args": {"vendor_id": "B", "message": "Please send copper instead, same price ok?"}}
        return {"thought": "done", "tool": "finish", "args": {}}
    use(monkeypatch, pushy)
    st = graph.run()
    assert "refused" in st["agent_trace"][0]["observation"]
    # C was left unhandled by the model -> guardrail sent the rule template
    assert any(t["source"] == "rules" for t in st["agent_trace"])
    assert st["clarifications"][0]["vendor_id"] == "C" and st["clarifications"][0]["author"] == "rules"


def test_repeated_calls_are_refused_and_loop_is_bounded(monkeypatch):
    use(monkeypatch, lambda p: {"thought": "hmm", "tool": "lookup_datasheet", "args": {"brand": "AnnaWire"}})
    st = graph.run()
    obs = [t["observation"] for t in st["agent_trace"] if t["tool"] == "lookup_datasheet"]
    assert isinstance(obs[0], dict) and all("refused" in o for o in obs[1:])
    assert len([t for t in st["agent_trace"] if t["round"] == 1]) <= 9


def test_recorded_run_replays_without_model(monkeypatch):
    use(monkeypatch, good_agent)
    graph.run()                                   # live (fake) run records every step
    monkeypatch.setenv("LLM_PROVIDER", "cached")
    st = graph.run()                              # replay
    assert [t["tool"] for t in st["agent_trace"]][:3] == ["check_licence", "send_clarification", "finish"]
    assert st["agent_trace"][0]["source"] == "recorded fake:planner-3b"
