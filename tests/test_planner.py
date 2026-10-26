"""Tool-using planner: driven by a scripted stand-in model (no network needed)."""
import json

import pytest

from procureiq import llm
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
def fake_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")


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
    assert tools == ["check_licence", "send_clarification"]       # round auto-completes after the message
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
    assert [t["tool"] for t in st["agent_trace"]] == ["check_licence", "send_clarification"]
    assert st["agent_trace"][0]["source"] == "recorded fake:planner-3b"


def test_stuck_agent_is_stopped_after_two_refusals(monkeypatch):
    # what qwen2.5:3b actually did: keep re-sending after the vendor was already messaged
    def stuck(prompt):
        return {"thought": "ask B", "tool": "send_clarification",
                "args": {"vendor_id": "B", "message": "Please confirm your licence number sir"}}
    use(monkeypatch, stuck)
    st = graph.run()
    r1 = [t for t in st["agent_trace"] if t["round"] == 1 and t["source"] != "rules"]
    assert len(r1) == 2                                      # stopped after two refusals
    assert st["clarifications"][0]["author"] == "rules"      # guardrail handled C


def test_reply_can_only_update_what_was_asked():
    """Real qwen2.5:3b reply extraction invented quantity_coils=1 and quantity 90 m.
    Only the fields we asked about may change."""
    from procureiq.ingest import ingest_demo
    from procureiq.tools.normalise import apply_updates, scope_for
    from procureiq.tools.rules import evaluate

    reqs, quotes, _ = ingest_demo()
    c = next(q for q in quotes if q.vendor_id == "C")
    over_extracted = {"fields": {
        "quantity_coils": {"raw": "1", "value": "1", "confidence": 0.8},
        "quantity_m": {"raw": "90 m", "value": "90 m", "confidence": 0.8},
        "insulation": {"raw": "N/A", "value": "N/A", "confidence": 0.8},
        "oxygen_index": {"raw": "31", "value": "31", "confidence": 0.8},
        "temperature_index_c": {"raw": "260 deg C", "value": "260 deg C", "confidence": 0.8},
        "fr_test_report": {"raw": "NABL-TR-26-04417", "value": "NABL-TR-26-04417", "confidence": 0.8}}}
    allowed = scope_for(["Request FR type test report (oxygen index, temperature index)"])
    changed = apply_updates(c, over_extracted, allowed=allowed)
    assert set(changed) == {"oxygen_index", "temperature_index_c", "fr_test_report"}
    assert c.val("quantity_m") == 1800 and c.val("quantity_coils") == 20
