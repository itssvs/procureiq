"""Part 4: agent loop, clarification, escalation, human gate, API."""
import os

import pytest

os.environ["LLM_PROVIDER"] = "cached"

from fastapi.testclient import TestClient  # noqa: E402

from procureiq import audit  # noqa: E402
from procureiq.agent import graph  # noqa: E402
from procureiq.agent.vendor_sim import VendorSimulator  # noqa: E402
from procureiq.api import app  # noqa: E402


def statuses(state):
    return {q["vendor_id"]: q["status"] for q in state["evaluation"]["quotes"]}


def test_full_loop_clarifies_c_then_awaits_approval():
    state = graph.run()
    assert state["status"] == "AWAITING_APPROVAL"
    assert state["round"] == 1
    assert statuses(state) == {"A": "COMPLIANT", "B": "EXCLUDED", "C": "COMPLIANT"}
    memo = state["memo"]
    assert memo["recommended"]["vendor_id"] == "A"
    assert [r["vendor_id"] for r in memo["ranking"]] == ["A", "C"]
    assert memo["excluded"][0]["evidence"][0]["label"] == "SUBSTITUTED"
    assert memo["decision"] is None                               # never auto-awards

    steps = [e["step"] for e in audit.events(state["run_id"])]
    for s in ("1_extract_requirements", "2_ingest", "4_verify", "8_clarify",
              "4_reverify_r1", "9_route_for_approval"):
        assert s in steps


def test_silent_vendor_escalates_after_round_cap():
    state = graph.run(sim=VendorSimulator(replies={}), max_rounds=3)
    assert statuses(state)["C"] == "CONDITIONAL"
    assert state["round"] == 1                    # silent vendors are not re-asked
    assert state["escalations"] and "C" in state["escalations"][0]
    assert state["memo"]["conditional"][0]["vendor_id"] == "C"


def test_round_cap_zero_escalates_immediately():
    state = graph.run(max_rounds=0)
    assert state["round"] == 0 and state["escalations"]


def test_human_gate():
    state = graph.run()
    rid = state["run_id"]
    with pytest.raises(graph.ApprovalError):
        graph.approve(rid, "B", "manager")                        # excluded quote can't be awarded
    d = graph.approve(rid, "A", "manager", "as recommended")
    assert d["followed_recommendation"]
    assert audit.get_run(rid)["status"] == "APPROVED"
    with pytest.raises(graph.ApprovalError):
        graph.approve(rid, "C", "manager")                        # already decided


def test_api_roundtrip():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/runs/demo").json()
    rid = r["run_id"]
    assert r["recommended"]["vendor_id"] == "A"
    assert c.get(f"/runs/{rid}/memo").json()["clarification_rounds"] == 1
    assert len(c.get(f"/runs/{rid}/audit").json()) > 10
    assert c.post(f"/runs/{rid}/approve", json={"vendor_id": "B", "approver": "m"}).status_code == 409
    assert c.post(f"/runs/{rid}/approve", json={"vendor_id": "C", "approver": "m",
                                                "note": "faster local supplier"}).status_code == 200
    assert c.get(f"/runs/{rid}").json()["status"] == "APPROVED"
    assert c.get("/runs/nope").status_code == 404
