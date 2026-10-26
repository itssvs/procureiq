"""The nine-step agent loop as a LangGraph state machine (Slide 4).

  extract_requirements (1)
        │
  ingest_quotes (2-3: parse + normalise)
        │
  verify (4-7: claim check, rules, cost, anomalies)  ◄─────────┐
        │                                                       │
   decide ──► clarify (8: read-back / proof request, re-ingest) ┘   max N rounds
        │
        ├──► escalate (unresolved after N rounds → human)
        │
  route_for_approval (9: evidence memo, human gate, audit log)

The graph never awards. It stops at AWAITING_APPROVAL; a human calls approve().
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .. import audit
from ..config import DEMO_QUOTES, DEMO_REQUEST, MAX_CLARIFY_ROUNDS
from ..schemas import Quote, RequirementSet
from ..tools.extract import extract_quote, extract_reply, extract_requirements
from ..tools.normalise import apply_updates
from ..tools.rules import Evaluation, evaluate
from .vendor_sim import VendorSimulator, draft_clarification


class AgentState(TypedDict, total=False):
    run_id: str
    inputs: dict
    requirements: dict
    quotes: list[dict]
    evaluation: dict
    round: int
    max_rounds: int
    silent: list[str]                 # vendors that did not answer
    clarifications: list[dict]        # history of messages and replies
    escalations: list[str]
    memo: dict
    status: str
    sources: dict                     # which LLM provider / cache served each step


def _reqs(s: AgentState) -> RequirementSet:
    return RequirementSet.model_validate(s["requirements"])


def _quotes(s: AgentState) -> list[Quote]:
    return [Quote.model_validate(q) for q in s["quotes"]]


def _eval(s: AgentState) -> Evaluation:
    return Evaluation.model_validate(s["evaluation"])


# --------------------------------------------------------------------- nodes
def node_extract_requirements(s: AgentState) -> dict:
    reqs, src = extract_requirements(Path(s["inputs"]["request"]))
    mand = sum(r.priority == "mandatory" for r in reqs.requirements)
    audit.log(s["run_id"], "1_extract_requirements",
              f"Extracted {len(reqs.requirements)} requirements ({mand} mandatory) via {src}",
              data=[f"[{r.priority}] {r.key} {r.op} {r.value}" for r in reqs.requirements])
    audit.save_state(s["run_id"], {}, request_id=reqs.request_id)
    return {"requirements": reqs.model_dump(), "sources": {**s.get("sources", {}), "requirements": src}}


def node_ingest_quotes(s: AgentState) -> dict:
    quotes, sources = [], dict(s.get("sources", {}))
    for vid, info in s["inputs"]["quotes"].items():
        q, doc, src = extract_quote(vid, Path(info["file"]), info["channel"])
        q.vendor = info.get("vendor") or q.vendor
        quotes.append(q.model_dump())
        sources[f"quote_{vid}"] = src
        low = q.low_confidence()
        audit.log(s["run_id"], "2_ingest", f"{vid} {q.vendor}: parsed {info['channel']} "
                  f"({len(q.fields)} fields, quality {doc.quality}) via {src}"
                  + (f"; low confidence: {', '.join(low)}" if low else ""),
                  data={"notes": doc.notes})
        flagged = [k for k, f in q.fields.items() if f.flags and "derived" not in f.flags]
        if flagged:
            audit.log(s["run_id"], "3_normalise", f"{vid}: flagged wording in {', '.join(flagged)}")
    return {"quotes": quotes, "sources": sources}


def node_verify(s: AgentState) -> dict:
    ev = evaluate(_reqs(s), _quotes(s))
    rnd = s.get("round", 0)
    touched = {c["vendor_id"] for c in s.get("clarifications", []) if c["round"] == rnd}
    for e in ev.quotes:
        if rnd and e.vendor_id not in touched:
            continue                      # re-verify only reports vendors that changed
        counts: dict[str, int] = {}
        for c in e.claims:
            counts[c.label] = counts.get(c.label, 0) + 1
        msg = f"{e.vendor_id} {e.status}" + (f" rank {e.rank}" if e.rank else "") + \
              " | " + ", ".join(f"{k.lower()} {v}" for k, v in sorted(counts.items()))
        if e.failed:
            msg += f" | failed: {e.failed[0]}"
        audit.log(s["run_id"], "4_verify" if rnd == 0 else f"4_reverify_r{rnd}", msg)
        for a in e.anomalies:
            if a.severity != "info":
                audit.log(s["run_id"], "7_anomaly", f"{e.vendor_id}: {a.code} — {a.message}")
    audit.save_state(s["run_id"], {**s, "evaluation": ev.model_dump()})
    return {"evaluation": ev.model_dump()}


def decide(s: AgentState) -> str:
    ev = _eval(s)
    pending = [e.vendor_id for e in ev.quotes
               if e.status == "CONDITIONAL" and e.vendor_id not in s.get("silent", [])]
    unresolved = [e.vendor_id for e in ev.quotes if e.status == "CONDITIONAL"]
    if pending and s.get("round", 0) < s.get("max_rounds", MAX_CLARIFY_ROUNDS):
        return "clarify"
    if unresolved:
        return "escalate"
    return "approve"


def make_clarify_node(sim: VendorSimulator):
    def node_clarify(s: AgentState) -> dict:
        rnd = s.get("round", 0) + 1
        ev = _eval(s)
        quotes = _quotes(s)
        by_id = {q.vendor_id: q for q in quotes}
        silent = list(s.get("silent", []))
        history = list(s.get("clarifications", []))
        sources = dict(s.get("sources", {}))

        for e in ev.quotes:
            if e.status != "CONDITIONAL" or e.vendor_id in silent:
                continue
            q = by_id[e.vendor_id]
            msg = draft_clarification(q, e, rnd)
            audit.log(s["run_id"], "8_clarify", f"Round {rnd}: sent {len(e.open_items)} question(s) to "
                      f"{q.vendor}", data={"message": msg})
            entry = {"round": rnd, "vendor_id": q.vendor_id, "vendor": q.vendor, "sent": msg,
                     "reply": None, "changed": []}
            reply_path = sim.send(q.vendor_id, msg)
            if reply_path is None:
                silent.append(q.vendor_id)
                audit.log(s["run_id"], "8_clarify", f"{q.vendor} did not reply", actor=f"vendor:{q.vendor_id}")
            else:
                data, src = extract_reply(q.vendor_id, reply_path)
                changed = apply_updates(q, data, source=f"clarification_r{rnd}")
                sources[f"reply_{q.vendor_id}_r{rnd}"] = src
                entry["reply"] = Path(reply_path).read_text(encoding="utf-8")
                entry["changed"] = changed
                audit.log(s["run_id"], "8_clarify", f"Reply received; updated {', '.join(changed) or 'nothing'}",
                          actor=f"vendor:{q.vendor_id}",
                          data={k: q.fields[k].value for k in changed})
            history.append(entry)
        return {"round": rnd, "quotes": [q.model_dump() for q in quotes], "silent": silent,
                "clarifications": history, "sources": sources}
    return node_clarify


def node_escalate(s: AgentState) -> dict:
    ev = _eval(s)
    names = [f"{e.vendor_id} {e.vendor}" for e in ev.quotes if e.status == "CONDITIONAL"]
    audit.log(s["run_id"], "8_escalate", f"Unresolved after {s.get('round', 0)} round(s): "
              f"{', '.join(names)} → manual review")
    return {"escalations": names}


def build_memo(s: AgentState) -> dict:
    ev = _eval(s)
    ranked = sorted([e for e in ev.quotes if e.rank], key=lambda e: e.rank)

    def claim_view(e):
        return [{"key": c.key, "label": c.label, "reason": c.reason,
                 "evidence": [x.model_dump() for x in c.evidence]} for c in e.claims]

    return {
        "request_id": ev.request_id,
        "summary": ev.summary,
        "recommended": None if not ranked else {
            "vendor_id": ranked[0].vendor_id, "vendor": ranked[0].vendor,
            "price_pre_tax": ranked[0].price_pre_tax, "price_with_tax": ranked[0].price_with_tax},
        "ranking": [{"rank": e.rank, "vendor_id": e.vendor_id, "vendor": e.vendor, "score": e.score,
                     "price_pre_tax": e.price_pre_tax, "delivery_days": e.delivery_days,
                     "warranty_years": e.warranty_years, "pareto": e.pareto} for e in ranked],
        "excluded": [{"vendor_id": e.vendor_id, "vendor": e.vendor, "price_pre_tax": e.price_pre_tax,
                      "failed": e.failed,
                      "evidence": [c for c in claim_view(e) if c["label"] in ("CONTRADICTED", "SUBSTITUTED")]}
                     for e in ev.quotes if e.status == "EXCLUDED"],
        "conditional": [{"vendor_id": e.vendor_id, "vendor": e.vendor, "open_items": e.open_items}
                        for e in ev.quotes if e.status == "CONDITIONAL"],
        "clarification_rounds": s.get("round", 0),
        "escalations": s.get("escalations", []),
        "requires_human_approval": True,
        "decision": None,
    }


def node_route_for_approval(s: AgentState) -> dict:
    memo = build_memo(s)
    rec = memo["recommended"]
    audit.log(s["run_id"], "9_route_for_approval",
              f"Memo ready. Recommended {rec['vendor']} (₹{rec['price_pre_tax']:,.0f}). Awaiting human sign-off."
              if rec else "No compliant quote. Awaiting human decision.", data={"summary": memo["summary"]})
    final = {**s, "memo": memo, "status": "AWAITING_APPROVAL"}
    audit.save_state(s["run_id"], final, status="AWAITING_APPROVAL")
    return {"memo": memo, "status": "AWAITING_APPROVAL"}


# --------------------------------------------------------------------- graph
def build_graph(sim: Optional[VendorSimulator] = None):
    g = StateGraph(AgentState)
    g.add_node("extract_requirements", node_extract_requirements)
    g.add_node("ingest_quotes", node_ingest_quotes)
    g.add_node("verify", node_verify)
    g.add_node("clarify", make_clarify_node(sim or VendorSimulator()))
    g.add_node("escalate", node_escalate)
    g.add_node("route_for_approval", node_route_for_approval)

    g.set_entry_point("extract_requirements")
    g.add_edge("extract_requirements", "ingest_quotes")
    g.add_edge("ingest_quotes", "verify")
    g.add_conditional_edges("verify", decide,
                            {"clarify": "clarify", "escalate": "escalate", "approve": "route_for_approval"})
    g.add_edge("clarify", "verify")
    g.add_edge("escalate", "route_for_approval")
    g.add_edge("route_for_approval", END)
    return g.compile()


def demo_inputs() -> dict:
    return {"request": str(DEMO_REQUEST),
            "quotes": {vid: {**info, "file": str(info["file"])} for vid, info in DEMO_QUOTES.items()}}


def run(inputs: Optional[dict] = None, sim: Optional[VendorSimulator] = None,
        max_rounds: int = MAX_CLARIFY_ROUNDS) -> AgentState:
    run_id = audit.new_run()
    audit.log(run_id, "0_start", "Run started", data={"max_rounds": max_rounds})
    graph = build_graph(sim)
    return graph.invoke({"run_id": run_id, "inputs": inputs or demo_inputs(), "round": 0,
                         "max_rounds": max_rounds, "silent": [], "clarifications": [], "sources": {}},
                        {"recursion_limit": 50})


# ------------------------------------------------------------- human gate
class ApprovalError(ValueError):
    pass


def approve(run_id: str, vendor_id: str, approver: str, note: str = "") -> dict:
    """Human sign-off. Only a COMPLIANT quote can be awarded."""
    run_row = audit.get_run(run_id)
    if run_row is None:
        raise ApprovalError(f"unknown run {run_id}")
    if run_row["status"] != "AWAITING_APPROVAL":
        raise ApprovalError(f"run is {run_row['status']}, not awaiting approval")
    state = run_row["state"]
    ev = Evaluation.model_validate(state["evaluation"])
    try:
        q = ev.by_id(vendor_id)
    except StopIteration:
        raise ApprovalError(f"unknown vendor {vendor_id}")
    if q.status != "COMPLIANT":
        audit.log(run_id, "approval", f"Blocked award to {q.vendor}: quote is {q.status}",
                  actor=f"human:{approver}")
        raise ApprovalError(f"{q.vendor} is {q.status}; only a COMPLIANT quote can be awarded")
    decision = {"vendor_id": vendor_id, "vendor": q.vendor, "approver": approver, "note": note,
                "followed_recommendation": q.rank == 1}
    state["memo"]["decision"] = decision
    state["status"] = "APPROVED"
    audit.log(run_id, "approval", f"Approved award to {q.vendor} (rank {q.rank})"
              + (f": {note}" if note else ""), actor=f"human:{approver}", data=decision)
    audit.save_state(run_id, state, status="APPROVED")
    return decision


def reject_all(run_id: str, approver: str, note: str) -> None:
    run_row = audit.get_run(run_id)
    if run_row is None:
        raise ApprovalError(f"unknown run {run_id}")
    state = run_row["state"]
    state["status"] = "REJECTED"
    audit.log(run_id, "approval", f"All quotes rejected: {note}", actor=f"human:{approver}")
    audit.save_state(run_id, state, status="REJECTED")


def summary(state: AgentState) -> dict[str, Any]:
    return {"run_id": state["run_id"], "status": state["status"], "rounds": state.get("round", 0),
            "summary": state["memo"]["summary"], "recommended": state["memo"]["recommended"]}
