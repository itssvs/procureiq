"""Steps 5-6: rule engine (pass / fail / unverified) and cost + trade-off ranking.

Ranking rule (Slide 5): only quotes with no contradicted or substituted
mandatory claim are considered. A mandatory claim that is merely unverified
makes the quote CONDITIONAL — listed, proof requested, not ranked yet.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..schemas import Quote, Requirement, RequirementSet
from .anomaly import Anomaly, detect
from .verifier import SPEC_CLAIMS, ClaimResult, verify_claim

Outcome = Literal["pass", "fail", "unverified"]
Status = Literal["COMPLIANT", "CONDITIONAL", "EXCLUDED"]

WEIGHTS = {"price": 0.6, "delivery": 0.2, "warranty": 0.2}
PREFERRED_PENALTY = 0.05      # per unmet preferred requirement


class RuleResult(BaseModel):
    key: str
    priority: str
    outcome: Outcome
    detail: str
    label: Optional[str] = None       # verifier label for spec claims


class QuoteEvaluation(BaseModel):
    vendor_id: str
    vendor: str
    channel: str
    claims: list[ClaimResult]
    rules: list[RuleResult]
    anomalies: list[Anomaly]
    status: Status
    failed: list[str] = Field(default_factory=list)       # mandatory failures, human readable
    open_items: list[str] = Field(default_factory=list)   # what would resolve CONDITIONAL
    price_pre_tax: Optional[float] = None
    price_with_tax: Optional[float] = None
    delivery_days: Optional[float] = None
    warranty_years: Optional[float] = None
    score: Optional[float] = None
    rank: Optional[int] = None
    pareto: bool = False


class Evaluation(BaseModel):
    request_id: str
    quotes: list[QuoteEvaluation]
    summary: str = ""

    def by_id(self, vid: str) -> QuoteEvaluation:
        return next(q for q in self.quotes if q.vendor_id == vid)


LABEL_TO_OUTCOME = {"VERIFIED": "pass", "UNVERIFIABLE": "unverified",
                    "CONTRADICTED": "fail", "SUBSTITUTED": "fail"}


def _commercial(req: Requirement, q: Quote) -> RuleResult:
    f = q.fields.get(req.key)
    if f is None or f.value is None:
        return RuleResult(key=req.key, priority=req.priority, outcome="unverified", detail="not stated")
    v = f.value
    ok = {"gte": v >= req.value, "lte": v <= req.value, "eq": v == req.value}[req.op]
    sym = {"gte": "≥", "lte": "≤", "eq": "="}[req.op]
    return RuleResult(key=req.key, priority=req.priority, outcome="pass" if ok else "fail",
                      detail=f"{v:g} {sym} {req.value:g} required" if ok else f"{v:g}, needs {sym} {req.value:g}")


def evaluate_quote(reqs: RequirementSet, q: Quote, peers: list[Quote]) -> QuoteEvaluation:
    claims, rules = [], []
    for req in reqs.requirements:
        if req.key in SPEC_CLAIMS:
            c = verify_claim(req, q)
            claims.append(c)
            rules.append(RuleResult(key=req.key, priority=req.priority, outcome=LABEL_TO_OUTCOME[c.label],
                                    detail=c.reason, label=c.label))
        else:
            rules.append(_commercial(req, q))

    failed = [f"{r.key}: {r.detail}" for r in rules if r.priority == "mandatory" and r.outcome == "fail"]
    unverified = [r for r in rules if r.priority == "mandatory" and r.outcome == "unverified"]
    status: Status = "EXCLUDED" if failed else ("CONDITIONAL" if unverified else "COMPLIANT")

    open_items = [c.action for c in claims if c.label == "UNVERIFIABLE" and c.action]
    # low-confidence commercial fields also need a read-back
    for k in q.low_confidence():
        if k not in SPEC_CLAIMS:
            f = q.fields[k]
            open_items.append(f"Read-back: please confirm {k} = {f.value:g}" if isinstance(f.value, (int, float))
                              else f"Read-back: please confirm {k}")

    pre = q.val("total_price_inr")
    gst = q.val("gst_pct", 18)
    return QuoteEvaluation(
        vendor_id=q.vendor_id, vendor=q.vendor, channel=q.channel, claims=claims, rules=rules,
        anomalies=detect(q, reqs, peers), status=status, failed=failed, open_items=open_items,
        price_pre_tax=pre, price_with_tax=None if pre is None else round(pre * (1 + gst / 100)),
        delivery_days=q.val("delivery_days"), warranty_years=q.val("warranty_years"))


def _pareto(evals: list[QuoteEvaluation]) -> None:
    """Non-dominated on (lower price, faster delivery, longer warranty)."""
    pts = [e for e in evals if None not in (e.price_pre_tax, e.delivery_days, e.warranty_years)]
    for e in pts:
        e.pareto = not any(
            o is not e and o.price_pre_tax <= e.price_pre_tax and o.delivery_days <= e.delivery_days
            and o.warranty_years >= e.warranty_years and
            (o.price_pre_tax, o.delivery_days, -o.warranty_years) != (e.price_pre_tax, e.delivery_days, -e.warranty_years)
            for o in pts)


def rank(reqs: RequirementSet, evals: list[QuoteEvaluation]) -> None:
    """Score and rank COMPLIANT quotes only. Score 0..1, higher is better."""
    pool = [e for e in evals if e.status == "COMPLIANT" and e.price_pre_tax]
    _pareto([e for e in evals if e.status != "EXCLUDED"])
    if not pool:
        return
    pmin = min(e.price_pre_tax for e in pool)
    dmax = max(e.delivery_days or 0 for e in pool) or 1
    wmax = max(e.warranty_years or 0 for e in pool) or 1
    for e in pool:
        s = (WEIGHTS["price"] * pmin / e.price_pre_tax
             + WEIGHTS["delivery"] * (1 - (e.delivery_days or dmax) / (dmax * 2))
             + WEIGHTS["warranty"] * (e.warranty_years or 0) / wmax)
        missed = sum(1 for r in e.rules if r.priority == "preferred" and r.outcome != "pass")
        e.score = round(s - PREFERRED_PENALTY * missed, 3)
    for i, e in enumerate(sorted(pool, key=lambda x: -x.score), start=1):
        e.rank = i


def evaluate(reqs: RequirementSet, quotes: list[Quote]) -> Evaluation:
    evals = [evaluate_quote(reqs, q, quotes) for q in quotes]
    rank(reqs, evals)
    ranked = sorted([e for e in evals if e.rank], key=lambda e: e.rank)
    cheapest = min((e for e in evals if e.price_pre_tax), key=lambda e: e.price_pre_tax, default=None)
    parts = []
    if ranked:
        parts.append(f"Recommended: {ranked[0].vendor} (₹{ranked[0].price_pre_tax:,.0f}).")
    if cheapest and cheapest.status == "EXCLUDED" and ranked:
        saving = 1 - cheapest.price_pre_tax / ranked[0].price_pre_tax
        parts.append(f"The cheapest quote, {cheapest.vendor}, is {saving:.0%} cheaper but excluded: "
                     f"{cheapest.failed[0]}")
    cond = [e.vendor for e in evals if e.status == "CONDITIONAL"]
    if cond:
        parts.append(f"Awaiting proof from: {', '.join(cond)}.")
    return Evaluation(request_id=reqs.request_id, quotes=evals, summary=" ".join(parts))
