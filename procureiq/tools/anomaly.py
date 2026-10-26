"""Step 7: anomaly detection — arithmetic, tax, price outliers, fine print."""
from __future__ import annotations

import re
from statistics import median
from typing import Literal

from pydantic import BaseModel

from .. import kb
from ..schemas import Quote, RequirementSet

Severity = Literal["info", "warning", "critical"]
GST_SLABS = {0, 5, 12, 18, 28}
FINE_PRINT = {
    r"will send later|to follow|later": "Document promised later; not on file at evaluation time",
    r"as per availability|subject to (stock|availability)": "Supply depends on availability",
    r"ex[- ]?works|freight extra|\+ ?freight": "Freight not included",
    r"price(s)? subject to change|rate may vary": "Price not firm",
    r"or equivalent|or similar|equivalent make": "Substitution allowed by wording",
}


class Anomaly(BaseModel):
    code: str
    severity: Severity
    message: str
    field: str = ""


def detect(q: Quote, reqs: RequirementSet, peers: list[Quote]) -> list[Anomaly]:
    out: list[Anomaly] = []
    qty, rate, total = q.val("quantity_coils"), q.val("unit_price_inr"), q.val("total_price_inr")

    # arithmetic: qty x rate == total
    if qty and rate and total:
        expected = qty * rate
        if abs(expected - total) > 0.01 * expected:
            out.append(Anomaly(code="ARITHMETIC", severity="critical", field="total_price_inr",
                               message=f"{qty:g} × ₹{rate:,.0f} = ₹{expected:,.0f}, quote says ₹{total:,.0f}"))

    # GST must be a real slab
    gst = q.val("gst_pct")
    if gst is not None and gst not in GST_SLABS:
        out.append(Anomaly(code="GST_SLAB", severity="warning", field="gst_pct",
                           message=f"GST {gst:g}% is not a valid GST slab (0/5/12/18/28) — likely misread"))

    # quantity vs requested
    req = reqs.get("quantity_m")
    if req and q.val("quantity_m") is not None and q.val("quantity_m") < req.value:
        out.append(Anomaly(code="SHORT_QTY", severity="critical", field="quantity_m",
                           message=f"Quotes {q.val('quantity_m'):g} m, request is {req.value:g} m"))

    # price vs market reference band
    if rate:
        for ref in kb.market_reference():
            lo, hi = ref["typical_price_inr"]
            if rate < lo * 0.95:
                pct = (lo - rate) / lo * 100
                out.append(Anomaly(code="PRICE_LOW", severity="warning", field="unit_price_inr",
                                   message=f"₹{rate:,.0f}/coil is {pct:.0f}% below the market band "
                                           f"₹{lo:,}–{hi:,} for {ref['item']} ({ref['id']}). "
                                           "Unusually cheap often means a different spec."))
            elif rate > hi * 1.10:
                out.append(Anomaly(code="PRICE_HIGH", severity="info", field="unit_price_inr",
                                   message=f"₹{rate:,.0f}/coil is above the market band ₹{lo:,}–{hi:,}"))

    # deviation from peer median
    peer_rates = [p.val("unit_price_inr") for p in peers if p.val("unit_price_inr")]
    if rate and len(peer_rates) >= 3:
        med = median(peer_rates)
        dev = (rate - med) / med
        if abs(dev) > 0.15:
            out.append(Anomaly(code="PEER_OUTLIER", severity="info", field="unit_price_inr",
                               message=f"{dev:+.0%} vs median quote ₹{med:,.0f}/coil"))

    # fine print in the vendor's own words
    seen = set()
    for key, f in q.fields.items():
        for pat, msg in FINE_PRINT.items():
            if f.raw and re.search(pat, f.raw, re.I) and msg not in seen:
                seen.add(msg)
                out.append(Anomaly(code="FINE_PRINT", severity="warning", field=key,
                                   message=f'{msg}: "{f.raw}"'))
    return out
