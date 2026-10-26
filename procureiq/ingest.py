"""Steps 1-3 of the agent loop: extract requirements, ingest quotes, normalise.

Run:  python -m procureiq.ingest
"""
from __future__ import annotations

from .config import DEMO_QUOTES, DEMO_REQUEST
from .schemas import Quote, RequirementSet
from .tools.extract import extract_quote, extract_requirements


def ingest_demo() -> tuple[RequirementSet, list[Quote], dict]:
    reqs, req_src = extract_requirements(DEMO_REQUEST)
    quotes, meta = [], {"requirements_source": req_src, "quotes": {}}
    for vid, info in DEMO_QUOTES.items():
        q, doc, src = extract_quote(vid, info["file"], info["channel"])
        q.vendor = info["vendor"] or q.vendor
        quotes.append(q)
        meta["quotes"][vid] = {"source": src, "parse_quality": doc.quality, "notes": doc.notes}
    return reqs, quotes, meta


def _fmt(v):
    return "—" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def main() -> None:
    reqs, quotes, meta = ingest_demo()
    print(f"\nREQUIREMENTS  ({len(reqs.requirements)} specs, via {meta['requirements_source']})")
    print(f"  {reqs.item}  |  budget ₹{reqs.budget_inr:,.0f}")
    for r in reqs.requirements:
        print(f"  [{r.priority[:4]}] {r.key:<20} {r.op:<14} {_fmt(r.value):<10} {r.unit or ''}")

    for q in quotes:
        m = meta["quotes"][q.vendor_id]
        print(f"\nQUOTE {q.vendor_id} — {q.vendor} ({q.channel}, via {m['source']}, "
              f"parse quality {m['parse_quality']})")
        for k, f in q.fields.items():
            mark = " ⚠ LOW" if f.confidence < 0.75 else ""
            flag = f"  {f.flags}" if f.flags else ""
            print(f"  {k:<22} {_fmt(f.value):<18} conf {f.confidence:.2f}{mark}{flag}")
        if q.low_confidence():
            print(f"  -> read-back needed for: {', '.join(q.low_confidence())}")


if __name__ == "__main__":
    main()
