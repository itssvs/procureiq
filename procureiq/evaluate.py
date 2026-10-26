"""Steps 4-7 on the demo data: verify claims, apply rules, detect anomalies, rank.

Run:  python -m procureiq.evaluate
"""
from __future__ import annotations

from .ingest import ingest_demo
from .tools.rules import Evaluation, evaluate

ICON = {"VERIFIED": "✅", "UNVERIFIABLE": "❔", "CONTRADICTED": "❌", "SUBSTITUTED": "⚠️ ",
        "pass": "✅", "fail": "❌", "unverified": "❔"}


def print_evaluation(ev: Evaluation) -> None:
    for e in ev.quotes:
        head = f"#{e.rank}" if e.rank else e.status
        print(f"\n{'=' * 78}\n{e.vendor_id} — {e.vendor} ({e.channel})   ₹{e.price_pre_tax:,.0f} + GST   "
              f"→ {e.status} {head if e.rank else ''}  {'[pareto]' if e.pareto else ''}")
        print("-" * 78)
        for c in e.claims:
            print(f"  {ICON[c.label]} {c.label:<13} {c.key:<20} {c.reason}")
            for evd in c.evidence[1:3]:
                print(f"       └ {evd.id}: {evd.text}")
        for r in e.rules:
            if r.label is None:
                print(f"  {ICON[r.outcome]} {r.outcome.upper():<13} {r.key:<20} {r.detail} [{r.priority}]")
        for a in e.anomalies:
            print(f"  ⚑ {a.severity.upper():<13} {a.code:<20} {a.message}")
        if e.failed:
            print(f"  FAILED: {'; '.join(e.failed)}")
        if e.open_items:
            print(f"  OPEN:   {'; '.join(e.open_items)}")
        if e.score is not None:
            print(f"  score {e.score}")
    print(f"\n{ev.summary}\n")


def main() -> None:
    reqs, quotes, _ = ingest_demo()
    print_evaluation(evaluate(reqs, quotes))


if __name__ == "__main__":
    main()
