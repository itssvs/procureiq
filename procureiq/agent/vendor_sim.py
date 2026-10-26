"""Vendor simulator + clarification drafting (step 8).

In production the outbound message would go by email / WhatsApp Business API
and the reply would arrive on the same channel. For the demo, the simulator
returns a pre-written reply file for vendors that answer, and nothing for
vendors that stay silent (which triggers escalation after the round cap).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import DEMO_CLARIFICATION_REPLIES
from ..schemas import Quote
from ..tools.rules import QuoteEvaluation


LABELS = {"isi_licence": "ISI licence number", "total_price_inr": "Total price (₹, before GST)",
          "gst_pct": "GST rate (%)", "unit_price_inr": "Rate per coil (₹)", "fire_grade": "Fire grade",
          "conductor_material": "Conductor material", "voltage_grade_kv": "Voltage grade (kV)",
          "delivery_days": "Delivery (days from PO)", "warranty_years": "Warranty (years)"}


def draft_clarification(q: Quote, ev: QuoteEvaluation, round_no: int) -> str:
    """Deterministic, polite request listing exactly what is missing.

    Read-back lines quote the value as we read it, so the vendor corrects it
    rather than us guessing (Slide 6)."""
    readbacks, proofs = [], []
    for item in ev.open_items:
        if item.startswith("Read-back:"):
            key_val = item.split("confirm", 1)[1].strip()
            key, _, val = key_val.partition("=")
            key = key.strip()
            label = LABELS.get(key, key.replace("_", " "))
            readbacks.append(f"  • {label}: we read \"{val.strip()}\" — is this correct?")
        else:
            proofs.append(f"  • {item.removeprefix('Request ').removeprefix('request ')}")
    lines = [f"Subject: Clarification {round_no} on your quote {q.quote_ref or ''}".rstrip(),
             "", f"Dear {q.vendor},", "",
             "Thank you for your quotation. Before we can evaluate it, please help us with the following:"]
    if readbacks:
        lines += ["", "Please confirm (read-back):", *readbacks]
    if proofs:
        lines += ["", "Please share supporting documents:", *proofs]
    lines += ["", "Quotes are compared only on verified specifications, so a quick reply helps your offer.",
              "", "Regards,", "Procurement, Sunrise Manufacturing"]
    return "\n".join(lines)


class VendorSimulator:
    def __init__(self, replies: Optional[dict[str, Path]] = None):
        self.replies = dict(DEMO_CLARIFICATION_REPLIES if replies is None else replies)
        self.sent: dict[str, list[str]] = {}

    def send(self, vendor_id: str, message: str) -> Optional[Path]:
        """Deliver a message; return the reply file if the vendor answers, else None.
        Each vendor answers at most once (a second request gets silence)."""
        self.sent.setdefault(vendor_id, []).append(message)
        if len(self.sent[vendor_id]) == 1:
            return self.replies.get(vendor_id)
        return None
