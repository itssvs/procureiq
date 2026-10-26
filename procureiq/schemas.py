"""Canonical data shapes shared across the pipeline.

Everything the agent passes between steps is one of these models, so every
LLM output is schema-validated before the next step (Slide 9: validated
outputs).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Priority = Literal["mandatory", "preferred"]
Op = Literal["eq", "gte", "lte", "valid_licence", "proven"]


# --------------------------------------------------------------- requirement
class Requirement(BaseModel):
    """One buyer spec, e.g. conductor_material == copper (mandatory)."""

    key: str                       # canonical spec key, see SPEC_KEYS
    op: Op = "eq"
    value: Any = None
    unit: Optional[str] = None
    priority: Priority = "mandatory"
    source_text: str = ""          # the words in the email this came from
    via: Optional[str] = None      # "rules" when added by the safety net, not the model


class RequirementSet(BaseModel):
    request_id: str
    requester: str = ""
    item: str = ""
    budget_inr: Optional[float] = None
    requirements: list[Requirement]

    def get(self, key: str) -> Optional[Requirement]:
        return next((r for r in self.requirements if r.key == key), None)


# ---------------------------------------------------------------------- quote
class Field_(BaseModel):
    """A single value read from a quote, with how sure we are about it."""

    value: Any = None
    unit: Optional[str] = None
    confidence: float = 1.0        # 0..1, from LLM self-report x OCR quality
    raw: str = ""                  # vendor's exact words
    flags: list[str] = Field(default_factory=list)   # e.g. ["equivalence_clause"]


class Quote(BaseModel):
    vendor_id: str
    vendor: str
    channel: Literal["pdf", "excel", "email", "whatsapp", "photo", "voice"]
    quote_ref: str = ""
    brand: str = ""
    fields: dict[str, Field_] = Field(default_factory=dict)
    source_file: str = ""

    def val(self, key: str, default: Any = None) -> Any:
        f = self.fields.get(key)
        return default if f is None or f.value is None else f.value

    def low_confidence(self, threshold: float = 0.75) -> list[str]:
        return [k for k, f in self.fields.items() if f.value is not None and f.confidence < threshold]


# Canonical keys. Requirement extraction and quote extraction both map onto
# these, which is what makes claims comparable.
SPEC_KEYS = {
    # product specs
    "conductor_material": "copper | aluminium",
    "size_sqmm": "conductor cross-section in sq mm",
    "cores": "number of cores",
    "voltage_grade_kv": "voltage grade in kV",
    "insulation": "insulation type, e.g. PVC Type A",
    "standard": "product standard, e.g. IS 694:2010",
    "isi_licence": "BIS / ISI licence number, e.g. CM/L-4100012345",
    "fire_grade": "FR | FR-LSH | none",
    "conductor_class": "2 (stranded) | 5 (flexible)",
    # commercial
    "quantity_m": "total length in metres",
    "delivery_days": "days from PO to delivery",
    "warranty_years": "warranty in years",
}

# Extra quote-only fields that act as evidence for spec claims
EVIDENCE_KEYS = {
    "oxygen_index": "FR test: oxygen index",
    "temperature_index_c": "FR test: temperature index in deg C",
    "fr_test_report": "test report number for FR claim",
    "conductor_detail": "any other words describing the conductor (grade, alloy)",
    "resistance_ohm_per_km": "conductor DC resistance",
    "coil_length_m": "length of one coil in metres",
    "quantity_coils": "number of coils",
    "unit_price_inr": "price per coil, before tax",
    "total_price_inr": "total before tax",
    "gst_pct": "GST percent",
}
