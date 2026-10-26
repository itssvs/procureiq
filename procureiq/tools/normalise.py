"""Normaliser: map vendor wording and units onto one canonical schema.

- Units:       pint ("1100V" -> 1.1 kV, "2 weeks" -> 14 days, "2 yrs" -> 2 years)
- Numbers:     Indian formats ("4,55,000", "Rs 3.44 L", "₹4.2 lakh")
- Categories:  rapidfuzz against kb/synonyms.yaml ("AL 1350" -> aluminium)
- Flags:       "or equivalent" style wording is flagged for the verifier
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional

import pint
import yaml
from rapidfuzz import fuzz, process

from ..config import KB_DIR
from ..schemas import Field_, Quote, Requirement, RequirementSet

ureg = pint.UnitRegistry()
ureg.define("sqmm = millimeter ** 2")
ureg.define("lakh = 100000")

UNIT_ALIASES = {
    "sq mm": "sqmm", "sq.mm": "sqmm", "sq. mm": "sqmm", "sqmm": "sqmm", "mm2": "sqmm", "mm²": "sqmm",
    "kv": "kilovolt", "v": "volt", "volt": "volt", "volts": "volt",
    "m": "meter", "mtr": "meter", "mtrs": "meter", "metre": "meter", "metres": "meter", "meter": "meter",
    "meters": "meter", "km": "kilometer",
    "day": "day", "days": "day", "week": "week", "weeks": "week", "wk": "week",
    "yr": "year", "yrs": "year", "year": "year", "years": "year", "month": "month", "months": "month",
    "°c": "degC", "deg c": "degC", "c": "degC",
}

# canonical key -> pint unit the value is stored in
CANONICAL_UNITS = {
    "size_sqmm": "sqmm",
    "voltage_grade_kv": "kilovolt",
    "quantity_m": "meter",
    "coil_length_m": "meter",
    "delivery_days": "day",
    "warranty_years": "year",
}
CATEGORICAL = {"conductor_material", "insulation", "fire_grade", "cores", "conductor_class"}

# Plausible ranges in canonical units. A reading outside the range is not trusted
# (e.g. "H00V" read as 0 V, or 1100 taken as kV when the vendor meant volts).
PLAUSIBLE = {"size_sqmm": (0.5, 400), "voltage_grade_kv": (0.1, 33), "quantity_m": (1, 100000),
             "coil_length_m": (5, 2000), "delivery_days": (0, 365), "warranty_years": (0, 25)}


def _plausible(key: str, v: Optional[float]) -> bool:
    lo, hi = PLAUSIBLE.get(key, (float("-inf"), float("inf")))
    return v is not None and lo <= v <= hi


@lru_cache(maxsize=1)
def synonyms() -> dict:
    return yaml.safe_load((KB_DIR / "synonyms.yaml").read_text(encoding="utf-8"))


# -------------------------------------------------------------------- numbers
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(raw: Any) -> Optional[float]:
    """'4,55,000' -> 455000; 'Rs 3.44 L' -> 344000; '22,750 / coil' -> 22750."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).lower().replace("₹", " ").replace("rs.", " ").replace("rs", " ")
    m = _NUM.search(s)
    if not m:
        return None
    val = float(m.group().replace(",", ""))
    tail = s[m.end():].strip()
    if re.match(r"^(l|lakh|lakhs|lac)\b", tail):
        val *= 100000
    elif re.match(r"^(cr|crore)\b", tail):
        val *= 10000000
    return val


def parse_quantity(raw: Any, key: str) -> tuple[Optional[float], float]:
    """Return (value in canonical unit, confidence). Unknown unit -> assume canonical, conf 0.7."""
    target = CANONICAL_UNITS[key]
    if raw is None:
        return None, 0.0
    if isinstance(raw, (int, float)):
        v = float(raw)
        if key == "voltage_grade_kv" and v >= 50:      # a bare 1100 means volts
            v = v / 1000
        return v, 1.0
    s = str(raw).lower().strip()
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*([a-z°.² ]*)", s)
    if not m:
        return None, 0.0
    num = float(m.group(1).replace(",", ""))
    unit_txt = m.group(2).strip().rstrip(".")
    unit = None
    for alias in sorted(UNIT_ALIASES, key=len, reverse=True):   # longest alias first
        if unit_txt.startswith(alias):
            unit = UNIT_ALIASES[alias]
            break
    if unit is None:
        if unit_txt == "":                               # bare number: assume canonical unit
            if key == "voltage_grade_kv" and num >= 50:
                num = num / 1000
            return num, 1.0
        return num, 0.7
    try:
        q = ureg.Quantity(num, unit).to(target)
        return round(float(q.magnitude), 4), 1.0
    except pint.DimensionalityError:
        return num, 0.5


# ----------------------------------------------------------------- categories
def canon_category(key: str, raw: Any) -> tuple[Optional[Any], float]:
    """Fuzzy-map a raw phrase to its canonical value. Returns (value, score 0..1)."""
    if raw is None:
        return None, 0.0
    table = synonyms().get(key)
    if not table:
        return raw, 1.0
    text = str(raw).lower()
    best_val, best_score = None, 0.0
    for canon, variants in table.items():
        # exact token hit (e.g. "al" inside "al 1350 grade") beats fuzzy similarity
        for v in variants:
            if re.search(rf"(?<![a-z]){re.escape(v.lower())}(?![a-z])", text):
                score = 1.0
                break
        else:
            match = process.extractOne(text, [v.lower() for v in variants], scorer=fuzz.token_set_ratio)
            score = (match[1] / 100) if match else 0.0
        if score > best_score:
            best_val, best_score = canon, score
    if best_val is not None and key in ("cores", "conductor_class"):
        best_val = int(best_val)
    return (best_val, round(best_score, 2)) if best_score >= 0.6 else (None, round(best_score, 2))


def equivalence_flag(raw: Any) -> bool:
    if not raw:
        return False
    text = str(raw).lower()
    return any(p in text for p in synonyms()["equivalence_phrases"])


def norm_licence(raw: Any) -> Optional[str]:
    if not raw:
        return None
    m = re.search(r"(?:cm\s*/?\s*l)?\s*-?\s*(\d{10})", str(raw), re.I)
    return f"CM/L-{m.group(1)}" if m else None


def norm_standard(raw: Any) -> Optional[str]:
    if not raw:
        return None
    m = re.search(r"is\s*[:\-]?\s*(\d{3,5})", str(raw), re.I)
    return f"IS {m.group(1)}" if m else None


# ------------------------------------------------------------ public helpers
def normalise_value(key: str, value: Any, raw: Any = None) -> tuple[Any, float]:
    """Canonicalise one spec value. Uses `value` if already clean, else `raw`."""
    src = value if value not in (None, "") else raw
    if key in CANONICAL_UNITS:
        for cand in (raw, value):                        # vendor's words first, then model's reading
            if cand in (None, ""):
                continue
            v, score = parse_quantity(cand, key)
            if _plausible(key, v):
                return v, score
        v, _ = parse_quantity(src, key)
        return v, 0.3 if v is not None else 0.0          # implausible: forces a read-back
    if key in ("cores", "conductor_class"):
        for cand in (value, raw):
            if isinstance(cand, (int, float)):
                return int(cand), 1.0
            if isinstance(cand, str) and re.fullmatch(r"\s*\d\s*", cand):
                return int(cand), 1.0
    if key in CATEGORICAL:
        # prefer the raw words: they may contain the real material behind a label
        v, s = canon_category(key, raw or value)
        if v is None and value is not None:
            v, s = canon_category(key, value)
        return v, s
    if key == "isi_licence":
        v = norm_licence(src)
        return v, 1.0 if v else 0.0
    if key == "standard":
        v = norm_standard(src)
        return v, 1.0 if v else 0.0
    if key in ("unit_price_inr", "total_price_inr", "resistance_ohm_per_km", "oxygen_index",
               "temperature_index_c", "gst_pct", "quantity_coils"):
        v = parse_number(value) if value not in (None, "") else None
        return (v if v is not None else parse_number(raw)), 1.0
    return src, 1.0


def normalise_requirements(data: dict) -> RequirementSet:
    reqs = []
    for r in data["requirements"]:
        key = r["key"]
        val = r.get("value")
        if r.get("op", "eq") not in ("valid_licence", "proven"):
            val, _ = normalise_value(key, val, r.get("source_text"))
        reqs.append(Requirement(key=key, op=r.get("op", "eq"), value=val, via=r.get("via"),
                                unit=CANONICAL_UNITS.get(key), priority=r.get("priority", "mandatory"),
                                source_text=r.get("source_text", "")))
    return RequirementSet(request_id=data.get("request_id", "REQ"), requester=data.get("requester", ""),
                          item=data.get("item", ""), budget_inr=parse_number(data.get("budget_inr")),
                          requirements=reqs)


def normalise_quote(vendor_id: str, channel: str, data: dict, source_file: str = "",
                    line_confidence: Optional[dict[str, float]] = None) -> Quote:
    """Build the canonical Quote. Confidence = LLM confidence x normalisation score
    x (for photos) OCR confidence of the line the value came from."""
    fields: dict[str, Field_] = {}
    for key, f in (data.get("fields") or {}).items():
        raw = f.get("raw", "")
        value, norm_score = normalise_value(key, f.get("value"), raw)
        conf = float(f.get("confidence", 1.0)) * (norm_score if value is not None else 1.0)
        if line_confidence and raw and len(raw.strip()) >= 4:     # cross-check model vs OCR
            hit = process.extractOne(raw, list(line_confidence), scorer=fuzz.partial_ratio)
            if hit and hit[1] >= 80:
                conf = min(conf, line_confidence[hit[0]])
        flags = ["equivalence_clause"] if equivalence_flag(raw) else []
        if f.get("via"):
            flags.append(f"via:{f['via']}")
        fields[key] = Field_(value=value, unit=CANONICAL_UNITS.get(key), confidence=round(conf, 2),
                             raw=raw, flags=flags)

    _reconcile(fields)
    return Quote(vendor_id=vendor_id, vendor=data.get("vendor", vendor_id), channel=channel,
                 quote_ref=data.get("quote_ref", ""), brand=data.get("brand", ""),
                 fields=fields, source_file=source_file)


def _reconcile(fields: dict[str, Field_]) -> None:
    """Cross-check numbers against each other (self-consistency), as a buyer would."""
    coils = fields.get("quantity_coils")
    n = coils.value if coils and coils.value else None
    clen = fields.get("coil_length_m")
    qty = fields.get("quantity_m")

    # a small model often puts one coil's length in the total-length slot
    if n and n > 1 and qty and qty.value and (clen is None or not clen.value) and qty.value < 500:
        fields["coil_length_m"] = Field_(value=qty.value, unit="meter", confidence=qty.confidence, raw=qty.raw,
                                         flags=["interpreted as per-coil length"])
        clen = fields["coil_length_m"]
    if n and clen and clen.value:
        derived = n * clen.value
        if qty is None or qty.value is None or qty.value in (clen.value,) or qty.value < derived:
            fields["quantity_m"] = Field_(value=derived, unit="meter",
                                          confidence=min(coils.confidence, clen.confidence),
                                          raw=f"{coils.raw} × {clen.raw}", flags=["derived"])

    # total price: pick the reading (model's number or vendor's words) that matches qty × rate
    tot, rate = fields.get("total_price_inr"), fields.get("unit_price_inr")
    if tot and rate and n and rate.value:
        expected = n * rate.value
        cands = [c for c in (tot.value, parse_number(tot.raw)) if c]
        best = min(cands, key=lambda c: abs(c - expected), default=None)
        if best is not None and abs(best - expected) <= 0.01 * expected and best != tot.value:
            tot.value = best
            tot.flags = [*tot.flags, "reconciled with qty × rate"]


# What each kind of question allows the vendor's reply to change.
ASK_SCOPE = {
    "fr": {"fire_grade", "oxygen_index", "temperature_index_c", "fr_test_report"},
    "licen": {"isi_licence", "standard"},
    "datasheet": {"conductor_material", "conductor_detail", "resistance_ohm_per_km", "standard"},
    "resistance": {"conductor_material", "conductor_detail", "resistance_ohm_per_km"},
}


def scope_for(open_items: list[str]) -> set[str]:
    """Fields a reply may update, given what we asked. Read-backs name their field directly."""
    keys: set[str] = set()
    for item in open_items:
        low = item.lower()
        m = re.search(r"confirm (\w+)\s*=", item)
        if m:
            keys.add(m.group(1))
        m = re.search(r"confirm (\w+)$", item.strip())
        if m:
            keys.add(m.group(1))
        for word, ks in ASK_SCOPE.items():
            if word in low:
                keys |= ks
    return keys


def apply_updates(quote: Quote, reply_data: dict, source: str = "clarification",
                  allowed: Optional[set[str]] = None) -> list[str]:
    """Merge a vendor's clarification reply into a quote. Returns the keys that changed.

    Only fields in `allowed` (what we actually asked about) can change, so a model that
    over-extracts from the reply can't overwrite values that were already verified."""
    update = normalise_quote(quote.vendor_id, quote.channel, reply_data)
    changed = []
    for k, f in update.fields.items():
        if f.value is None or (allowed is not None and k not in allowed):
            continue
        old = quote.fields.get(k)
        if old is None or old.value != f.value or old.confidence < f.confidence:
            f.flags = [*f.flags, f"updated:{source}"]
            quote.fields[k] = f
            changed.append(k)
    _reconcile(quote.fields)
    return changed
