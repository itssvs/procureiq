"""LLM extraction: messy text -> structured JSON (the LLM *proposes*).

Two jobs:
  extract_requirements(email)  -> RequirementSet  (~12 specs, mandatory/preferred)
  extract_quote(file, channel) -> Quote           (vendor claims with confidence)

Output is schema-checked, then normalised deterministically.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from ..llm import extract_json
from ..schemas import EVIDENCE_KEYS, SPEC_KEYS, Quote, RequirementSet
from .normalise import normalise_quote, normalise_requirements
from .parse import ParsedDoc, parse_any


TEXT_SUFFIXES = {".txt", ".eml", ".csv", ".md"}


def _file_key(prefix: str, path: Path) -> str:
    """Cache key from file content. Text files are hashed with normalised line endings so a
    recording made on Windows (CRLF checkout) replays on Linux (LF) and vice versa."""
    data = Path(path).read_bytes()
    if Path(path).suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    digest = hashlib.sha1(data).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _seed_key(path: Path) -> str:
    return f"seed_{Path(path).stem}"


# -------------------------------------------------------------- requirements
REQ_PROMPT = """You are a procurement analyst. Read the purchase request below and
extract the buyer's requirements as JSON.

Use ONLY these requirement keys (skip any the buyer did not mention):
{keys}

For each requirement give:
  key, op ("eq" | "gte" | "lte" | "valid_licence" | "proven"),
  value (number or short string, null for valid_licence),
  priority ("mandatory" if the buyer says must / only / required / audit will
            check; "preferred" if they say preferred / would be good / ideally),
  source_text (the exact words from the email).
Use op "valid_licence" for "ISI marked with a valid licence".
Use op "proven" with value "FR" for a fire grade that must be evidenced.
Use op "lte" for deadlines ("within 14 days") and "gte" for minimums ("at least").
Each key appears at most once. "ISI marked / licence" is isi_licence, not insulation.
"as per IS 694" is standard. "PVC insulated" is insulation.

Also return request_id (short slug), requester, item (one line), budget_inr (number or null).

Return only JSON shaped like:
{{"request_id": "...", "requester": "...", "item": "...", "budget_inr": 450000,
  "requirements": [{{"key": "...", "op": "eq", "value": "...", "priority": "mandatory",
                     "source_text": "..."}}]}}

PURCHASE REQUEST:
---
{text}
---"""


OPS = {"eq", "gte", "lte", "valid_licence", "proven"}


def _check_requirements(data: dict) -> dict:
    """Clean a model's requirement list. Small models drift, so we repair rather than reject:
    unknown keys are dropped, bad priorities default to mandatory, duplicates are merged."""
    items = data.get("requirements")
    if not isinstance(items, list):
        raise ValueError("no requirements list")
    clean, seen = [], set()
    for r in items:
        if not isinstance(r, dict):
            continue
        key = str(r.get("key", "")).strip().lower()
        if r.get("op") == "valid_licence" or ("licen" in str(r.get("source_text", "")).lower()
                                              and key in ("insulation", "standard")):
            key = "isi_licence"                     # model filed the licence rule under another key
        if key not in SPEC_KEYS or key in seen:
            continue
        seen.add(key)
        op = r.get("op") if r.get("op") in OPS else ("valid_licence" if key == "isi_licence" else "eq")
        if key == "fire_grade" and op == "eq":
            op = "proven"
        pr = str(r.get("priority", "")).lower()
        clean.append({"key": key, "op": op, "value": r.get("value"),
                      "priority": pr if pr in ("mandatory", "preferred") else "mandatory",
                      "source_text": str(r.get("source_text", ""))[:300]})
    if len(clean) < 5:
        raise ValueError(f"only {len(clean)} usable requirements")
    return {**{k: data.get(k) for k in ("request_id", "requester", "item", "budget_inr")}, "requirements": clean}


# Safety net for requirements a small model merges or drops. Only adds missing keys,
# never overrides the model, and every addition is marked so it's visible.
REQ_HINTS = [
    ("standard", "eq", re.compile(r"\bIS\s*:?\s*(\d{3,5})\b"), lambda m: f"IS {m.group(1)}"),
    ("isi_licence", "valid_licence", re.compile(r"\bISI\b[^.\n]*licen", re.I), lambda m: None),
    ("insulation", "eq", re.compile(r"\b(PVC|XLPE)\b", re.I), lambda m: m.group(1).upper()),
    ("fire_grade", "proven", re.compile(r"\bFR\b|flame[- ]retardant", re.I), lambda m: "FR"),
]
PREFERRED_WORDS = re.compile(r"prefer|would be good|ideally|nice to have", re.I)


def _requirement_hints(data: dict, text: str) -> dict:
    have = {r["key"] for r in data["requirements"]}
    sentences = re.split(r"(?<=[.\n])\s+", text)
    for key, op, rx, val in REQ_HINTS:
        if key in have:
            continue
        for sent in sentences:
            m = rx.search(sent)
            if m:
                data["requirements"].append({
                    "key": key, "op": op, "value": val(m), "source_text": sent.strip()[:300], "via": "rules",
                    "priority": "preferred" if PREFERRED_WORDS.search(sent) else "mandatory"})
                break
    return data


def extract_requirements(path: Path) -> tuple[RequirementSet, str]:
    doc = parse_any(path, "email")
    keys = "\n".join(f"  {k}: {v}" for k, v in SPEC_KEYS.items())
    data, source = extract_json(REQ_PROMPT.format(keys=keys, text=doc.text),
                                cache_key=_file_key("requirements", path),
                                validator=_check_requirements, seed_key=_seed_key(path))
    data = _check_requirements(data)                # also repairs recorded outputs on replay
    return normalise_requirements(_requirement_hints(data, doc.text)), source


# --------------------------------------------------------------------- quotes
QUOTE_PROMPT = """You are reading a vendor's quotation for electrical wire. It may be
a PDF, a WhatsApp chat, OCR of a handwritten note (expect OCR errors), or a transcript.

Extract every claim the vendor makes. Use these field keys (omit absent ones):
{keys}

For each field return {{"raw": exact vendor words, "value": your reading,
"confidence": 0.0-1.0}}.
Rules:
- Copy the vendor's words into raw EXACTLY, including phrases like "or equivalent".
- Never invent a value that is not in the text. Missing -> omit the field.
- Lower the confidence when text is unclear, OCR looks garbled, or digits may be misread.
- If the conductor is described twice (e.g. "copper or equivalent" and later "AL 1350"),
  put the first in conductor_material and the second in conductor_detail.
- Keep units in value as written ("1100 V", "90 m", "12 days", "2 yrs").
- quantity_m is the TOTAL length of the whole order. One coil's length goes in coil_length_m.
- Indian numbers: "3,44,000" = 344000 and "4.2 L" = 420000.

Also return vendor (company name), quote_ref, brand.

Return only JSON: {{"vendor": "...", "quote_ref": "...", "brand": "...",
"fields": {{"<key>": {{"raw": "...", "value": ..., "confidence": 0.9}}}}}}

QUOTATION ({channel}):
---
{text}
---"""


ALLOWED = set(SPEC_KEYS) | set(EVIDENCE_KEYS)
FILLER = {"n/a", "na", "none", "null", "-", "--", "not mentioned", "not specified", "not stated",
          "unknown", "not available", "nil", ""}


def _check_quote(data: dict) -> dict:
    """Clean a model's quote extraction: accept flat values, drop unknown keys, clamp confidence."""
    fields = data.get("fields")
    if not isinstance(fields, dict):           # model put fields at the top level
        fields = {k: v for k, v in data.items() if k in ALLOWED}
    clean = {}
    for k, f in fields.items():
        k = str(k).strip().lower()
        if k not in ALLOWED or f in (None, "", {}):
            continue
        if not isinstance(f, dict):
            f = {"raw": str(f), "value": f, "confidence": 0.8}
        if str(f.get("value", "")).strip().lower() in FILLER and str(f.get("raw", "")).strip().lower() in FILLER:
            continue                                   # model filler, not a vendor claim
        try:
            conf = min(max(float(f.get("confidence", 0.8)), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.8
        clean[k] = {"raw": str(f.get("raw") or f.get("value") or ""), "value": f.get("value"), "confidence": conf}
    if len(clean) < 3:
        raise ValueError(f"only {len(clean)} usable fields")
    return {"vendor": data.get("vendor", ""), "quote_ref": data.get("quote_ref", ""),
            "brand": data.get("brand", ""), "fields": clean}


# Regex safety net: critical evidence a small model sometimes drops. Only fills gaps,
# never overrides the model, and is marked so the dashboard can show it.
HINTS = {
    "isi_licence": re.compile(r"(CM\s*/\s*L\s*-?\s*\d{10})", re.I),
    "resistance_ohm_per_km": re.compile(r"(\d+(?:\.\d+)?)\s*(?:ohm|Ω)\s*/\s*km", re.I),
    "oxygen_index": re.compile(r"(?:oxygen index|\bOI\b)\s*[:\-]?\s*(\d{2})", re.I),
    "temperature_index_c": re.compile(r"(?:temperature index|\bTI\b)\s*[:\-]?\s*(\d{3})", re.I),
    "fr_test_report": re.compile(r"((?:NABL|TR)[-\w]*-\d{3,})", re.I),
}


def _apply_hints(data: dict, text: str) -> dict:
    fields = data.setdefault("fields", {})
    for key, rx in HINTS.items():
        if key in fields and fields[key].get("value") not in (None, ""):
            continue
        m = rx.search(text or "")
        if m:
            fields[key] = {"raw": m.group(0), "value": m.group(1), "confidence": 0.7, "via": "regex"}
    return data


def extract_quote(vendor_id: str, path: Path, channel: str) -> tuple[Quote, ParsedDoc, str]:
    path = Path(path)
    doc = parse_any(path, channel)
    keys = "\n".join(f"  {k}: {v}" for k, v in {**SPEC_KEYS, **EVIDENCE_KEYS}.items())
    text = doc.text or "(no OCR text available; read the attached image)"
    image: Optional[Path] = path if channel == "photo" else None
    data, source = extract_json(QUOTE_PROMPT.format(keys=keys, channel=channel, text=text),
                                cache_key=_file_key(f"quote_{vendor_id}", path),
                                image_path=image, validator=_check_quote,
                                seed_key=_seed_key(path))
    data = _apply_hints(_check_quote(data), doc.text)
    quote = normalise_quote(vendor_id, channel, data, source_file=path.name,
                            line_confidence=doc.line_confidence or None)
    return quote, doc, source


def extract_reply(vendor_id: str, path: Path) -> tuple[dict, str]:
    """Read a vendor's clarification reply into field updates (same schema as quotes)."""
    doc = parse_any(Path(path), "email")
    keys = "\n".join(f"  {k}: {v}" for k, v in {**SPEC_KEYS, **EVIDENCE_KEYS}.items())
    data, src = extract_json(QUOTE_PROMPT.format(keys=keys, channel="clarification reply", text=doc.text),
                             cache_key=_file_key(f"reply_{vendor_id}", path), validator=_check_quote,
                             seed_key=_seed_key(path))
    return _apply_hints(data, doc.text), src


def dump(obj) -> str:
    return json.dumps(obj.model_dump() if hasattr(obj, "model_dump") else obj, indent=2, ensure_ascii=False)
