"""LLM extraction: messy text -> structured JSON (the LLM *proposes*).

Two jobs:
  extract_requirements(email)  -> RequirementSet  (~12 specs, mandatory/preferred)
  extract_quote(file, channel) -> Quote           (vendor claims with confidence)

Output is schema-checked, then normalised deterministically.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from ..llm import extract_json
from ..schemas import EVIDENCE_KEYS, SPEC_KEYS, Quote, RequirementSet
from .normalise import normalise_quote, normalise_requirements
from .parse import ParsedDoc, parse_any


def _file_key(prefix: str, path: Path) -> str:
    digest = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]
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

Also return request_id (short slug), requester, item (one line), budget_inr (number or null).

Return only JSON shaped like:
{{"request_id": "...", "requester": "...", "item": "...", "budget_inr": 450000,
  "requirements": [{{"key": "...", "op": "eq", "value": "...", "priority": "mandatory",
                     "source_text": "..."}}]}}

PURCHASE REQUEST:
---
{text}
---"""


def _check_requirements(data: dict) -> None:
    assert isinstance(data.get("requirements"), list) and data["requirements"], "no requirements"
    for r in data["requirements"]:
        assert r.get("key") in SPEC_KEYS, f"unknown key {r.get('key')}"
        assert r.get("priority") in ("mandatory", "preferred"), "bad priority"


def extract_requirements(path: Path) -> tuple[RequirementSet, str]:
    doc = parse_any(path, "email")
    keys = "\n".join(f"  {k}: {v}" for k, v in SPEC_KEYS.items())
    data, source = extract_json(REQ_PROMPT.format(keys=keys, text=doc.text),
                                cache_key=_file_key("requirements", path),
                                validator=_check_requirements, seed_key=_seed_key(path))
    return normalise_requirements(data), source


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

Also return vendor (company name), quote_ref, brand.

Return only JSON: {{"vendor": "...", "quote_ref": "...", "brand": "...",
"fields": {{"<key>": {{"raw": "...", "value": ..., "confidence": 0.9}}}}}}

QUOTATION ({channel}):
---
{text}
---"""


def _check_quote(data: dict) -> None:
    fields = data.get("fields")
    assert isinstance(fields, dict) and fields, "no fields"
    allowed = set(SPEC_KEYS) | set(EVIDENCE_KEYS)
    for k, f in fields.items():
        assert k in allowed, f"unknown field {k}"
        assert 0 <= float(f.get("confidence", 1)) <= 1, "confidence out of range"


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
    quote = normalise_quote(vendor_id, channel, data, source_file=path.name,
                            line_confidence=doc.line_confidence or None)
    return quote, doc, source


def extract_reply(vendor_id: str, path: Path) -> tuple[dict, str]:
    """Read a vendor's clarification reply into field updates (same schema as quotes)."""
    doc = parse_any(Path(path), "email")
    keys = "\n".join(f"  {k}: {v}" for k, v in {**SPEC_KEYS, **EVIDENCE_KEYS}.items())
    return extract_json(QUOTE_PROMPT.format(keys=keys, channel="clarification reply", text=doc.text),
                        cache_key=_file_key(f"reply_{vendor_id}", path), validator=_check_quote,
                        seed_key=_seed_key(path))


def dump(obj) -> str:
    return json.dumps(obj.model_dump() if hasattr(obj, "model_dump") else obj, indent=2, ensure_ascii=False)
