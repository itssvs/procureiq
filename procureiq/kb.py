"""Read-only access to the knowledge base (kb/cables.yaml)."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import yaml

from .config import KB_DIR


@lru_cache(maxsize=1)
def kb() -> dict:
    return yaml.safe_load((KB_DIR / "cables.yaml").read_text(encoding="utf-8"))


def datasheet_for(brand: str) -> Optional[dict]:
    b = (brand or "").lower().replace(" ", "")
    return next((d for d in kb()["datasheets"] if d["brand"].lower().replace(" ", "") == b), None)


def licence(licence_no: str) -> Optional[dict]:
    return next((c for c in kb()["cert_registry"] if c["licence_no"] == licence_no), None)


def resistance_limits(size_sqmm: float, conductor_class: int) -> dict[str, dict]:
    """{'copper': row, 'aluminium': row} for this size and class."""
    out = {}
    for r in kb()["conductor_resistance"]:
        if r["size_sqmm"] == size_sqmm and r["conductor_class"] == conductor_class:
            out[r["material"]] = r
    return out


def fire_grade(grade: str) -> Optional[dict]:
    return next((g for g in kb()["fire_grades"] if g["grade"] == grade), None)


def standard(code_prefix: str) -> Optional[dict]:
    c = (code_prefix or "").replace(" ", "").lower()
    return next((s for s in kb()["standards"] if s["code"].replace(" ", "").lower().startswith(c)), None)


def market_reference() -> list[dict]:
    return kb()["market_reference"]
