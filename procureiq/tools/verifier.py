"""Spec-Claim Verifier — the novel core (Slide 5).

Every vendor claim that answers a buyer requirement gets exactly one label:

  VERIFIED      claim matches evidence in the knowledge base
  UNVERIFIABLE  no evidence on file, or the value itself was read with low
                confidence. Never guessed, never a pass.
  CONTRADICTED  claim conflicts with a standard, datasheet or registry entry
  SUBSTITUTED   "or equivalent" wording hides a different material / rating

Pure, deterministic Python. Each result cites evidence ids from kb/cables.yaml
or the vendor's own words, so a human can audit every decision.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from .. import kb
from ..schemas import Quote, Requirement
from .normalise import canon_category

Label = Literal["VERIFIED", "UNVERIFIABLE", "CONTRADICTED", "SUBSTITUTED"]
CONF_THRESHOLD = 0.75

SPEC_CLAIMS = ["conductor_material", "size_sqmm", "cores", "voltage_grade_kv", "insulation",
               "standard", "isi_licence", "fire_grade", "conductor_class"]


class Evidence(BaseModel):
    id: str                 # kb id (e.g. RES-AL-2.5-C2) or QUOTE:<field>
    text: str


class ClaimResult(BaseModel):
    key: str
    priority: str
    required: Any = None
    claimed: Any = None
    label: Label
    reason: str
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Optional[float] = None
    action: Optional[str] = None       # what would resolve it, e.g. "request FR test report"


def _ev_quote(q: Quote, key: str) -> Evidence:
    f = q.fields[key]
    return Evidence(id=f"QUOTE:{key}", text=f'"{f.raw}" (confidence {f.confidence:.2f})')


def _result(req: Requirement, q: Quote, label: Label, reason: str, evidence=None,
            action: Optional[str] = None) -> ClaimResult:
    f = q.fields.get(req.key)
    return ClaimResult(key=req.key, priority=req.priority, required=req.value,
                       claimed=None if f is None else f.value, label=label, reason=reason,
                       evidence=evidence or [], confidence=None if f is None else f.confidence,
                       action=action)


# ------------------------------------------------------------- claim checkers
def _check_material(req: Requirement, q: Quote, ds: Optional[dict]) -> ClaimResult:
    claimed = q.val("conductor_material")
    flagged = "equivalence_clause" in q.fields["conductor_material"].flags
    evidence = [_ev_quote(q, "conductor_material")]
    actual: list[tuple[str, Evidence]] = []        # independent signals of the real material

    if "conductor_detail" in q.fields:
        mat, _ = canon_category("conductor_material", q.fields["conductor_detail"].raw)
        if mat:
            actual.append((mat, _ev_quote(q, "conductor_detail")))
    res = q.val("resistance_ohm_per_km")
    size, cls = q.val("size_sqmm"), q.val("conductor_class", 2)
    if res is not None and size is not None:
        lim = kb.resistance_limits(size, cls)
        cu = lim.get("copper")
        if cu and res > cu["max_ohm_per_km"] * 1.05:
            al = lim.get("aluminium")
            mat = "aluminium" if al and res <= al["max_ohm_per_km"] * 1.05 else "not copper"
            actual.append((mat, Evidence(id=cu["id"], text=f"Copper {size} sq mm class {cls} max "
                                         f"{cu['max_ohm_per_km']} ohm/km; quoted {res} ohm/km")))
    if ds and ds.get("conductor_material"):
        actual.append((ds["conductor_material"], Evidence(
            id=ds["id"], text=f"{ds['product']}: conductor {ds['conductor_material']}")))

    if claimed != req.value:
        return _result(req, q, "CONTRADICTED", f"Quoted {claimed}, buyer requires {req.value}.", evidence)

    conflicts = [(m, e) for m, e in actual if m != req.value]
    if conflicts:
        evidence += [e for _, e in conflicts]
        found = conflicts[0][0]
        if flagged:
            return _result(req, q, "SUBSTITUTED",
                           f'"{q.fields["conductor_material"].raw}" — the "or equivalent" hides '
                           f"{'an' if found[0] in 'aeiou' else 'a'} {found} conductor ({len(conflicts)} independent signal(s)).", evidence)
        return _result(req, q, "CONTRADICTED", f"Claimed {claimed} but evidence shows {found}.", evidence)

    if actual:
        return _result(req, q, "VERIFIED", f"{claimed} confirmed by {len(actual)} source(s).",
                       evidence + [e for _, e in actual])
    return _result(req, q, "UNVERIFIABLE", "No datasheet or test value on file to confirm the material.",
                   evidence, action="Request datasheet or conductor resistance test value")


def _check_against_datasheet(req: Requirement, q: Quote, ds: Optional[dict], ds_key: str) -> ClaimResult:
    claimed = q.val(req.key)
    evidence = [_ev_quote(q, req.key)]
    if claimed != req.value:
        return _result(req, q, "CONTRADICTED", f"Quoted {claimed}, buyer requires {req.value}.", evidence)
    if not ds:
        return _result(req, q, "UNVERIFIABLE", f"No datasheet on file for brand '{q.brand}'.", evidence,
                       action="Request product datasheet")
    ds_val = ds.get(ds_key)
    if req.key == "insulation" and ds_val:
        ds_val, _ = canon_category("insulation", ds_val)
    ev = Evidence(id=ds["id"], text=f"{ds['product']}: {ds_key} = {ds.get(ds_key)}")
    if ds_val is None:
        return _result(req, q, "UNVERIFIABLE", f"Datasheet does not state {ds_key}.", evidence + [ev],
                       action=f"Request evidence for {req.key}")
    if ds_val != claimed:
        return _result(req, q, "CONTRADICTED", f"Quote says {claimed}; datasheet says {ds_val}.", evidence + [ev])
    return _result(req, q, "VERIFIED", f"{claimed} matches the brand datasheet.", evidence + [ev])


def _check_standard(req: Requirement, q: Quote, ds: Optional[dict]) -> ClaimResult:
    claimed = q.val("standard")
    evidence = [_ev_quote(q, "standard")]
    std = kb.standard(claimed)
    if not std:
        return _result(req, q, "UNVERIFIABLE", f"{claimed} is not in the knowledge base.", evidence)
    evidence.append(Evidence(id=std["id"], text=f"{std['code']}: {std['title']}"))
    if not str(claimed).replace(" ", "").lower().startswith(str(req.value).replace(" ", "").lower()):
        return _result(req, q, "CONTRADICTED", f"Quoted {claimed}, buyer requires {req.value}.", evidence)
    if ds and ds.get("standard") and ds["standard"].replace(" ", "").lower().startswith(
            str(req.value).replace(" ", "").lower()):
        evidence.append(Evidence(id=ds["id"], text=f"{ds['product']}: certified to {ds['standard']}"))
        return _result(req, q, "VERIFIED", f"Brand datasheet certifies {ds['standard']}.", evidence)
    if ds:
        evidence.append(Evidence(id=ds["id"], text=f"{ds['product']}: no standard certification on record"))
    return _result(req, q, "UNVERIFIABLE", f'Vendor says "{claimed}" but no certification is on record.',
                   evidence, action="Request BIS licence / test certificate")


def _check_licence(req: Requirement, q: Quote) -> ClaimResult:
    lic = q.val("isi_licence")
    if "isi_licence" not in q.fields or lic is None:
        raw = q.fields["isi_licence"].raw if "isi_licence" in q.fields else "not mentioned"
        return _result(req, q, "UNVERIFIABLE", f'No licence number given ("{raw}").',
                       [_ev_quote(q, "isi_licence")] if "isi_licence" in q.fields else [],
                       action="Request ISI licence number")
    evidence = [_ev_quote(q, "isi_licence")]
    entry = kb.licence(lic)
    if entry is None:
        return _result(req, q, "CONTRADICTED", f"{lic} is not in the BIS licence registry.",
                       evidence + [Evidence(id="CERT-REGISTRY", text=f"no entry for {lic}")])
    ev = Evidence(id=entry["id"], text=f"{entry['licence_no']}: {entry['holder']} ({entry['brand']}), "
                                       f"{entry['standard']}, {entry['status']} until {entry['valid_until']}")
    if entry["status"] != "valid":
        return _result(req, q, "CONTRADICTED", f"Licence {lic} is {entry['status']}.", evidence + [ev])
    if q.brand and entry["brand"].lower() != q.brand.lower():
        return _result(req, q, "CONTRADICTED", f"Licence belongs to {entry['brand']}, quoted brand is {q.brand}.",
                       evidence + [ev])
    return _result(req, q, "VERIFIED", f"Valid licence held by {entry['holder']}.", evidence + [ev])


def _check_fire(req: Requirement, q: Quote, ds: Optional[dict]) -> ClaimResult:
    claimed = q.val("fire_grade")
    evidence = [_ev_quote(q, "fire_grade")]
    if claimed != req.value:
        return _result(req, q, "CONTRADICTED", f"Quoted {claimed}, buyer requires {req.value}.", evidence)
    grade = kb.fire_grade(req.value)
    need = grade["requires"]
    ev_grade = Evidence(id=grade["id"], text=f"{grade['grade']} requires oxygen index ≥ "
                        f"{need['oxygen_index_min']}, temperature index ≥ {need['temperature_index_min_c']} °C")

    # evidence from the vendor's own test values first, then the brand datasheet
    oi, ti, src = q.val("oxygen_index"), q.val("temperature_index_c"), "quote"
    if (oi is None or ti is None) and ds and ds.get("fire_grade") == req.value:
        oi, ti, src = ds.get("oxygen_index"), ds.get("temperature_index_c"), "datasheet"
    if oi is None or ti is None:
        return _result(req, q, "UNVERIFIABLE",
                       f'"{claimed}" claimed, but no oxygen / temperature index or test report on file.',
                       evidence + [ev_grade] + ([Evidence(id=ds["id"], text=f"{ds['product']}: fire grade not stated")]
                                                if ds else []),
                       action="Request FR type test report (oxygen index, temperature index)")
    for k in ("oxygen_index", "temperature_index_c", "fr_test_report"):
        if k in q.fields:
            evidence.append(_ev_quote(q, k))
    if ds and src == "datasheet":
        evidence.append(Evidence(id=ds["id"], text=f"{ds['product']}: OI {oi}, TI {ti} °C"))
    if oi < need["oxygen_index_min"] or ti < need["temperature_index_min_c"]:
        return _result(req, q, "CONTRADICTED", f"Test values OI {oi:g}, TI {ti:g} °C are below the {req.value} minimum.",
                       evidence + [ev_grade])
    return _result(req, q, "VERIFIED", f"OI {oi:g}, TI {ti:g} °C meet {req.value} ({src}).", evidence + [ev_grade])


DATASHEET_KEYS = {"size_sqmm": "size_sqmm", "cores": "cores", "voltage_grade_kv": "voltage_grade_kv",
                  "insulation": "insulation", "conductor_class": "conductor_class"}


def verify_claim(req: Requirement, q: Quote) -> ClaimResult:
    f = q.fields.get(req.key)
    # 1) gate on presence and read confidence: an uncertain value is never judged
    if req.key != "isi_licence" and (f is None or f.value is None):
        return _result(req, q, "UNVERIFIABLE", "Vendor did not state this.",
                       action=f"Ask vendor to confirm {req.key}")
    if f is not None and f.value is not None and f.confidence < CONF_THRESHOLD:
        return _result(req, q, "UNVERIFIABLE",
                       f'Read as "{f.value}" from "{f.raw}" at confidence {f.confidence:.2f}; '
                       "not judged until the vendor confirms.", [_ev_quote(q, req.key)],
                       action=f"Read-back: please confirm {req.key} = {f.value}")
    # 2) check against evidence
    ds = kb.datasheet_for(q.brand)
    if req.key == "conductor_material":
        return _check_material(req, q, ds)
    if req.key == "standard":
        return _check_standard(req, q, ds)
    if req.key == "isi_licence":
        return _check_licence(req, q)
    if req.key == "fire_grade":
        return _check_fire(req, q, ds)
    return _check_against_datasheet(req, q, ds, DATASHEET_KEYS[req.key])
