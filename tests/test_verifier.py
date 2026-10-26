"""Part 3: spec-claim verifier, rule engine, anomalies, ranking."""
import copy
import json
import os

import pytest

os.environ["LLM_PROVIDER"] = "cached"

from procureiq.config import CACHE_DIR  # noqa: E402
from procureiq.ingest import ingest_demo  # noqa: E402
from procureiq.schemas import Field_  # noqa: E402
from procureiq.tools.normalise import apply_updates  # noqa: E402
from procureiq.tools.rules import evaluate  # noqa: E402
from procureiq.tools.verifier import verify_claim  # noqa: E402


@pytest.fixture(scope="module")
def demo():
    return ingest_demo()


def labels(ev, vid):
    return {c.key: c.label for c in ev.by_id(vid).claims}


def test_slide7_scenario(demo):
    reqs, quotes, _ = demo
    ev = evaluate(reqs, quotes)
    a, b, c = ev.by_id("A"), ev.by_id("B"), ev.by_id("C")

    assert a.status == "COMPLIANT" and a.rank == 1
    assert set(labels(ev, "A").values()) == {"VERIFIED"}

    assert b.status == "EXCLUDED" and b.rank is None
    assert labels(ev, "B")["conductor_material"] == "SUBSTITUTED"
    assert labels(ev, "B")["isi_licence"] == "UNVERIFIABLE"
    assert {"PRICE_LOW", "FINE_PRINT"} <= {x.code for x in b.anomalies}

    assert c.status == "CONDITIONAL" and c.rank is None
    assert labels(ev, "C")["fire_grade"] == "UNVERIFIABLE"
    assert labels(ev, "C")["isi_licence"] == "UNVERIFIABLE"      # low confidence, NOT contradicted
    assert "GST_SLAB" in {x.code for x in c.anomalies}

    assert "18% cheaper but excluded" in ev.summary


def test_cheapest_is_not_recommended(demo):
    reqs, quotes, _ = demo
    ev = evaluate(reqs, quotes)
    cheapest = min(ev.quotes, key=lambda e: e.price_pre_tax)
    assert cheapest.rank is None


def test_clarification_makes_c_compliant(demo):
    reqs, quotes, _ = demo
    quotes = copy.deepcopy(quotes)
    c = next(q for q in quotes if q.vendor_id == "C")
    reply = json.loads((CACHE_DIR / "seed_vendor_c_clarification_reply.json").read_text(encoding="utf-8"))
    changed = apply_updates(c, reply)
    assert {"isi_licence", "oxygen_index", "gst_pct"} <= set(changed)

    ev = evaluate(reqs, quotes)
    assert ev.by_id("C").status == "COMPLIANT"
    assert labels(ev, "C")["fire_grade"] == "VERIFIED"
    assert labels(ev, "C")["isi_licence"] == "VERIFIED"
    assert ev.by_id("A").rank == 1 and ev.by_id("C").rank == 2


def test_aluminium_without_equivalence_is_contradicted(demo):
    reqs, quotes, _ = demo
    b = copy.deepcopy(next(q for q in quotes if q.vendor_id == "B"))
    b.fields["conductor_material"].flags = []
    assert verify_claim(reqs.get("conductor_material"), b).label == "CONTRADICTED"


def test_expired_or_wrong_brand_licence_is_contradicted(demo):
    reqs, quotes, _ = demo
    a = copy.deepcopy(next(q for q in quotes if q.vendor_id == "A"))
    a.fields["isi_licence"] = Field_(value="CM/L-4100055501", raw="CM/L-4100055501", confidence=0.95)
    r = verify_claim(reqs.get("isi_licence"), a)
    assert r.label == "CONTRADICTED" and "expired" in r.reason
    a.fields["isi_licence"] = Field_(value="CM/L-4100067890", raw="CM/L-4100067890", confidence=0.95)
    assert verify_claim(reqs.get("isi_licence"), a).label == "CONTRADICTED"   # AnnaWire's licence


def test_fr_values_below_minimum_are_contradicted(demo):
    reqs, quotes, _ = demo
    a = copy.deepcopy(next(q for q in quotes if q.vendor_id == "A"))
    a.fields["oxygen_index"] = Field_(value=26, raw="OI 26", confidence=0.95)
    assert verify_claim(reqs.get("fire_grade"), a).label == "CONTRADICTED"


def test_missing_data_is_never_a_pass(demo):
    reqs, quotes, _ = demo
    a = copy.deepcopy(next(q for q in quotes if q.vendor_id == "A"))
    del a.fields["fire_grade"]
    assert verify_claim(reqs.get("fire_grade"), a).label == "UNVERIFIABLE"
    a.brand = "UnknownBrand"
    assert verify_claim(reqs.get("cores"), a).label == "UNVERIFIABLE"


def test_arithmetic_anomaly(demo):
    reqs, quotes, _ = demo
    quotes = copy.deepcopy(quotes)
    a = next(q for q in quotes if q.vendor_id == "A")
    a.fields["total_price_inr"] = Field_(value=402000, raw="4,02,000", confidence=0.95)
    ev = evaluate(reqs, quotes)
    assert "ARITHMETIC" in {x.code for x in ev.by_id("A").anomalies}
