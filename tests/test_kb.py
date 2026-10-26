"""Sanity checks on the knowledge base and demo inputs (Part 1)."""
import yaml

from procureiq.config import DEMO_QUOTES, DEMO_REQUEST, KB_DIR


def load(name):
    return yaml.safe_load((KB_DIR / name).read_text(encoding="utf-8"))


def test_kb_ids_are_unique():
    kb = load("cables.yaml")
    ids = [e["id"] for section in ("standards", "conductor_resistance", "fire_grades",
                                   "cert_registry", "datasheets", "market_reference")
           for e in kb[section]]
    assert len(ids) == len(set(ids)), "duplicate evidence ids in kb/cables.yaml"
    assert len(ids) >= 15


def test_copper_and_aluminium_resistance_differ():
    kb = load("cables.yaml")
    res = {(r["material"], r["conductor_class"]): r["max_ohm_per_km"]
           for r in kb["conductor_resistance"] if r["size_sqmm"] == 2.5}
    assert res[("aluminium", 2)] > res[("copper", 2)] * 1.5


def test_synonyms_cover_materials():
    syn = load("synonyms.yaml")
    assert {"copper", "aluminium"} <= set(syn["conductor_material"])
    assert "or equivalent" in syn["equivalence_phrases"]


def test_demo_inputs_exist():
    assert DEMO_REQUEST.exists()
    for vendor in DEMO_QUOTES.values():
        assert vendor["file"].exists(), f"missing {vendor['file']} — run scripts/generate_demo_data.py"
