"""Part 2: normaliser and ingestion (runs offline from the seed cache)."""
import os

from procureiq.tools.normalise import (canon_category, equivalence_flag, norm_licence,
                                       parse_number, parse_quantity)


def test_indian_numbers():
    assert parse_number("4,55,000") == 455000
    assert parse_number("Rs 3.44 L") == 344000
    assert parse_number("₹4.2 lakh") == 420000
    assert parse_number("22,750 / coil") == 22750


def test_units():
    assert parse_quantity("1100V", "voltage_grade_kv")[0] == 1.1
    assert parse_quantity("2 weeks", "delivery_days")[0] == 14
    assert parse_quantity("90 mtr", "coil_length_m")[0] == 90
    assert parse_quantity("2.5 sq mm", "size_sqmm")[0] == 2.5
    assert parse_quantity("2 yrs", "warranty_years")[0] == 2


def test_categories():
    assert canon_category("conductor_material", "AL 1350 grade")[0] == "aluminium"
    assert canon_category("conductor_material", "electrolytic grade copper")[0] == "copper"
    assert canon_category("fire_grade", "FR house wire")[0] == "FR"
    assert canon_category("conductor_class", "Class 2 stranded")[0] == 2


def test_equivalence_and_licence():
    assert equivalence_flag("2.5 sqmm copper or equivalent FR")
    assert not equivalence_flag("electrolytic grade copper")
    assert norm_licence("IS! CM/L-4100067390") == "CM/L-4100067390"


def test_demo_ingest_offline():
    os.environ["LLM_PROVIDER"] = "cached"
    from procureiq.ingest import ingest_demo

    reqs, quotes, _ = ingest_demo()
    assert len(reqs.requirements) == 12
    by = {q.vendor_id: q for q in quotes}
    assert "equivalence_clause" in by["B"].fields["conductor_material"].flags
    assert by["B"].val("total_price_inr") == 344000
    assert "isi_licence" in by["C"].low_confidence()
    assert all(q.val("quantity_m") == 1800 for q in quotes)
