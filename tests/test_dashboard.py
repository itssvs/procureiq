"""Part 5: the Streamlit dashboard renders and its human gate works."""
import os

import pytest

os.environ["LLM_PROVIDER"] = "cached"

from streamlit.testing.v1 import AppTest  # noqa: E402

from procureiq import audit  # noqa: E402
from procureiq.config import ROOT  # noqa: E402

APP = str(ROOT / "app.py")


def test_dashboard_run_and_approve():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    at.sidebar.button[0].click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Compliant"] == "2" and metrics["Excluded"] == "1"
    next(b for b in at.button if b.label == "Approve award").click().run()
    assert not at.exception
    assert "Kaveri Cables" in at.success[0].value


def test_dashboard_escalation_when_vendor_silent():
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.toggle[0].set_value(False).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Escalated" in w.value for w in at.warning)
