"""Shared test setup.

Logic tests run against the hand-written seed outputs only, so they give the same
result on every machine. tests/test_recorded.py checks the real recorded model run.
"""
import shutil

import pytest

from procureiq import audit, config, llm


@pytest.fixture(autouse=True)
def seed_only_cache(tmp_path, monkeypatch, request):
    monkeypatch.setenv("LLM_PROVIDER", "cached")
    monkeypatch.setattr(audit, "DB_PATH", tmp_path / "test.db")
    if request.node.get_closest_marker("recorded"):
        return
    cache = tmp_path / "cache"
    cache.mkdir()
    for f in config.CACHE_DIR.glob("seed_*.json"):
        shutil.copy(f, cache / f.name)
    monkeypatch.setattr(llm, "CACHE_DIR", cache)


def pytest_configure(config):
    config.addinivalue_line("markers", "recorded: run against the committed recordings of a real model run")
