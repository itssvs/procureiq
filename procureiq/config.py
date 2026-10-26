"""Central paths and settings."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional at import time
    pass

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "kb"
DATA_DIR = ROOT / "data"
DEMO_DIR = DATA_DIR / "demo"
CACHE_DIR = DATA_DIR / "cache"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cached").lower()   # read live via llm.provider()
MAX_CLARIFY_ROUNDS = int(os.getenv("MAX_CLARIFY_ROUNDS", "3"))

# Demo scenario: which file belongs to which vendor
DEMO_REQUEST = DEMO_DIR / "request_email.txt"
DEMO_QUOTES = {
    "A": {"vendor": "Kaveri Cables & Wires", "channel": "pdf", "file": DEMO_DIR / "vendor_a_quote.pdf"},
    "B": {"vendor": "Marina Electric Traders", "channel": "whatsapp", "file": DEMO_DIR / "vendor_b_whatsapp.txt"},
    "C": {"vendor": "Anna Nagar Wire House", "channel": "photo", "file": DEMO_DIR / "vendor_c_handwritten.png"},
}
DEMO_CLARIFICATION_REPLIES = {"C": DEMO_DIR / "vendor_c_clarification_reply.txt"}
