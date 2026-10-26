# ProcureIQ

**Autonomous requirement-to-quotation procurement agent.**
*Not the cheapest quote. The cheapest quote that actually meets the spec.*

Project Eclipse · Challenge 04: Procurement

ProcureIQ reads a buyer's request, ingests vendor quotes from messy channels
(PDF, WhatsApp text, handwritten photos), and checks **every vendor claim
against evidence** before price is considered. Each claim gets one label:

| Label | Meaning |
|---|---|
| ✅ **VERIFIED** | Claim matches evidence in the knowledge base |
| ❔ **UNVERIFIABLE** | No evidence on file — never guessed, never a pass |
| ❌ **CONTRADICTED** | Claim conflicts with a standard, datasheet or registry |
| ⚠️ **SILENTLY SUBSTITUTED** | "Or equivalent" hides a different material, rating or certification |

Only quotes with no contradicted or substituted mandatory claim are ranked.
A human approves every award — the agent never auto-awards.

**Rules decide, LLM proposes:** the LLM only turns messy text into structured
data. Every compliance decision is deterministic Python with cited evidence.

## Status

| Part | Scope | Status |
|---|---|---|
| 1 | Skeleton, knowledge base, demo data | ✅ done |
| 2 | Ingestion, extraction, normalisation | ✅ done |
| 3 | Spec-claim verifier, rule engine, ranking | ⏳ |
| 4 | LangGraph agent loop, FastAPI | ⏳ |
| 5 | Streamlit approval dashboard | ⏳ |
| 6 | Deployment and docs | ⏳ |

## Repository layout

```
procureiq/            Python package (agent, tools)
kb/                   Knowledge base: standards, datasheets, cert registry, synonyms
data/demo/            Demo inputs (request email + 3 vendor quotes + clarification reply)
data/cache/           Cached LLM outputs so the demo runs offline
scripts/              Demo data generator
tests/                Unit tests
```

## Demo scenario

A facilities manager emails a request for 1800 m of 2.5 sq mm copper FR wire.
Three vendors reply:

| Vendor | Channel | Quote | What the agent should find |
|---|---|---|---|
| A — Kaveri Cables | PDF | ₹4.20 L | All claims verified → compliant, rank 1 |
| B — Marina Electric | WhatsApp | ₹3.44 L | "Copper or equivalent" but aluminium conductor → **substituted**, no ISI licence → excluded |
| C — Anna Nagar Wire | Handwritten photo | ₹4.55 L | FR grade unverifiable → clarification → proof received → re-verified |

Vendor B is 18% cheaper than A and is still excluded from the ranking.

> Knowledge base note: standard references describe real Indian Standards
> (IS 694, IS 8130, IS 10810). Brand datasheets, the BIS licence registry and
> market prices are **fictional demo data**.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # default LLM_PROVIDER=cached needs no key
python scripts/generate_demo_data.py
```

Tesseract OCR (optional, used only as a fallback for photos):
Windows installer from the UB Mannheim build; Linux `apt install tesseract-ocr`.

## LLM providers (all free)

Set `LLM_PROVIDER` in `.env`:

- `cached` — no key, uses saved outputs in `data/cache/` (default, demo-safe)
- `gemini` — free key from Google AI Studio; also reads the handwritten photo
- `groq` — free key from console.groq.com

## Run the ingestion step (Part 2)

```bash
python -m procureiq.ingest     # prints 12 extracted specs + 3 normalised quotes
python -m pytest -q
```

Every field carries a confidence score (LLM self-report × normalisation score ×
OCR line confidence for photos). Fields under 0.75 go to vendor read-back.

**About `data/cache/seed_*.json`:** these are seed outputs written in the exact
schema the LLM returns, so the demo runs with no API key. With `LLM_PROVIDER=gemini`
or `groq`, live outputs are validated and saved alongside them as
`data/cache/<step>_<file-hash>.json`, and those take priority.

## Tech stack

LangGraph · Gemini / Groq (swappable) · pdfplumber, PyMuPDF, Tesseract ·
pint, rapidfuzz · pandas, scikit-learn · FastAPI · SQLite · Streamlit
