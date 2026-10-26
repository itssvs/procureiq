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
| 3 | Spec-claim verifier, rule engine, ranking | ✅ done |
| 4 | LangGraph agent loop, FastAPI | ✅ done |
| 5 | Streamlit approval dashboard | ✅ done |
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

## Where the AI is

ProcureIQ runs a **local open-weight model** (Qwen2.5 3B through [Ollama](https://ollama.com)) by default. It's free, needs no API key, and procurement data never leaves the machine. Gemini or Groq can be switched in with one line in `.env`.

| The model does (proposes) | Deterministic code does (decides) |
|---|---|
| Reads the buyer email into 12 tagged requirements | Unit/synonym normalisation (pint, rapidfuzz) |
| Reads each quote (PDF, WhatsApp, **handwritten photo via Qwen2.5-VL**) into claims with confidence | Every compliance verdict, with cited evidence |
| **Plans step 8 as a tool-using agent:** chooses which tools to call (`check_licence`, `lookup_datasheet`, `inspect_field`, `send_clarification`, `escalate`, `finish`) and writes each vendor message in that vendor's tone | Guardrails: may only contact CONDITIONAL vendors, a checklist of every open item is appended by code, skipped vendors are handled by rule, and repeated calls and the step count are capped |
| Reads the vendor's free-text reply into field updates | Re-verification, ranking, the human approval gate |

Every model output is schema-checked and repaired (small models drift), with a regex safety net for licence numbers and test values. Each output is saved with the model that produced it. The dashboard and CLI label every AI step honestly:

- `ollama:qwen2.5:3b`: live
- `recorded ollama:qwen2.5:3b`: replay of a real earlier run
- `seed`: the hand-written placeholder used before any model has run

### Setup for the local model

```bash
# install Ollama from https://ollama.com/download, then:
ollama pull qwen2.5:3b        # text + agent planning (~2 GB)
ollama pull qwen2.5vl:3b      # reads the handwritten photo (~3 GB)
cp .env.example .env          # LLM_PROVIDER=ollama
python -m procureiq.agent     # full run; prints which model produced each step
```

In the dashboard, the **Live model** toggle runs the model now. On a laptop CPU this takes minutes. With the toggle off, the dashboard replays the last recorded run instantly, which is what the hosted demo does.

## Run the ingestion step (Part 2)

```bash
python -m procureiq.ingest     # prints 12 extracted specs + 3 normalised quotes
python -m pytest -q
```

Every field carries a confidence score (LLM self-report × normalisation score ×
OCR line confidence for photos). Fields under 0.75 go to vendor read-back.

**About `data/cache/`:** `seed_*.json` are hand-written placeholders in the exact schema the model returns, so the project runs even before any model is installed. Live outputs are saved next to them as `<step>_<hash>.json`, tagged with the model that produced them, and take priority over the seeds.

## Run the verifier (Part 3)

```bash
python -m procureiq.evaluate   # every claim labelled, with cited evidence, then ranked
```

| Vendor | Result |
|---|---|
| A | All 9 spec claims VERIFIED → **COMPLIANT, rank 1** |
| B | Conductor **SUBSTITUTED** (quote wording, 12.1 Ω/km resistance and brand datasheet all say aluminium); ISI licence missing; price 12% below market band → **EXCLUDED** |
| C | ISI licence read at low confidence (< 0.75) → not judged, read-back; FR claim has no test values → **CONDITIONAL** |

The status of each quote depends only on its mandatory requirements:

- **EXCLUDED** if any mandatory requirement fails (contradicted or substituted).
- **CONDITIONAL** if no mandatory requirement fails but at least one is unverified. The open items list what proof would resolve it.
- **COMPLIANT** if every mandatory requirement passes. Only compliant quotes get ranked.

The ranking score is 0.6 × price + 0.2 × delivery + 0.2 × warranty, with a penalty for each unmet preferred requirement. A Pareto flag marks quotes that are non-dominated on price, delivery and warranty.

## Run the full agent (Part 4)

```bash
python -m procureiq.agent                  # whole loop + audit trail in the terminal
uvicorn procureiq.api:app --reload         # API at http://127.0.0.1:8000/docs
```

The LangGraph state machine (`procureiq/agent/graph.py`) runs these nodes in order:

1. `extract_requirements`
2. `ingest_quotes`
3. `verify` (runs claim checks, rules, anomaly checks and ranking)
4. `decide`, which routes to one of three places:
   - `clarify`, which loops back to `verify`. There are at most `MAX_CLARIFY_ROUNDS` rounds.
   - `escalate` to a human, when vendors are silent or the round cap is reached.
   - `route_for_approval`

In the demo, Vendor C gets a read-back of its misread licence, total and GST, plus a request for its FR test report. Its reply is parsed and merged into the quote, and the re-verify step turns C into **COMPLIANT, rank 2**.

The run then stops at `AWAITING_APPROVAL`. Only a human `POST /runs/{id}/approve` can award, and only to a COMPLIANT quote. Every step, vendor message and decision is written to SQLite (`data/runs/procureiq.db`).

| Endpoint | Purpose |
|---|---|
| `POST /runs/demo` | Run the built-in scenario |
| `POST /runs` | Run on uploaded files (needs a Gemini/Groq key for unseen files) |
| `GET /runs/{id}` · `/memo` · `/audit` | State, approval memo, audit trail |
| `POST /runs/{id}/approve` · `/reject` | Human decision |

## Approval dashboard (Part 5)

```bash
streamlit run app.py          # opens http://localhost:8501
```

Press **▶ Run agent** in the sidebar. The dashboard has six tabs:

- **Decision:** the shortlist, a price vs delivery chart, and the human approval form. Only COMPLIANT quotes can be selected. It also shows why each excluded quote was excluded, with evidence, plus the anomalies.
- **Claim verification:** a requirement × vendor matrix with the four labels, and the full evidence trail for every claim.
- **Clarifications:** what the agent sent, what the vendor replied, the fields that were updated, and each vendor's status before and after.
- **Quotes:** the original input (PDF, chat or photo) next to the extracted fields and their confidence bars.
- **Requirements:** the buyer email next to the 12 extracted specs.
- **Audit log:** every step, vendor message and human decision.

Turn off **"Vendor C answers clarification"** in the sidebar to see the agent escalate an unresolved quote to a human instead.

## Tech stack

LangGraph · Qwen2.5 3B / Qwen2.5-VL 3B via Ollama (local) · Gemini / Groq (optional) · pdfplumber, PyMuPDF, Tesseract ·
pint, rapidfuzz · pandas · FastAPI · SQLite · Streamlit
