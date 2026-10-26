"""FastAPI backend.

    uvicorn procureiq.api:app --reload      → http://127.0.0.1:8000/docs

Endpoints
  GET  /health
  POST /runs/demo                 run the agent on the built-in scenario
  POST /runs                      run on uploaded files (request + quotes)
  GET  /runs                      recent runs
  GET  /runs/{run_id}             full state: requirements, quotes, evaluation, memo
  GET  /runs/{run_id}/memo        approval memo only
  GET  /runs/{run_id}/audit       ordered audit trail
  POST /runs/{run_id}/approve     human award (COMPLIANT quotes only)
  POST /runs/{run_id}/reject      human rejects all quotes
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import __version__, audit
from .agent import graph
from .config import DATA_DIR
from .llm import provider

app = FastAPI(title="ProcureIQ", version=__version__,
              description="Autonomous requirement-to-quotation procurement agent. "
                          "Verifies every vendor claim before price is considered; never auto-awards.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = DATA_DIR / "runs" / "uploads"
CHANNEL_BY_EXT = {".pdf": "pdf", ".xlsx": "excel", ".xls": "excel", ".png": "photo", ".jpg": "photo",
                  ".jpeg": "photo", ".txt": "email", ".eml": "email"}


class Decision(BaseModel):
    vendor_id: str
    approver: str
    note: str = ""


class Rejection(BaseModel):
    approver: str
    note: str


def _get(run_id: str) -> dict:
    row = audit.get_run(run_id)
    if row is None:
        raise HTTPException(404, f"run {run_id} not found")
    return row


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "llm_provider": provider()}


@app.post("/runs/demo")
def run_demo(max_rounds: int = 3):
    return graph.summary(graph.run(max_rounds=max_rounds))


@app.post("/runs")
async def run_uploaded(request: UploadFile = File(..., description="Buyer request (.txt / .eml)"),
                       quotes: list[UploadFile] = File(..., description="Vendor quotes (pdf, xlsx, png/jpg, txt)"),
                       vendor_names: Optional[str] = Form(None, description="Comma-separated, same order as quotes"),
                       whatsapp: Optional[str] = Form(None, description="Comma-separated indexes of .txt quotes "
                                                                         "that are WhatsApp exports")):
    """Run the agent on your own files. Without an LLM key, unseen files cannot be read
    (cached mode only knows the demo files) — set LLM_PROVIDER=gemini or groq."""
    run_dir = UPLOAD_DIR / audit.new_run().replace("RUN-", "UP-")
    run_dir.mkdir(parents=True, exist_ok=True)

    def save(u: UploadFile) -> Path:
        p = run_dir / Path(u.filename).name
        with p.open("wb") as fh:
            shutil.copyfileobj(u.file, fh)
        return p

    names = [n.strip() for n in vendor_names.split(",")] if vendor_names else []
    wa = {int(i) for i in whatsapp.split(",")} if whatsapp else set()
    inputs = {"request": str(save(request)), "quotes": {}}
    for i, u in enumerate(quotes):
        p = save(u)
        channel = "whatsapp" if i in wa else CHANNEL_BY_EXT.get(p.suffix.lower())
        if channel is None:
            raise HTTPException(400, f"unsupported file type: {p.name}")
        vid = chr(ord("A") + i)
        inputs["quotes"][vid] = {"vendor": names[i] if i < len(names) else "", "channel": channel, "file": str(p)}
    try:
        state = graph.run(inputs=inputs, sim=graph.VendorSimulator(replies={}))
    except RuntimeError as e:          # no LLM and no cache for these files
        raise HTTPException(422, str(e))
    return graph.summary(state)


@app.get("/runs")
def list_runs(limit: int = 20):
    return audit.list_runs(limit)


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    return _get(run_id)


@app.get("/runs/{run_id}/memo")
def get_memo(run_id: str):
    memo = _get(run_id)["state"].get("memo")
    if not memo:
        raise HTTPException(409, "run has not produced a memo yet")
    return memo


@app.get("/runs/{run_id}/audit")
def get_audit(run_id: str):
    _get(run_id)
    return audit.events(run_id)


@app.post("/runs/{run_id}/approve")
def approve(run_id: str, d: Decision):
    try:
        return graph.approve(run_id, d.vendor_id, d.approver, d.note)
    except graph.ApprovalError as e:
        raise HTTPException(409, str(e))


@app.post("/runs/{run_id}/reject")
def reject(run_id: str, r: Rejection):
    try:
        graph.reject_all(run_id, r.approver, r.note)
    except graph.ApprovalError as e:
        raise HTTPException(404, str(e))
    return {"status": "REJECTED"}
