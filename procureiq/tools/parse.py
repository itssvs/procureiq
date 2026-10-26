"""Channel parsers: turn any input file into plain text plus a quality score.

pdf      -> pdfplumber (PyMuPDF fallback)
excel    -> pandas / openpyxl, flattened row by row
text     -> email or WhatsApp export, chat prefixes stripped
photo    -> Tesseract OCR with per-line confidence
voice    -> transcript file next to the audio (Whisper run offline, cached)
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDoc:
    text: str
    channel: str
    quality: float = 1.0                     # 0..1, overall extraction quality
    line_confidence: dict[str, float] = field(default_factory=dict)   # OCR only
    notes: list[str] = field(default_factory=list)


def parse_pdf(path: Path) -> ParsedDoc:
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)
    quality = 1.0 if len(text.strip()) > 50 else 0.3
    notes = [] if quality == 1.0 else ["PDF has little text; may be a scan"]
    return ParsedDoc(text=text, channel="pdf", quality=quality, notes=notes)


def parse_excel(path: Path) -> ParsedDoc:
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None, header=None)
    lines = []
    for name, df in sheets.items():
        lines.append(f"[sheet: {name}]")
        for row in df.fillna("").astype(str).values.tolist():
            cells = [c for c in row if c.strip()]
            if cells:
                lines.append(" | ".join(cells))
    return ParsedDoc(text="\n".join(lines), channel="excel")


_WA_PREFIX = re.compile(r"^\[\d{1,2}/\d{1,2}/\d{2,4},\s*[\d:]+\s*[AP]M\]\s*[^:]+:\s*", re.I)


def parse_text(path: Path, channel: str = "email") -> ParsedDoc:
    raw = path.read_text(encoding="utf-8")
    if channel == "whatsapp":
        lines = [_WA_PREFIX.sub("", ln).strip() for ln in raw.splitlines()]
        raw = "\n".join(ln for ln in lines if ln)
    return ParsedDoc(text=raw, channel=channel)


def parse_image(path: Path) -> ParsedDoc:
    """OCR a photo. Returns text and a confidence per line (0..1)."""
    try:
        import pytesseract
        from PIL import Image, ImageOps

        img = ImageOps.grayscale(Image.open(path))
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception as e:  # tesseract binary missing, etc.
        return ParsedDoc(text="", channel="photo", quality=0.0,
                         notes=[f"OCR unavailable ({type(e).__name__}); vision LLM or cache used"])

    lines: dict[tuple, list[tuple[str, float]]] = defaultdict(list)
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if word.strip() and conf >= 0:
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines[key].append((word, conf / 100))

    out_lines, line_conf = [], {}
    for words in lines.values():
        line = " ".join(w for w, _ in words)
        out_lines.append(line)
        line_conf[line] = round(sum(c for _, c in words) / len(words), 2)
    quality = round(sum(line_conf.values()) / len(line_conf), 2) if line_conf else 0.0
    return ParsedDoc(text="\n".join(out_lines), channel="photo", quality=quality,
                     line_confidence=line_conf)


def parse_voice(path: Path) -> ParsedDoc:
    """Voice notes are transcribed offline with Whisper; we read the transcript.

    Convention: vendor_x.ogg -> vendor_x.transcript.txt
    """
    tx = path.with_suffix(".transcript.txt")
    if tx.exists():
        return ParsedDoc(text=tx.read_text(encoding="utf-8"), channel="voice", quality=0.85,
                         notes=["Whisper transcript (cached)"])
    return ParsedDoc(text="", channel="voice", quality=0.0, notes=["no transcript found"])


def parse_any(path: Path, channel: str) -> ParsedDoc:
    path = Path(path)
    if channel == "pdf":
        return parse_pdf(path)
    if channel == "excel":
        return parse_excel(path)
    if channel == "photo":
        return parse_image(path)
    if channel == "voice":
        return parse_voice(path)
    return parse_text(path, channel)
