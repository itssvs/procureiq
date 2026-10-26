"""Provider-swappable LLM layer: gemini | groq | cached.

The LLM only *proposes* structured data from messy text. It never decides
compliance. Every successful live call is written to data/cache/, and any
failure (no key, rate limit, network) falls back to that cache, so the demo
always runs.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from .config import CACHE_DIR, LLM_PROVIDER


class LLMUnavailable(RuntimeError):
    pass


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def load_cached(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_cached(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates ```json fences)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object in model output")
        text = text[start:end + 1]
    return json.loads(text)


# ------------------------------------------------------------------ providers
def _call_gemini(prompt: str, image_path: Optional[Path]) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMUnavailable("GEMINI_API_KEY not set")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    parts: list[Any] = [prompt]
    if image_path is not None:
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        parts.append(types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime))
    resp = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=parts,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
    )
    return resp.text


def _call_groq(prompt: str, image_path: Optional[Path]) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMUnavailable("GROQ_API_KEY not set")
    from groq import Groq

    # Groq text models can't see images; photos reach it as OCR text instead.
    client = Groq(api_key=key)
    resp = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


PROVIDERS = {"gemini": _call_gemini, "groq": _call_groq}


def extract_json(prompt: str, cache_key: str, image_path: Optional[Path] = None,
                 validator=None, retries: int = 2, seed_key: Optional[str] = None) -> tuple[dict, str]:
    """Run a structured-extraction prompt. Returns (data, source).

    source is the provider name, or "cache" when served from data/cache/.
    `seed_key` names a committed seed output (data/cache/seed_*.json) used when
    the exact file has never been seen live, e.g. a regenerated demo PDF.
    `validator` (e.g. a pydantic model's model_validate) is applied to every
    live output; invalid output is retried, then we fall back to cache.
    """
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()
    last_err: Optional[Exception] = None

    if provider in PROVIDERS:
        for _ in range(retries + 1):
            try:
                data = _parse_json(PROVIDERS[provider](prompt, image_path))
                if validator:
                    validator(data)
                save_cached(cache_key, data)
                return data, provider
            except LLMUnavailable as e:
                last_err = e
                break
            except Exception as e:  # bad JSON, schema error, rate limit, network
                last_err = e

    cached = load_cached(cache_key)
    if cached is not None:
        return cached, "cache"
    if seed_key and (seeded := load_cached(seed_key)) is not None:
        return seeded, "cache"
    raise RuntimeError(f"LLM provider '{provider}' failed ({last_err}) and no cache for '{cache_key}'")
