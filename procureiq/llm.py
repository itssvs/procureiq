"""Provider-swappable LLM layer: ollama (local, default) | gemini | groq | cached.

The LLM *proposes*: it turns messy text into structured data, plans which
tools to call, and writes vendor messages. It never decides compliance.

Every successful live call is saved to data/cache/ together with which model
produced it. Any failure (model not running, quota, bad JSON) falls back to
that cache, so the demo always runs. Sources are reported honestly:

  "ollama:qwen2.5:3b"            live call just now
  "recorded ollama:qwen2.5:3b"   replay of an earlier live call
  "seed"                         hand-written seed output (no model has run yet)
"""
from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .config import CACHE_DIR

META = "_meta"


class LLMUnavailable(RuntimeError):
    pass


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "cached").lower()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def load_cached(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_cached(key: str, data: dict, model: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {**data, META: {"model": model, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}}
    _cache_path(key).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def _source_of_cached(data: dict) -> str:
    meta = data.get(META)
    return f"recorded {meta['model']}" if meta else "seed"


def _parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates ```json fences and chatter)."""
    text = (text or "").strip()
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
def _image_b64(path: Path, max_side: int = 1024) -> str:
    """Downscale before sending: far fewer image tokens, much faster on a CPU."""
    import io

    from PIL import Image

    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _call_ollama(prompt: str, image_path: Optional[Path], system: Optional[str]) -> tuple[str, str]:
    """Local open-weight model via Ollama (https://ollama.com). No key, data stays on the machine."""
    import httpx

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:3b") if image_path else \
        os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    msg: dict[str, Any] = {"role": "user", "content": prompt}
    if image_path is not None:
        msg["images"] = [_image_b64(Path(image_path))]
    messages = ([{"role": "system", "content": system}] if system else []) + [msg]
    try:
        r = httpx.post(f"{host}/api/chat", timeout=float(os.getenv("OLLAMA_TIMEOUT", "600")),
                       json={"model": model, "messages": messages, "stream": False, "format": "json",
                             "options": {"temperature": 0, "num_ctx": 8192}})
    except httpx.ConnectError as e:
        raise LLMUnavailable(f"Ollama not running at {host} ({e})")
    if r.status_code == 404:
        raise LLMUnavailable(f"model {model} not pulled — run: ollama pull {model}")
    r.raise_for_status()
    return r.json()["message"]["content"], f"ollama:{model}"


def _call_gemini(prompt: str, image_path: Optional[Path], system: Optional[str]) -> tuple[str, str]:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMUnavailable("GEMINI_API_KEY not set")
    from google import genai
    from google.genai import types

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=key)
    parts: list[Any] = [prompt]
    if image_path is not None:
        mime = "image/png" if Path(image_path).suffix.lower() == ".png" else "image/jpeg"
        parts.append(types.Part.from_bytes(data=Path(image_path).read_bytes(), mime_type=mime))
    resp = client.models.generate_content(
        model=model, contents=parts,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0,
                                           system_instruction=system))
    return resp.text, f"gemini:{model}"


def _call_groq(prompt: str, image_path: Optional[Path], system: Optional[str]) -> tuple[str, str]:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMUnavailable("GROQ_API_KEY not set")
    from groq import Groq

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    resp = Groq(api_key=key).chat.completions.create(
        model=model, messages=messages, temperature=0, response_format={"type": "json_object"})
    return resp.choices[0].message.content, f"groq:{model}"


PROVIDERS: dict[str, Callable] = {"ollama": _call_ollama, "gemini": _call_gemini, "groq": _call_groq}


def extract_json(prompt: str, cache_key: str, image_path: Optional[Path] = None,
                 validator: Optional[Callable[[dict], dict]] = None, retries: int = 1,
                 seed_key: Optional[str] = None, system: Optional[str] = None) -> tuple[dict, str]:
    """Run a JSON-producing prompt. Returns (data, source).

    `validator` receives the parsed JSON and returns a cleaned version (or
    raises). Invalid output is retried, then we fall back to cache, then to the
    hand-written seed named by `seed_key`.
    """
    prov = provider()
    errors: list[str] = []

    # PREFER_RECORDED=1: reuse an earlier real model output for this exact input when one
    # exists, and only call the model for steps never recorded (saves minutes on a CPU).
    if os.getenv("PREFER_RECORDED") == "1" and prov in PROVIDERS:
        rec = load_cached(cache_key)
        if rec is not None and rec.get(META):
            meta = rec.pop(META)
            return rec, f"recorded {meta['model']}"

    if prov in PROVIDERS:
        for attempt in range(retries + 1):
            try:
                text, model = PROVIDERS[prov](prompt, image_path, system)
                data = _parse_json(text)
                if validator:
                    data = validator(data)
                save_cached(cache_key, data, model)
                return data, model
            except LLMUnavailable as e:
                errors.append(str(e))
                break
            except Exception as e:  # bad JSON, schema error, quota, network
                errors.append(f"attempt {attempt + 1}: {type(e).__name__}: {e}"[:300])

    for key in filter(None, (cache_key, seed_key)):
        cached = load_cached(key)
        if cached is not None:
            src = _source_of_cached(cached)
            cached.pop(META, None)
            if errors:
                src += " (live failed)"
            return cached, src
    raise RuntimeError(f"LLM '{prov}' failed ({'; '.join(errors) or 'no live provider'}) "
                       f"and nothing cached for '{cache_key}'")


def last_error_hint() -> str:
    p = provider()
    return {"ollama": "Is Ollama running? Try: ollama run qwen2.5:3b",
            "gemini": "Check GEMINI_API_KEY in .env", "groq": "Check GROQ_API_KEY in .env"}.get(p, "")
