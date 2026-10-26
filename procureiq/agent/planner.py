"""Tool-using LLM agent for step 8 (investigate → clarify → escalate).

The model plans: it looks at the open issues, chooses which tools to call,
and writes each vendor message in that vendor's channel and tone. The tools
are deterministic and own every verdict — the model can't mark anything
compliant. Code guardrails then guarantee:

  * only CONDITIONAL vendors can be contacted (EXCLUDED ones failed a mandatory spec)
  * every open item is in the message (a checklist is appended by code)
  * no vendor is silently skipped (anything the model misses is handled by rule)
  * the loop is bounded (max steps, repeated calls are refused)

Each step is a JSON object {"thought", "tool", "args"} — plain JSON mode, so it
works with a 3B local model, and each step is cached so a recorded run replays
exactly.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Optional

from .. import audit, kb
from ..llm import extract_json
from ..schemas import Quote, RequirementSet
from ..tools.extract import extract_reply
from ..tools.normalise import apply_updates
from ..tools.rules import Evaluation, QuoteEvaluation
from .vendor_sim import LABELS, VendorSimulator, draft_clarification

MAX_STEPS = 8

SYSTEM = """You are ProcureIQ, a procurement agent working for the facilities team of a factory in Chennai.
Compliance verdicts come only from deterministic tools. You never decide or change a verdict.
Your job now: resolve the open issues on CONDITIONAL quotes. Investigate with tools when useful,
then ask each such vendor precisely for what is missing, or escalate to a human.
Never contact an EXCLUDED vendor: it failed a mandatory requirement and clarification cannot fix that.
Write vendor messages that suit the channel: WhatsApp = short, warm, Indian business English;
email or handwritten letter = polite and formal. Quote the exact value you read when asking for a read-back.
Reply with exactly one JSON object: {"thought": "<one sentence>", "tool": "<tool name>", "args": {...}}"""

PROMPT = """TOOLS
{tools}

SITUATION (round {round})
{situation}

STEPS SO FAR
{history}

Choose the single next tool call. When every CONDITIONAL vendor has been messaged or escalated, call finish."""


class Planner:
    def __init__(self, run_id: str, round_no: int, reqs: RequirementSet, quotes: dict[str, Quote],
                 ev: Evaluation, sim: VendorSimulator, silent: list[str]):
        self.run_id, self.round = run_id, round_no
        self.reqs, self.quotes, self.ev, self.sim = reqs, quotes, ev, sim
        self.silent = silent
        self.trace: list[dict] = []
        self.handled: dict[str, str] = {}          # vendor_id -> "messaged" | "escalated"
        self.history_entries: list[dict] = []      # clarification history for the dashboard
        self.changed_any = False
        self.tools: dict[str, tuple[Callable, str]] = {
            "inspect_field": (self.t_inspect_field, 'inspect_field(vendor_id, field): what the vendor wrote and our confidence'),
            "check_licence": (self.t_check_licence, "check_licence(licence_no): look up the BIS licence registry"),
            "lookup_datasheet": (self.t_lookup_datasheet, "lookup_datasheet(brand): facts on file for a brand"),
            "send_clarification": (self.t_send, "send_clarification(vendor_id, message): message a CONDITIONAL vendor "
                                                "(a checklist of open items is appended automatically)"),
            "escalate": (self.t_escalate, "escalate(vendor_id, reason): hand a vendor to the human approver"),
            "finish": (self.t_finish, "finish(summary): end this round"),
        }

    # ------------------------------------------------------------ situation
    def _pending(self) -> list[QuoteEvaluation]:
        return [e for e in self.ev.quotes if e.status == "CONDITIONAL"
                and e.vendor_id not in self.silent and e.vendor_id not in self.handled]

    def situation(self) -> str:
        lines = []
        for e in self.ev.quotes:
            q = self.quotes[e.vendor_id]
            line = f"- {e.vendor_id} {e.vendor} [{e.status}] channel={q.channel} brand={q.brand or '?'}"
            if e.status == "CONDITIONAL":
                line += "\n    open: " + "; ".join(e.open_items)
                flags = [a.message for a in e.anomalies if a.severity != "info"]
                if flags:
                    line += "\n    anomalies: " + "; ".join(flags)
                sample = next((f.raw for f in q.fields.values() if len(f.raw) > 15), "")
                if sample:
                    line += f'\n    vendor writes like: "{sample[:80]}"'
            elif e.status == "EXCLUDED":
                line += f"  (do not contact: {e.failed[0][:90]})"
            if e.vendor_id in self.silent:
                line += "  (did not reply earlier)"
            if e.vendor_id in self.handled:
                line += f"  (already {self.handled[e.vendor_id]} this round)"
            lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------ tools
    def t_inspect_field(self, vendor_id: str = "", field: str = "", **_) -> Any:
        q = self.quotes.get(vendor_id)
        if q is None:
            return f"error: unknown vendor {vendor_id}"
        f = q.fields.get(field)
        if f is None:
            return f"{vendor_id} did not state {field}"
        return {"value": f.value, "vendor_wrote": f.raw, "confidence": f.confidence}

    def t_check_licence(self, licence_no: str = "", **_) -> Any:
        e = kb.licence(str(licence_no).strip())
        if e is None:
            return f"{licence_no}: NOT in registry (may be a misread — confirm with vendor before concluding)"
        return {k: e[k] for k in ("licence_no", "holder", "brand", "standard", "status", "valid_until")}

    def t_lookup_datasheet(self, brand: str = "", **_) -> Any:
        ds = kb.datasheet_for(brand)
        if ds is None:
            return f"no datasheet on file for {brand}"
        return {k: v for k, v in ds.items() if v is not None and k != "id"} | {"evidence_id": ds["id"]}

    def t_send(self, vendor_id: str = "", message: str = "", **_) -> Any:
        e = next((x for x in self.ev.quotes if x.vendor_id == vendor_id), None)
        if e is None:
            return f"error: unknown vendor {vendor_id}"
        if e.status != "CONDITIONAL":
            return f"refused: {vendor_id} is {e.status}; only CONDITIONAL vendors can be asked for clarification"
        if vendor_id in self.handled:
            return f"refused: {vendor_id} already {self.handled[vendor_id]} this round"
        if len(str(message).strip()) < 20:
            return "refused: message too short; write the request to the vendor"
        return self._deliver(e, str(message).strip(), author="agent")

    def t_escalate(self, vendor_id: str = "", reason: str = "", **_) -> Any:
        if vendor_id not in self.quotes:
            return f"error: unknown vendor {vendor_id}"
        self.handled[vendor_id] = "escalated"
        audit.log(self.run_id, "8_escalate", f"Agent escalated {vendor_id}: {reason}")
        return f"{vendor_id} handed to the human approver"

    def t_finish(self, summary: str = "", **_) -> Any:
        return "round finished"

    # ------------------------------------------------------------ delivery
    def _checklist(self, e: QuoteEvaluation) -> str:
        items = []
        for it in e.open_items:
            if it.startswith("Read-back:"):
                key, _, val = it.split("confirm", 1)[1].strip().partition("=")
                items.append(f"[ ] {LABELS.get(key.strip(), key.strip())}: we read \"{val.strip()}\" — correct?")
            else:
                items.append(f"[ ] {it.removeprefix('Request ')}")
        return "\n".join(items)

    def _deliver(self, e: QuoteEvaluation, body: str, author: str) -> str:
        q = self.quotes[e.vendor_id]
        full = f"{body}\n\nItems to confirm:\n{self._checklist(e)}" if author == "agent" else body
        self.handled[e.vendor_id] = "messaged"
        audit.log(self.run_id, "8_clarify", f"Round {self.round}: message to {q.vendor} "
                  f"({'written by agent' if author == 'agent' else 'rule template'})", data={"message": full})
        entry = {"round": self.round, "vendor_id": q.vendor_id, "vendor": q.vendor, "sent": full,
                 "author": author, "reply": None, "changed": []}
        reply_path = self.sim.send(q.vendor_id, full)
        if reply_path is None:
            self.silent.append(q.vendor_id)
            audit.log(self.run_id, "8_clarify", f"{q.vendor} did not reply", actor=f"vendor:{q.vendor_id}")
            self.history_entries.append(entry)
            return "message delivered; no reply from vendor"
        data, src = extract_reply(q.vendor_id, reply_path)
        changed = apply_updates(q, data, source=f"clarification_r{self.round}")
        self.changed_any |= bool(changed)
        entry["reply"] = reply_path.read_text(encoding="utf-8")
        entry["changed"] = changed
        entry["reply_source"] = src
        self.history_entries.append(entry)
        audit.log(self.run_id, "8_clarify", f"Reply received; updated {', '.join(changed) or 'nothing'}",
                  actor=f"vendor:{q.vendor_id}", data={k: q.fields[k].value for k in changed})
        return {"reply_received": True, "fields_updated": {k: q.fields[k].value for k in changed},
                "note": "verdicts will be re-checked by the verifier after this round"}

    # ------------------------------------------------------------ loop
    def _check_step(self, data: dict) -> dict:
        tool = data.get("tool") or data.get("action") or data.get("name")
        args = data.get("args") or data.get("arguments") or data.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if tool not in self.tools:
            raise ValueError(f"unknown tool {tool!r}")
        return {"thought": str(data.get("thought", ""))[:400], "tool": tool, "args": args if isinstance(args, dict) else {}}

    def _history_text(self) -> str:
        if not self.trace:
            return "(none yet)"
        out = []
        for i, t in enumerate(self.trace[-6:], start=max(1, len(self.trace) - 5)):
            obs = t["observation"] if isinstance(t["observation"], str) else json.dumps(t["observation"], default=str)
            out.append(f"{i}. {t['tool']}({json.dumps(t['args'], ensure_ascii=False)[:300]}) -> {obs[:300]}")
        return "\n".join(out)

    def run(self) -> bool:
        """Run the agent loop. Returns False if no model (live or recorded) was available."""
        tools_txt = "\n".join(f"- {desc}" for _, desc in self.tools.values())
        seen_calls: set[str] = set()
        for _ in range(MAX_STEPS):
            prompt = PROMPT.format(tools=tools_txt, round=self.round, situation=self.situation(),
                                   history=self._history_text())
            key = "agent_" + hashlib.sha1((SYSTEM + prompt).encode()).hexdigest()[:16]
            try:
                step, src = extract_json(prompt, cache_key=key, system=SYSTEM, validator=self._check_step)
            except RuntimeError:
                if not self.trace:
                    return False                      # nothing available: caller uses rules
                break
            step = self._check_step(step)
            sig = step["tool"] + json.dumps(step["args"], sort_keys=True)
            if sig in seen_calls and step["tool"] != "finish":
                obs: Any = "refused: you already made this exact call. Message the vendor, escalate, or finish."
            else:
                seen_calls.add(sig)
                fn = self.tools[step["tool"]][0]
                try:
                    obs = fn(**step["args"])
                except TypeError as err:
                    obs = f"error: bad arguments ({err})"
            self.trace.append({**step, "observation": obs, "source": src, "round": self.round})
            audit.log(self.run_id, "8_agent", f"{step['tool']}({', '.join(f'{k}={v}' for k, v in step['args'].items())[:160]})"
                      + (f" — {step['thought']}" if step["thought"] else ""), data={"observation": obs, "model": src})
            if step["tool"] == "finish":
                break
        self.guardrail()
        return True

    def guardrail(self) -> None:
        """Anything the model left unhandled is handled by rule, and logged as such."""
        for e in self._pending():
            audit.log(self.run_id, "8_guardrail", f"Agent did not handle {e.vendor_id}; sending rule-based request")
            self.trace.append({"thought": "guardrail: vendor left unhandled by the agent", "tool": "send_clarification",
                               "args": {"vendor_id": e.vendor_id}, "observation": "rule template sent",
                               "source": "rules", "round": self.round})
            self._deliver(e, draft_clarification(self.quotes[e.vendor_id], e, self.round), author="rules")


def rules_only_round(run_id: str, round_no: int, quotes: dict[str, Quote], ev: Evaluation,
                     sim: VendorSimulator, silent: list[str]) -> Planner:
    """Deterministic fallback when no model is available: template message to every CONDITIONAL vendor."""
    p = Planner(run_id, round_no, None, quotes, ev, sim, silent)   # type: ignore[arg-type]
    audit.log(run_id, "8_agent", "No model available (live or recorded); using rule-based clarification")
    p.trace.append({"thought": "no LLM available — rule-based fallback", "tool": "(rules)", "args": {},
                    "observation": "", "source": "rules", "round": round_no})
    p.guardrail()
    return p
