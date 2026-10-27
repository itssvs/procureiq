"""ProcureIQ approval dashboard (Streamlit).

    streamlit run app.py
"""
from __future__ import annotations

import html
import os
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from procureiq import audit
from procureiq.agent import graph
from procureiq.agent.vendor_sim import VendorSimulator
from procureiq.config import DEMO_QUOTES, DEMO_REQUEST

st.set_page_config(page_title="ProcureIQ", page_icon="🔎", layout="wide")

# ------------------------------------------------------------------ styling
LABEL_STYLE = {
    "VERIFIED": ("#0f7b4f", "#e3f5ec", "✓"),
    "UNVERIFIABLE": ("#9a6700", "#fff4d6", "?"),
    "CONTRADICTED": ("#b42318", "#fde7e5", "✕"),
    "SUBSTITUTED": ("#7a2e98", "#f3e6fa", "⇄"),
}
STATUS_STYLE = {"COMPLIANT": ("#0f7b4f", "#e3f5ec"), "CONDITIONAL": ("#9a6700", "#fff4d6"),
                "EXCLUDED": ("#b42318", "#fde7e5")}
SPEC_NAMES = {"conductor_material": "Conductor material", "size_sqmm": "Size (sq mm)", "cores": "Cores",
              "voltage_grade_kv": "Voltage grade", "insulation": "Insulation", "standard": "Standard",
              "isi_licence": "ISI licence", "fire_grade": "Fire grade (FR)", "conductor_class": "Conductor class",
              "quantity_m": "Quantity (m)", "delivery_days": "Delivery (days)", "warranty_years": "Warranty (yrs)",
              "conductor_detail": "Conductor detail", "oxygen_index": "Oxygen index", "temperature_index_c":
              "Temperature index (°C)", "fr_test_report": "FR test report", "resistance_ohm_per_km":
              "Resistance (Ω/km)", "coil_length_m": "Coil length (m)", "quantity_coils": "Coils",
              "unit_price_inr": "Rate per coil (₹)", "total_price_inr": "Total (₹, pre-GST)", "gst_pct": "GST (%)"}


def nice_failure(text: str) -> str:
    key, _, rest = text.partition(": ")
    return f"{SPEC_NAMES.get(key, key)} — {rest}" if rest else text


def fmt_val(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.0f}" if v >= 1000 else f"{v:g}"
    return str(v)

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1280px;}
.piq-badge {display:inline-block; padding:2px 9px; border-radius:999px; font-size:0.78rem;
            font-weight:600; letter-spacing:.02em; white-space:nowrap;}
.piq-matrix {width:100%; border-collapse:collapse; font-size:0.9rem;}
.piq-matrix th, .piq-matrix td {padding:8px 10px; border-bottom:1px solid rgba(128,128,128,.18); text-align:left;}
.piq-matrix th {font-weight:600; opacity:.75; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em;}
.piq-matrix td.mand::after {content:" ●"; color:#b42318; font-size:.6rem; vertical-align:middle;}
.piq-banner {padding:14px 18px; border-radius:10px; border:1px solid rgba(128,128,128,.25); margin:.4rem 0 1rem;}
.piq-sub {opacity:.7; font-size:.85rem;}
.piq-ev {font-size:.82rem; opacity:.85; margin:2px 0 2px 14px;}
.piq-ev code {font-size:.76rem;}
</style>""", unsafe_allow_html=True)


def badge(label: str) -> str:
    fg, bg, icon = LABEL_STYLE.get(label, ("#444", "#eee", ""))
    return f'<span class="piq-badge" style="color:{fg};background:{bg}">{icon} {label}</span>'


def status_badge(status: str, rank=None) -> str:
    fg, bg = STATUS_STYLE.get(status, ("#444", "#eee"))
    txt = f"{status} · RANK {rank}" if rank else status
    return f'<span class="piq-badge" style="color:{fg};background:{bg}">{txt}</span>'


def inr(v) -> str:
    if v is None:
        return "—"
    v = float(v)
    return f"₹{v / 100000:.2f} L" if v >= 100000 else f"₹{v:,.0f}"


def esc(x) -> str:
    return html.escape(str(x))


def source_pill(src: str) -> str:
    """Honest label for where an AI output came from."""
    src = src or ""
    if src.startswith("recorded"):
        fg, bg, txt = "#1f3a5f", "#e6eef8", f"🧠 {src.removeprefix('recorded ')} · recorded run"
    elif src == "seed" or src.startswith("seed"):
        fg, bg, txt = "#6b6b6b", "#eeeeee", "seed output (no model run yet)"
    elif src == "rules":
        fg, bg, txt = "#6b6b6b", "#eeeeee", "⚙ rule fallback"
    elif src:
        fg, bg, txt = "#0f7b4f", "#e3f5ec", f"🧠 {src} · live"
    else:
        return ""
    if "(live failed)" in src:
        txt = txt.replace(" (live failed)", "") + " · live call failed"
    return f'<span class="piq-badge" style="color:{fg};background:{bg}">{esc(txt)}</span>'


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🔎 ProcureIQ")
    st.caption("Not the cheapest quote. The cheapest quote that actually meets the spec.")
    st.divider()
    st.markdown("**Demo scenario**")
    st.caption("Facilities email + 3 vendor quotes: PDF, WhatsApp text and a handwritten photo.")
    vendor_c_replies = st.toggle("Vendor C answers clarification", value=True,
                                 help="Turn off to see the agent escalate an unresolved quote to a human.")
    max_rounds = st.slider("Max clarification rounds", 0, 3, 3)
    configured = os.getenv("LLM_PROVIDER", "cached")
    live = False
    if configured != "cached":
        live = st.toggle(f"Live model ({configured})", value=False,
                         help="On: the model reads every document and plans the agent's actions now "
                              "(minutes on a laptop CPU). Off: replay the last recorded model run instantly.")
    run_clicked = st.button("▶  Run agent", type="primary", width="stretch")

    runs = audit.list_runs(15)
    if runs:
        st.divider()
        with st.expander(f"Past runs ({len(runs)})"):
            ids = [r["run_id"] for r in runs]
            pick = st.selectbox("Open a past run", ids, index=None, placeholder="Choose a run",
                                format_func=lambda r: f"{r[4:]} · {next(x['status'] for x in runs if x['run_id'] == r)}")
            if pick and pick != st.session_state.get("run_id"):
                st.session_state["run_id"] = pick
                st.rerun()
    st.divider()
    prov = configured
    if prov == "ollama":
        try:
            import httpx

            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            tags = [m["name"] for m in httpx.get(f"{host}/api/tags", timeout=1.5).json().get("models", [])]
            want = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
            st.caption(f"🧠 Local model: **{want}** " + ("✅ ready" if any(t.startswith(want) for t in tags)
                                                      else f"⚠ not pulled — `ollama pull {want}`"))
        except Exception:
            st.caption("🧠 Local model: ⚠ Ollama not running — using recorded outputs")
    else:
        st.caption(f"🧠 LLM provider: **{prov}**" + (" (replaying recorded model outputs)" if prov == "cached" else ""))
    st.caption("The model reads and plans; deterministic rules give every verdict. The agent never auto-awards.")

STEP_NAMES = {"extract_requirements": "Reading the buyer's email", "ingest_quotes": "Reading the 3 vendor quotes",
              "verify": "Verifying every claim against the knowledge base", "clarify": "Agent handling open issues",
              "escalate": "Escalating unresolved vendors", "route_for_approval": "Writing the approval memo"}


def run_with_progress(sim, max_rounds, use_live):
    """Run the agent and stream each step's audit events into a status box."""
    os.environ["LLM_PROVIDER"] = configured if use_live else "cached"
    shown = {"n": 0}
    box = st.status("Agent running…", expanded=True)

    def on_step(node, rid):
        evs = audit.events(rid)
        box.markdown(f"**{STEP_NAMES.get(node, node)}**")
        for e in evs[shown["n"]:]:
            if e["step"] == "0_start":
                continue
            who = "🏭 " if e["actor"].startswith("vendor") else ("👤 " if e["actor"].startswith("human") else "")
            box.markdown(f"<div class='piq-ev'>{who}<code>{esc(e['step'])}</code> {esc(e['message'][:180])}</div>",
                         unsafe_allow_html=True)
        shown["n"] = len(evs)
        time.sleep(0.35)                     # pacing so a person can follow; the work itself is done

    try:
        state = graph.run(sim=sim, max_rounds=max_rounds, on_step=on_step)
    finally:
        os.environ["LLM_PROVIDER"] = configured
    box.update(label=f"Done: {state['memo']['summary'][:90]}…", state="complete", expanded=False)
    return state


if run_clicked:
    st.session_state["run_id"] = run_with_progress(
        VendorSimulator() if vendor_c_replies else VendorSimulator(replies={}), max_rounds, live)["run_id"]
    st.rerun()

run_id = st.session_state.get("run_id")
REPLAY = configured == "cached"
if not run_id:
    st.title("ProcureIQ")
    st.markdown("#### Not the cheapest quote. The cheapest quote that actually meets the spec.")
    st.markdown(
        "An AI procurement agent that reads a buyer's request and vendor quotes from messy channels, "
        "checks **every vendor claim against evidence before price is considered**, chases missing proof "
        "from vendors, and hands a human an evidence-backed shortlist. Every claim gets one label:")
    st.markdown(" ".join(badge(lbl) for lbl in LABEL_STYLE), unsafe_allow_html=True)
    st.markdown("")
    c = st.columns(4)
    for col, (n, head, body) in zip(c, [
        ("1", "Read", "A local model (Qwen2.5 3B) reads the email; a vision model (Qwen2.5-VL) reads the handwritten quote."),
        ("2", "Verify", "Deterministic code checks each claim against standards, datasheets and the BIS registry."),
        ("3", "Clarify", "A tool-using agent decides whom to ask for what, writes the message, and reads the reply."),
        ("4", "Approve", "Only compliant quotes are ranked. A human approves; the agent never awards.")]):
        col.markdown(f'<div class="piq-banner"><b>{n} · {head}</b><br><span class="piq-sub">{body}</span></div>',
                     unsafe_allow_html=True)
    st.info("**Press ▶ Run agent** in the sidebar to process the inputs below. Then open **Try it yourself** "
            "to change a vendor's claim and watch the verdicts change.")
    if REPLAY:
        st.caption("ℹ️ This free host has no GPU, so the **model's text outputs are replayed** from a real run of "
                   "Qwen2.5 on a laptop (each is labelled *recorded run*). Everything else (verification, rules, "
                   "anomaly checks, the agent loop, guardrails, ranking, approval) **executes on every run**. "
                   "To run the model live, see the README (Ollama, ~8 min on CPU).")

    st.markdown("##### The inputs")
    t = st.tabs(["Buyer email", "A · PDF quote", "B · WhatsApp chat", "C · Handwritten photo"])
    t[0].code(DEMO_REQUEST.read_text(encoding="utf-8"), language=None, wrap_lines=True)
    try:
        import pymupdf as fitz

        with fitz.open(DEMO_QUOTES["A"]["file"]) as doc:
            t[1].image(doc[0].get_pixmap(dpi=100).tobytes("png"), width=700)
    except Exception:
        t[1].caption("(PDF preview unavailable)")
    t[2].code(DEMO_QUOTES["B"]["file"].read_text(encoding="utf-8"), language=None, wrap_lines=True)
    t[3].image(str(DEMO_QUOTES["C"]["file"]), width=520)
    st.stop()

row = audit.get_run(run_id)
state = row["state"]
if not state.get("memo"):
    st.warning("This run did not finish. Start a new run from the sidebar.")
    st.stop()

reqs = state["requirements"]
ev = state["evaluation"]
init_ev = state.get("initial_evaluation", ev)
memo = state["memo"]
quotes = {q["vendor_id"]: q for q in state["quotes"]}
evals = {e["vendor_id"]: e for e in ev["quotes"]}
init_status = {e["vendor_id"]: e["status"] for e in init_ev["quotes"]}
status = row["status"]

# ------------------------------------------------------------------ header
st.markdown(f"### {esc(reqs['item'])}")
st.markdown(f'<span class="piq-sub">{esc(reqs["requester"])} · budget {inr(reqs["budget_inr"])} · '
            f'{len(reqs["requirements"])} requirements · run <code>{esc(run_id)}</code> · '
            f'status <b>{esc(status)}</b></span>', unsafe_allow_html=True)

excluded = [e for e in ev["quotes"] if e["status"] == "EXCLUDED"]
compliant = [e for e in ev["quotes"] if e["status"] == "COMPLIANT"]
cheapest = min(ev["quotes"], key=lambda e: e["price_pre_tax"] or 1e18)
rec = memo["recommended"]

k = st.columns(5)
k[0].metric("Quotes received", len(ev["quotes"]))
k[1].metric("Compliant", len(compliant))
k[2].metric("Excluded", len(excluded))
k[3].metric("Clarification rounds", state.get("round", 0))
if rec and cheapest["status"] == "EXCLUDED":
    k[4].metric("Cheapest quote", inr(cheapest["price_pre_tax"]))
    k[4].markdown(f'<span class="piq-badge" style="color:#b42318;background:#fde7e5">'
                  f'{1 - cheapest["price_pre_tax"] / rec["price_pre_tax"]:.0%} cheaper · EXCLUDED</span>',
                  unsafe_allow_html=True)
else:
    k[4].metric("Recommended", inr(rec["price_pre_tax"]) if rec else "—")

if rec:
    st.markdown(f'<div class="piq-banner"><b>Recommendation:</b> {esc(rec["vendor"])} at {inr(rec["price_pre_tax"])} '
                f'+ GST ({inr(rec["price_with_tax"])} incl.). '
                + (f'<br><span class="piq-sub">{esc(cheapest["vendor"])} is cheaper but was excluded: '
                   f'{esc(nice_failure(cheapest["failed"][0]))}</span>' if cheapest["status"] == "EXCLUDED" else "")
                + '<br><span class="piq-sub">Human approval required — the agent never auto-awards.</span></div>',
                unsafe_allow_html=True)
else:
    st.error("No quote meets all mandatory requirements. See escalations and decide manually.")

tabs = st.tabs(["Decision", "Claim verification", "Clarifications", "Try it yourself", "Quotes", "Requirements", "Audit log"])

# ------------------------------------------------------------------ Decision
with tabs[0]:
    left, right = st.columns([7, 4], gap="large")
    with left:
        st.markdown("##### Shortlist")
        rows = []
        for e in sorted(ev["quotes"], key=lambda x: (x["rank"] or 99, x["price_pre_tax"] or 0)):
            rows.append({"Rank": str(e["rank"] or "—"), "Vendor": f'{e["vendor_id"]} · {e["vendor"]}',
                         "Status": e["status"], "Price (pre-GST)": inr(e["price_pre_tax"]),
                         "Delivery": f'{e["delivery_days"]:g} d' if e["delivery_days"] else "—",
                         "Warranty": f'{e["warranty_years"]:g} y' if e["warranty_years"] else "—",
                         "Score": f'{e["score"]:.2f}' if e["score"] is not None else "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Rank": st.column_config.TextColumn(width=40),
                                    "Delivery": st.column_config.TextColumn(width=60),
                                    "Warranty": st.column_config.TextColumn(width=60),
                                    "Score": st.column_config.TextColumn(width=50)})

        st.markdown("##### Price vs delivery")
        fig = go.Figure()
        for e in ev["quotes"]:
            if not e["price_pre_tax"]:
                continue
            fg, _ = STATUS_STYLE[e["status"]]
            fig.add_trace(go.Scatter(
                x=[e["delivery_days"]], y=[e["price_pre_tax"] / 100000], mode="markers+text",
                marker=dict(size=14 + 8 * (e["warranty_years"] or 0), color=fg,
                            symbol="x" if e["status"] == "EXCLUDED" else "circle", opacity=.85),
                text=[f'{e["vendor_id"]} · {e["status"].lower()}'], textposition="top center",
                name=e["vendor"], hovertemplate=f'{e["vendor"]}<br>₹%{{y:.2f}} L · %{{x}} days<br>'
                                                f'warranty {e["warranty_years"]:g} y<extra></extra>'))
        ys = [e["price_pre_tax"] / 100000 for e in ev["quotes"] if e["price_pre_tax"]]
        fig.update_yaxes(range=[min(ys) - 0.3, max(ys) + 0.35])
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                          xaxis_title="Delivery (days)", yaxis_title="Price, ₹ lakh (pre-GST)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Bubble size = warranty. ✕ = excluded: cheap, but fails a mandatory spec.")

    with right:
        st.markdown("##### Human approval")
        decision = memo.get("decision")
        if status == "APPROVED" and decision:
            st.success(f"Awarded to **{decision['vendor']}** by {decision['approver']}"
                       + (f" — “{decision['note']}”" if decision["note"] else ""))
            if not decision["followed_recommendation"]:
                st.caption("Note: approver chose a compliant quote other than rank 1; logged.")
        elif status == "REJECTED":
            st.error("All quotes rejected by approver. See audit log.")
        else:
            options = [e["vendor_id"] for e in sorted(compliant, key=lambda x: x["rank"])]
            if options:
                with st.form("approve"):
                    vid = st.radio("Award to", options, horizontal=False,
                                   format_func=lambda v: f'{v} · {evals[v]["vendor"]} — {inr(evals[v]["price_pre_tax"])}'
                                                         f' (rank {evals[v]["rank"]})')
                    who = st.text_input("Approver", value="Facilities Manager")
                    note = st.text_input("Note (optional)")
                    if st.form_submit_button("Approve award", type="primary"):
                        try:
                            graph.approve(run_id, vid, who, note)
                            st.rerun()
                        except graph.ApprovalError as err:
                            st.error(str(err))
            st.caption("Only COMPLIANT quotes can be awarded. Excluded and conditional quotes are blocked.")
            with st.expander("Reject all quotes"):
                why = st.text_input("Reason", key="rej")
                if st.button("Reject all") and why:
                    graph.reject_all(run_id, "Facilities Manager", why)
                    st.rerun()

        if memo["excluded"]:
            st.markdown("##### Why excluded")
            for x in memo["excluded"]:
                st.markdown(f'**{esc(x["vendor_id"])} · {esc(x["vendor"])}** — {inr(x["price_pre_tax"])}')
                for c in x["evidence"]:
                    st.markdown(f'{badge(c["label"])} <b>{esc(SPEC_NAMES.get(c["key"], c["key"]))}</b><br>'
                                f'<span class="piq-sub">{esc(c["reason"])}</span>', unsafe_allow_html=True)
                    for evd in c["evidence"]:
                        st.markdown(f'<div class="piq-ev"><code>{esc(evd["id"])}</code> {esc(evd["text"])}</div>',
                                    unsafe_allow_html=True)
        if memo["escalations"]:
            st.markdown("##### Escalated to you")
            for c in memo["conditional"]:
                st.warning(f'**{c["vendor"]}** unresolved: ' + "; ".join(c["open_items"]))

        anomalies = [(e["vendor_id"], a) for e in ev["quotes"] for a in e["anomalies"] if a["severity"] != "info"]
        if anomalies:
            st.markdown("##### Anomalies")
            for vid_, a in anomalies:
                st.markdown(f'<div class="piq-ev">⚑ <b>{esc(vid_)}</b> {esc(a["code"])} — {esc(a["message"])}</div>',
                            unsafe_allow_html=True)

# ------------------------------------------------------------------ Claim verification
with tabs[1]:
    st.markdown("Every spec claim checked against the knowledge base **before** price is considered. "
                "<span style='color:#b42318'>●</span> = mandatory.", unsafe_allow_html=True)
    vids = list(evals)
    head = "".join(f"<th>{esc(v)} · {esc(evals[v]['vendor'])}<br>{status_badge(evals[v]['status'], evals[v]['rank'])}"
                   f"</th>" for v in vids)
    body = ""
    for r in reqs["requirements"]:
        cells = ""
        for v in vids:
            rule = next(x for x in evals[v]["rules"] if x["key"] == r["key"])
            if rule["label"]:
                cells += f"<td>{badge(rule['label'])}</td>"
            else:
                sym = {"pass": "VERIFIED", "fail": "CONTRADICTED", "unverified": "UNVERIFIABLE"}[rule["outcome"]]
                fg, bg, _ = LABEL_STYLE[sym]
                cells += (f'<td><span class="piq-badge" style="color:{fg};background:{bg}">'
                          f'{rule["outcome"].upper()}</span> <span class="piq-sub">{esc(rule["detail"])}</span></td>')
        req_txt = f"{r['op'].replace('_', ' ')} {r['value'] if r['value'] is not None else ''}"
        body += (f"<tr><td class='{'mand' if r['priority'] == 'mandatory' else ''}'><b>"
                 f"{esc(SPEC_NAMES.get(r['key'], r['key']))}</b><br><span class='piq-sub'>{esc(req_txt)}</span></td>"
                 f"{cells}</tr>")
    st.markdown(f"<table class='piq-matrix'><tr><th>Requirement</th>{head}</tr>{body}</table>",
                unsafe_allow_html=True)

    st.markdown("")
    st.markdown("##### Evidence trail")
    for v in vids:
        e = evals[v]
        with st.expander(f"{v} · {e['vendor']} — {e['status']}", expanded=(e["status"] == "EXCLUDED")):
            for c in e["claims"]:
                st.markdown(f'{badge(c["label"])} <b>{esc(SPEC_NAMES.get(c["key"], c["key"]))}</b> — '
                            f'{esc(c["reason"])}', unsafe_allow_html=True)
                for evd in c["evidence"]:
                    st.markdown(f'<div class="piq-ev"><code>{esc(evd["id"])}</code> {esc(evd["text"])}</div>',
                                unsafe_allow_html=True)

# ------------------------------------------------------------------ Clarifications
with tabs[2]:
    trace = state.get("agent_trace", [])
    if trace:
        st.markdown("##### Agent reasoning")
        st.caption("The model chooses which tools to call and writes the vendor message. Tools return facts; "
                   "they never let the model change a verdict. Guardrails catch anything it skips.")
        for i, t in enumerate(trace, 1):
            args = ", ".join(f"{k}={str(v)[:60]!r}" if k != "message" else "message=…" for k, v in t["args"].items())
            st.markdown(f'<div style="margin:.35rem 0"><b>{i}.</b> <code>{esc(t["tool"])}({esc(args)})</code> '
                        f'{source_pill(t.get("source", ""))}<br><span class="piq-sub">💭 {esc(t["thought"])}</span></div>',
                        unsafe_allow_html=True)
            obs = t.get("observation")
            if obs not in (None, ""):
                txt = obs if isinstance(obs, str) else "; ".join(f"{k}: {v}" for k, v in obs.items())
                st.markdown(f'<div class="piq-ev">→ {esc(txt[:400])}</div>', unsafe_allow_html=True)
        st.divider()
    hist = state.get("clarifications", [])
    if not hist:
        st.info("No clarifications were needed in this run.")
    for c in hist:
        vid = c["vendor_id"]
        before, after = init_status.get(vid), evals[vid]["status"]
        st.markdown(f"##### Round {c['round']} · {esc(c['vendor'])}  "
                    f"{status_badge(before)} → {status_badge(after, evals[vid]['rank'])}", unsafe_allow_html=True)
        a, b = st.columns(2)
        a.markdown("**Agent → vendor**  " + ("(written by the model)" if c.get("author") == "agent"
                                              else "(rule template)" if c.get("author") == "rules" else ""))
        a.code(c["sent"], language=None, wrap_lines=True)
        b.markdown("**Vendor → agent**  " + (source_pill(c.get("reply_source", "")) and "parsed by "
                                             + source_pill(c.get("reply_source", ""))), unsafe_allow_html=True)
        if c["reply"]:
            b.code(c["reply"], language=None, wrap_lines=True)
            q = quotes[vid]
            changes = [{"Field": SPEC_NAMES.get(k, k), "New value": fmt_val(q["fields"][k]["value"]),
                        "Confidence": q["fields"][k]["confidence"]} for k in c["changed"]]
            b.markdown("**Fields updated and re-verified**")
            b.dataframe(pd.DataFrame(changes), hide_index=True, width="stretch")
        else:
            b.warning("No reply. Not re-asked; escalated to a human when the loop ends.")
    if memo["escalations"]:
        st.warning("Escalated: " + ", ".join(memo["escalations"]))

# ------------------------------------------------------------------ Try it yourself
with tabs[3]:
    import copy as _copy

    from procureiq.schemas import Quote, RequirementSet
    from procureiq.tools.normalise import equivalence_flag, normalise_value
    from procureiq.tools.rules import evaluate as _evaluate

    st.markdown("Change what a vendor claimed and the **real verifier and rule engine re-run right here**. "
                "No model or recording is involved in this tab, so the result is computed live.")
    PRESETS = {
        "Vendor A's ISI licence is actually expired": ("A", "isi_licence", "CM/L-4100055501"),
        "Vendor C promises delivery in 20 days": ("C", "delivery_days", "20 days"),
        "Vendor B drops the words “or equivalent”": ("B", "conductor_material", "2.5 sqmm copper FR house wire"),
        "Vendor A's total doesn't match qty × rate": ("A", "total_price_inr", "Rs 4,02,000"),
        "Vendor B sends a valid-looking licence (another firm's)": ("B", "isi_licence", "CM/L-4100012345"),
    }
    EDITABLE = ["conductor_material", "isi_licence", "voltage_grade_kv", "fire_grade", "oxygen_index",
                "temperature_index_c", "delivery_days", "warranty_years", "unit_price_inr", "total_price_inr",
                "resistance_ohm_per_km", "quantity_coils"]

    left, right = st.columns([2, 3], gap="large")
    with left:
        preset = st.radio("Pick a scenario", list(PRESETS) + ["Custom edit"], index=0)
        if preset == "Custom edit":
            vid_e = st.selectbox("Vendor", list(quotes), format_func=lambda v: f"{v} · {quotes[v]['vendor']}")
            key_e = st.selectbox("Claim", EDITABLE, format_func=lambda k: SPEC_NAMES.get(k, k))
            cur = quotes[vid_e]["fields"].get(key_e, {}).get("raw", "")
            val_e = st.text_input("Vendor now says", value=cur)
        else:
            vid_e, key_e, val_e = PRESETS[preset]
            st.markdown(f"**{quotes[vid_e]['vendor']}** · {SPEC_NAMES.get(key_e, key_e)} → `{val_e}`")
        go_btn = st.button("Re-verify with this change", type="primary")

    with right:
        if go_btn:
            q_after = _copy.deepcopy(state["quotes"])
            q = next(x for x in q_after if x["vendor_id"] == vid_e)
            value, _ = normalise_value(key_e, None, val_e)
            q["fields"][key_e] = {"value": value, "unit": None, "confidence": 0.95, "raw": val_e,
                                  "flags": (["equivalence_clause"] if equivalence_flag(val_e) else []) + ["edited in Try it"]}
            if key_e == "conductor_material" and not equivalence_flag(val_e):
                q["fields"].pop("conductor_detail", None)
            rs = RequirementSet.model_validate(reqs)
            ev2 = _evaluate(rs, [Quote.model_validate(x) for x in q_after]).model_dump()
            before = {e["vendor_id"]: e for e in ev["quotes"]}
            after = {e["vendor_id"]: e for e in ev2["quotes"]}
            st.markdown("##### Result")
            rows = []
            for v in after:
                b, a = before[v], after[v]
                rows.append({"Vendor": f"{v} · {a['vendor']}",
                             "Before": b["status"] + (f" · rank {b['rank']}" if b["rank"] else ""),
                             "After": a["status"] + (f" · rank {a['rank']}" if a["rank"] else ""),
                             "Changed": "◀" if (b["status"], b["rank"]) != (a["status"], a["rank"]) else ""})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            b_claims = {c["key"]: c for c in before[vid_e]["claims"]}
            changed = [c for c in after[vid_e]["claims"] if b_claims[c["key"]]["label"] != c["label"]]
            for c in changed:
                st.markdown(f'{badge(b_claims[c["key"]]["label"])} → {badge(c["label"])} '
                            f'<b>{esc(SPEC_NAMES.get(c["key"], c["key"]))}</b><br>'
                            f'<span class="piq-sub">{esc(c["reason"])}</span>', unsafe_allow_html=True)
                for evd in c["evidence"][1:]:
                    st.markdown(f'<div class="piq-ev"><code>{esc(evd["id"])}</code> {esc(evd["text"])}</div>',
                                unsafe_allow_html=True)
            new_rules = {r["key"]: r for r in after[vid_e]["rules"] if r["label"] is None}
            for k, r in new_rules.items():
                old = next(x for x in before[vid_e]["rules"] if x["key"] == k)
                if old["outcome"] != r["outcome"]:
                    st.markdown(f"**{SPEC_NAMES.get(k, k)}**: {old['outcome'].upper()} → {r['outcome'].upper()} "
                                f"<span class='piq-sub'>({esc(r['detail'])})</span>", unsafe_allow_html=True)
            new_anoms = {a["code"] for a in after[vid_e]["anomalies"]} - {a["code"] for a in before[vid_e]["anomalies"]}
            for a in after[vid_e]["anomalies"]:
                if a["code"] in new_anoms:
                    st.markdown(f'<div class="piq-ev">⚑ new anomaly <b>{esc(a["code"])}</b> — {esc(a["message"])}</div>',
                                unsafe_allow_html=True)
            st.markdown(f"<span class='piq-sub'>New summary: {esc(ev2['summary'])}</span>", unsafe_allow_html=True)
            st.caption("This is a sandbox: the recorded run and the approval decision are not changed.")
        else:
            st.caption("Pick a scenario and press **Re-verify**.")

# ------------------------------------------------------------------ Quotes
with tabs[4]:
    sel = st.radio("Vendor", list(quotes), horizontal=True,
                   format_func=lambda v: f"{v} · {quotes[v]['vendor']} ({quotes[v]['channel']})")
    q = quotes[sel]
    src = Path(state["inputs"]["quotes"][sel]["file"])
    a, b = st.columns([2, 3], gap="large")
    with a:
        st.markdown(f"**Original input** · `{src.name}`")
        if src.suffix.lower() in (".png", ".jpg", ".jpeg"):
            st.image(str(src), width="stretch")
        elif src.suffix.lower() == ".pdf":
            try:
                import pymupdf as fitz

                with fitz.open(src) as doc:
                    st.image(doc[0].get_pixmap(dpi=110).tobytes("png"), width="stretch")
            except Exception:
                st.caption("(PDF preview unavailable)")
        elif src.exists():
            st.code(src.read_text(encoding="utf-8"), language=None, wrap_lines=True)
    with b:
        st.markdown("**Extracted and normalised fields** " + source_pill(state["sources"].get(f"quote_{sel}", "")),
                    unsafe_allow_html=True)
        rows = []
        for key, f in q["fields"].items():
            rows.append({"Field": SPEC_NAMES.get(key, key), "Value": fmt_val(f["value"]),
                         "Vendor's words": f["raw"], "Confidence": f["confidence"],
                         "Flags": ", ".join(f["flags"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=560,
                     column_config={"Confidence": st.column_config.ProgressColumn(
                         "Confidence", min_value=0.0, max_value=1.0, format="%.2f")})
        st.caption("Below 0.75 confidence a value is never judged; the agent reads it back to the vendor. "
                   "`updated:clarification_r1` marks values corrected by the vendor's reply.")

# ------------------------------------------------------------------ Requirements
with tabs[5]:
    a, b = st.columns([2, 3], gap="large")
    with a:
        st.markdown("**Buyer request (input)**")
        st.code(Path(state["inputs"]["request"]).read_text(encoding="utf-8"), language=None, wrap_lines=True)
    with b:
        st.markdown("**Extracted requirements** " + source_pill(state["sources"].get("requirements", "")),
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{
            "Spec": SPEC_NAMES.get(r["key"], r["key"]), "Rule": r["op"].replace("_", " "),
            "Value": "—" if r["value"] is None else str(r["value"]), "Priority": r["priority"],
            "From the email": r["source_text"]} for r in reqs["requirements"]]),
            hide_index=True, width="stretch", height=480)

# ------------------------------------------------------------------ Audit
with tabs[6]:
    evs = audit.events(run_id)
    st.dataframe(pd.DataFrame([{"Time (UTC)": e["ts"][11:19], "Step": e["step"], "Actor": e["actor"],
                                "Event": e["message"]} for e in evs]),
                 hide_index=True, width="stretch", height=560)
    st.caption("Every agent step, vendor message and human decision is written to SQLite.")
