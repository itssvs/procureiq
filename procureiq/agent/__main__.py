"""Run the full agent on the demo and print what happened.

    python -m procureiq.agent
"""
from .. import audit
from ..llm import last_error_hint, provider
from .graph import run


def main() -> None:
    print(f"LLM provider: {provider()}  (set LLM_PROVIDER in .env)")
    state = run()
    print(f"\nRun {state['run_id']}  →  {state['status']}  ({state.get('round', 0)} clarification round(s))\n")
    print("WHO PRODUCED EACH AI STEP")
    for step, src in state.get("sources", {}).items():
        print(f"  {step:<22} {src}")
    if any("live failed" in s or s == "seed" for s in state.get("sources", {}).values()) and provider() != "cached":
        print(f"  ! some live calls failed — {last_error_hint()}")

    print("\nAGENT REASONING")
    for i, t in enumerate(state.get("agent_trace", []), 1):
        print(f"  {i}. [{t['source']}] {t['tool']}({', '.join(f'{k}={str(v)[:50]!r}' for k, v in t['args'].items())})")
        if t["thought"]:
            print(f"       thought: {t['thought']}")
        obs = t["observation"]
        print(f"       result : {str(obs)[:160]}")

    print("\nAUDIT TRAIL")
    for e in audit.events(state["run_id"]):
        print(f"  {e['step']:<22} {e['actor']:<10} {e['message'][:150]}")
    print(f"\n{state['memo']['summary']}")
    for c in state.get("clarifications", []):
        print(f"\n--- message to {c['vendor']} (round {c['round']}, {c['author']}) ---\n{c['sent']}")
    print("\nNo award is made automatically. Approve in the dashboard (streamlit run app.py).\n")


if __name__ == "__main__":
    main()
