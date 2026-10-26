"""Run the full agent on the demo and print the audit trail.

    python -m procureiq.agent
"""
from .. import audit
from .graph import run


def main() -> None:
    state = run()
    print(f"\nRun {state['run_id']}  →  {state['status']}  ({state.get('round', 0)} clarification round(s))\n")
    for e in audit.events(state["run_id"]):
        print(f"  {e['step']:<22} {e['actor']:<10} {e['message']}")
    memo = state["memo"]
    print(f"\n{memo['summary']}")
    for c in state.get("clarifications", []):
        print(f"\n--- clarification round {c['round']} to {c['vendor']} ---\n{c['sent']}")
    print("\nNo award is made automatically. Approve via the dashboard or "
          "POST /runs/{run_id}/approve.\n")


if __name__ == "__main__":
    main()
