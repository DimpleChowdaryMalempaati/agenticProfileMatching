"""CLI conversational interface for the matching agent."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path when run as module or script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matching_agent import MatchingAgent


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         Agentic Profile Matching — CLI Chat                 ║
║  LangGraph: Parse JD → Extract → Search → Rank → Report     ║
╠══════════════════════════════════════════════════════════════╣
║  Examples:                                                   ║
║   • Find me candidates with React and 3+ years experience    ║
║   • Compare the top 3 matches side by side                   ║
║   • Why did Alice rank higher than Bob?                      ║
║   • multi-round                                              ║
║   • prioritize AWS and Docker                                ║
║   • done                                                     ║
╚══════════════════════════════════════════════════════════════╝
"""


def interactive_chat() -> None:
    print(BANNER)
    agent = MatchingAgent()
    print("Type your request (or 'quit' to exit).\n")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        try:
            if user_input.lower() in {"multi-round", "multi round"} and agent.state.get("jd_text"):
                state = agent.run("multi-round screening")
            elif agent.state.get("awaiting_feedback") and user_input.lower() not in {
                "find",
            }:
                # Treat follow-ups as feedback/refinement unless clearly a new search
                if user_input.lower().startswith("find ") or user_input.lower().startswith(
                    "search "
                ):
                    state = agent.run(user_input)
                else:
                    state = agent.apply_feedback(user_input)
            else:
                state = agent.run(user_input)
        except Exception as exc:  # noqa: BLE001
            print(f"\nAgent error: {exc}\n")
            continue

        response = state.get("assistant_response") or state.get("error") or "(no response)"
        print(f"\nAgent>\n{response}\n")

        reasoning = state.get("reasoning") or []
        if reasoning:
            print("--- Reasoning trail ---")
            for line in reasoning[-6:]:
                print(f"  • {line}")
            print()


if __name__ == "__main__":
    interactive_chat()
