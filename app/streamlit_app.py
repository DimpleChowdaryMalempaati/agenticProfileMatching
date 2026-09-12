"""Streamlit chat UI for the LangGraph matching agent."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matching_agent import MatchingAgent

st.set_page_config(
    page_title="Agentic Profile Matching",
    page_icon="🔎",
    layout="wide",
)

st.title("Agentic Profile Matching")
st.caption(
    "LangGraph agent · Parse JD → Extract Requirements → Search → Rank → Report → Feedback"
)

with st.sidebar:
    st.header("Session")
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.agent = MatchingAgent()
        st.session_state.chat = []
        st.rerun()

    st.markdown("### Try these")
    st.code("Find me candidates with React and 3+ years experience", language=None)
    st.code("Compare the top 3 matches side by side", language=None)
    st.code("Why did Alice rank higher than Bob?", language=None)
    st.code("prioritize AWS and Docker", language=None)
    st.code("multi-round screening for full stack", language=None)

    agent: MatchingAgent = st.session_state.get("agent") or MatchingAgent()
    st.session_state.agent = agent
    if agent.state.get("shortlist"):
        st.markdown("### Current shortlist")
        for c in agent.state["shortlist"][:8]:
            st.write(f"**{c.get('rank')}.** {c.get('candidate_name')} — {c.get('match_score')}")

if "agent" not in st.session_state:
    st.session_state.agent = MatchingAgent()
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Ask the matching agent…")
if prompt:
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = st.session_state.agent
    with st.chat_message("assistant"):
        with st.spinner("Agent reasoning…"):
            try:
                if agent.state.get("awaiting_feedback") and not prompt.lower().startswith(
                    ("find ", "search ", "match ")
                ):
                    state = agent.apply_feedback(prompt)
                else:
                    state = agent.run(prompt)
                response = state.get("assistant_response") or state.get("error") or "No response"
            except Exception as exc:  # noqa: BLE001
                response = f"Error: {exc}"
                state = agent.state

        st.markdown(response)
        reasoning = state.get("reasoning") or []
        if reasoning:
            with st.expander("Agent reasoning trail"):
                for line in reasoning:
                    st.write(f"• {line}")
        if state.get("report"):
            with st.expander("Full match report"):
                st.markdown(state["report"])
        if state.get("recommendations"):
            with st.expander("Hire / No-hire decisions"):
                for r in state["recommendations"]:
                    st.write(
                        f"**{r.get('decision')}** — {r.get('candidate_name')} "
                        f"(score {r.get('match_score')}): {r.get('rationale')}"
                    )

    st.session_state.chat.append(("assistant", response))
