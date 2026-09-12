# Agentic Profile Matching (Milestone 3)

LangGraph-based recruiting agent that parses a job description, extracts must-have vs nice-to-have requirements, searches resumes (filesystem + RAG), ranks candidates, generates explainable reports, and refines results from human feedback.

## Deliverables checklist

| Requirement | Location |
|-------------|----------|
| LangGraph agent | `matching_agent.py`, `agent/` |
| State: history, JD understanding, shortlist, reasoning | `agent/state.py` |
| Workflow START→Parse→Extract→Search→Rank→Report→Feedback→END | `agent/graph.py`, `docs/state_machine.md` |
| FS tools (M1) | `tools/fs_tools.py` |
| RAG search (M2) | `tools/rag_search.py`, `core/resume_rag.py` |
| `extract_requirements` | `tools/extract_requirements.py` |
| `compare_candidates` | `tools/compare_candidates.py` |
| `generate_interview_questions` | `tools/interview_questions.py` |
| Conversational NL interface | `app/cli.py`, `app/streamlit_app.py` |
| Iterative re-rank + change explanations | `agent/nodes.py`, `screening/multi_round.py` |
| Multi-round screening + hire/no-hire | `screening/multi_round.py` |
| Explainability reports | `screening/explainability.py` |
| State machine diagram | `docs/state_machine.md` |
| 5+ test conversation flows | `tests/test_scenarios.py`, `tests/conversation_flows.md` |

## Folder structure

```
agenticProfileMatching/
├── matching_agent.py          # Main LangGraph agent entrypoint
├── config.py
├── agent/                     # State, graph, nodes, intent router
├── tools/                     # FS, RAG, extract/compare/interview
├── core/                      # RAG index + hybrid JobMatcher + errors
├── screening/                 # Multi-round + explainability
├── app/                       # CLI + Streamlit chat UI
├── resumes/                   # 34 sample resumes
├── job_descriptions/          # 6 sample JDs
├── scripts/setup_index.py
├── docs/state_machine.md
├── tests/
└── reports/                   # Generated match reports
```

## Setup

```bash
cd agenticProfileMatching
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # optional OPENAI_API_KEY

python scripts/setup_index.py
```

Default embeddings use ChromaDB’s bundled ONNX MiniLM (no API key).

## Usage

### One-shot query

```bash
python matching_agent.py "Find me candidates with React and 3+ years experience"
```

### Interactive CLI

```bash
python matching_agent.py -i
# or
python -m app.cli
```

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

### Multi-round screening

```bash
python matching_agent.py --multi-round job_descriptions/full_stack_developer.json
```

### Tests

```bash
python tests/test_scenarios.py
pytest tests/test_scenarios.py -v
```

## Example conversation

```
You> Find me candidates with React and 3+ years experience
Agent> Shortlist with scores + reasoning…

You> Compare the top 3 matches side by side
Agent> Shared/unique skills, experience ranking…

You> prioritize TypeScript and PostgreSQL
Agent> Re-ranked shortlist + what moved up/down…

You> multi-round screening
Agent> Round1 top10 → deep analysis → HIRE/MAYBE/NO_HIRE

You> done
```

## Demo video outline (5–6 min)

1. **0:00–0:40** — Architecture: show `docs/state_machine.md` + folder layout  
2. **0:40–2:00** — CLI/Streamlit: React + 3 years search; show reasoning trail  
3. **2:00–3:00** — Compare top 3; explain why A > B  
4. **3:00–4:00** — Refine requirements mid-chat; show ranking changes  
5. **4:00–5:20** — Multi-round screening + strengths/gaps + hire decisions  
6. **5:20–6:00** — Interview questions + saved report in `reports/`

## Error handling

- Tools return `{success, error: {message, code}}` envelopes (`core/exceptions.py`)
- Graph routes failures to an `error` node with recoverable user messaging
- Missing index → clear instruction to run `scripts/setup_index.py`
- Unknown candidates / empty JD → validation errors without crashing the session

## License

Academic assignment project.
