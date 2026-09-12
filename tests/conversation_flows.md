# Conversation Test Flows (5+)

These flows exercise Part A–C deliverables. Run after indexing:

```bash
python scripts/setup_index.py
python tests/test_scenarios.py
# or
pytest tests/test_scenarios.py -v
```

## Flow 1 — Natural language search
**User:** Find me candidates with React and 3+ years experience  
**Expected:** Agent parses query as JD, extracts React + years, RAG/hybrid search, ranked shortlist with reasoning.

## Flow 2 — Side-by-side compare
**User:** Find me full stack developers with React and Node.js  
**User:** Compare the top 3 matches side by side  
**Expected:** `compare_candidates` summary with shared/unique skills and experience ranking.

## Flow 3 — Explainability of ranks
**User:** Match candidates for full stack developer role with React  
**User:** Why did \<A\> rank higher than \<B\>?  
**Expected:** Score delta, matched-skill differences, and per-candidate reasoning.

## Flow 4 — Iterative refinement
**User:** Find me candidates with Python and machine learning  
**User:** prioritize AWS and Docker experience  
**Expected:** Requirements updated, re-rank, explanation of ranking changes.

## Flow 5 — Multi-round screening
**User / CLI:** `python matching_agent.py --multi-round job_descriptions/full_stack_developer.json`  
**Expected:** Round 1 top 10 → Round 2 deep strengths/gaps → Round 3 HIRE / MAYBE / NO_HIRE.

## Flow 6 — Interview pack
**User:** Find senior python ML engineers  
**User:** Generate interview questions for \<top candidate\>  
**Expected:** Behavioral + technical + gap-probe questions.

## Flow 7 — JD file full pipeline
**User:** `job_descriptions/data_scientist.json` (or path)  
**Expected:** Full graph path Parse → Extract → Search → Rank → Report.
