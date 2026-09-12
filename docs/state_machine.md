# State Machine Diagram — Matching Agent

## LangGraph workflow

```mermaid
flowchart TD
    START([START]) --> route[route_intent]

    route -->|search / refine / full_pipeline| parse[Parse JD]
    route -->|compare| compare[compare_candidates]
    route -->|explain_rank| explain[Explain ranking]
    route -->|interview| interview[generate_interview_questions]
    route -->|multi_round| multi[Multi-round screening]
    route -->|report| report[Generate Report]
    route -->|error| err[Error handler]

    parse --> extract[Extract Requirements]
    extract --> search[Search Resumes<br/>RAG + hybrid matcher]
    search --> rank[Rank Candidates]
    rank --> report

    report --> END1([END / await feedback])
    compare --> END2([END / await feedback])
    explain --> END3([END / await feedback])
    interview --> END4([END / await feedback])
    multi --> END5([END / await feedback])
    err --> END6([END / await feedback])

    feedback[Human Feedback Loop] -->|refine criteria| parse
    feedback -->|compare / explain / interview| route
    feedback -->|done| STOP([END])

    END1 -.->|user reply| feedback
```

## Agent state (tracked fields)

| Field | Purpose |
|-------|---------|
| `messages` | Conversation history |
| `jd_text` / `jd_title` | Job understanding |
| `requirements` | Must-have vs nice-to-have |
| `shortlist` / `reasoning` | Ranked candidates + trail |
| `refined_criteria` | Mid-conversation adjustments |
| `ranking_changes` | Explain re-rank deltas |
| `round1/2/3` artifacts | Multi-round screening |
| `awaiting_feedback` | Human-in-the-loop flag |

## Happy path (assignment wording)

**START → Parse JD → Extract Requirements → Search Resumes → Rank Candidates → Generate Report → Human Feedback Loop → END**
