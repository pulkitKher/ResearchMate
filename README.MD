# ResearchMate

**AI-Based Multi-Agent Research and Report Generation System**

ResearchMate is an agentic AI research assistant that uses multiple specialized AI agents — each responsible for a distinct part of the research workflow — instead of relying on a single LLM call to search, verify, summarize, and write. The core differentiator is a **Research Agent → Fact Verification Agent loop** built with LangGraph's conditional routing: when verification fails, the system triggers a targeted re-search using the failure reason as feedback, rather than blindly retrying or returning unverified output.

Minor project (7th Semester) | Team: Pulkit Kher, Shivansh Akar | Mentor: Mr. P.K. Sharma

---

## Why this isn't "just a ChatGPT wrapper"

A single LLM call can't judge or retry its own output. ResearchMate has an independent Verification Agent that evaluates Research Agent output and — if it fails — triggers an automatic, *refined* re-search via a conditional edge in the LangGraph state graph. That loop is the concrete, demonstrable proof of "agentic" behavior, not just prompt chaining.

---

## Project Status

### ✅ Phase 1 — Research & Verification Core (Complete)
- LangGraph state graph with conditional re-search loop
- Research Agent (retrieval-only, Tavily-based)
- Fact Verification Agent (LLM-judged, structured JSON output)
- Retry-capped loop with graceful failure handling
- End-to-end tested on both success and failure paths

### 🔜 Phase 2 — Summarization + Report Generation Agents
### 🔜 Phase 3 — FastAPI Backend
### 🔜 Phase 4 — MySQL Integration (phpMyAdmin / cPanel)
### 🔜 Phase 5 — Frontend (React/Bootstrap) + Google Auth
### 🔜 Phase 6 — Docker + Deployment (siddharthraje.com)

---

## Architecture (Planned — 5 Agents)

```
User Query / Uploaded Document
        │
        ▼
 ┌─────────────────┐      fails       ┌───────────────────┐
 │ Research Agent   │ ◄───────────────│ Fact Verification  │
 │ (Tavily search)  │   refined query │ Agent (LLM judge)  │
 └─────────────────┘                  └───────────────────┘
        │                                      │ passes
        │                                      ▼
        │                          ┌───────────────────────┐
        │                          │ Summarization Agent    │
        │                          └───────────────────────┘
        │                                      │
        │                                      ▼
        │                          ┌───────────────────────┐
        │                          │ Report Generation Agent│
        │                          └───────────────────────┘
        │                                      │
        ▼                                      ▼
 ┌─────────────────┐              Structured Report (MySQL)
 │   RAG Agent      │──── cross-checks doc claims vs web ────┘
 │ (FAISS, PDFs)     │
 └─────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | `openai/gpt-4o-mini` via OpenRouter |
| Web Search | Tavily |
| Backend (planned) | FastAPI |
| Vector Store (planned) | FAISS |
| Database (planned) | MySQL (phpMyAdmin / cPanel) |
| Frontend (planned) | React or Bootstrap + Google Auth |
| Deployment (planned) | Docker |
| Environment | Windows, Git Bash, Python venv |

---

## Project Structure (Phase 1)

```
ResearchMate/
├── venv/                          # virtual environment (not committed)
├── agents/
│   ├── research_agent.py          # retrieval-only node (Tavily)
│   └── verification_agent.py      # LLM judgment node (structured JSON)
├── state.py                       # ResearchState TypedDict (shared graph state)
├── graph.py                       # LangGraph StateGraph + conditional re-search loop
├── main.py                        # end-to-end runner
├── .env                           # API keys (not committed)
├── .gitignore
└── requirements.txt
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate      # Git Bash on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env in project root
OPENROUTER_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here

# 4. Run the Phase 1 pipeline
python main.py
```

---

## How the Re-Search Loop Works

1. **Research Agent** searches Tavily for the topic (or a refined query on retry).
2. **Fact Verification Agent** (`gpt-4o-mini`, `temperature=0`) judges relevance, source credibility, and cross-source agreement, returning structured JSON: `status`, `reason`, `verified_findings`.
3. **Router** (a conditional edge, not an agent) reads `verification_status`:
   - `"passed"` → findings are accumulated, graph reaches `END`.
   - `"failed"` and retries remain → `retry_count` is incremented and the **failure reason is passed back into Research Agent** to refine its next query — not a blind repeat.
   - `"failed"` and `max_retries` exhausted → graph stops gracefully with an honest `"failed"` status and zero fabricated findings.

### Verified test behavior
- **Success case:** real topic passed verification on the first attempt, returning 3 structured, cited findings.
- **Failure case:** a deliberately unanswerable topic failed verification 3 times in a row — each with a distinct, specific reason — with the Research Agent visibly refining its query each retry, before stopping cleanly at the retry cap with 0 fabricated findings.

---

## Advantages & Disadvantages

**Advantages:** self-correcting (not just self-generating), transparent per-decision reasoning, targeted rather than blind retries, bounded execution (no infinite loops), clean separation of concerns between agents, no fabricated output on failure.

**Disadvantages:** retries increase latency/cost, verification quality depends on a single LLM's judgment (not ground-truth fact-checking), a fixed retry cap is a blunt instrument, and results aren't reused/merged across retry rounds.

---

## License

Academic project — Bhartiya Institute of Technology, 7th Semester Minor Project.