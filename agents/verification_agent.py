import os
import json
from langchain_openai import ChatOpenAI
from state import ResearchState

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

VERIFICATION_PROMPT = """You are a strict fact-verification agent. You will be given a research topic and a list of search results (title, url, content snippet).

Your job:
1. Judge whether the results are RELEVANT to the topic.
2. Judge whether the sources appear CREDIBLE (reputable domains, not spam/ad content).
3. Judge whether there is enough CROSS-SOURCE AGREEMENT (at least 2 sources support the same key claims), where applicable.

Respond ONLY with valid JSON in this exact format, no markdown, no extra text:
{{
  "status": "passed" or "failed",
  "reason": "If failed, a specific and actionable reason describing what is missing or wrong, so a re-search can be targeted. If passed, a short confirmation summary.",
  "verified_findings": [
    {{"title": "...", "url": "...", "key_point": "one sentence summary of what this source confirms"}}
  ]
}}

Topic: {topic}

Search Results:
{results}
"""


def verify_node(state: ResearchState) -> dict:
    topic = state["topic"]
    results = state["search_results"]

    if not results:
        print("[Verification Agent] No results to verify — auto-fail")
        return {
            "verification_status": "failed",
            "verification_reason": "No search results were returned at all. Try a broader or rephrased query.",
        }

    results_text = "\n\n".join(
        f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
        for r in results
    )

    prompt = VERIFICATION_PROMPT.format(topic=topic, results=results_text)

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Defensive cleanup in case the model wraps output in markdown fences
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print("[Verification Agent] Failed to parse LLM output as JSON")
        return {
            "verification_status": "failed",
            "verification_reason": "Verification agent output was malformed; retrying search.",
        }

    status = parsed.get("status", "failed")
    reason = parsed.get("reason", "")
    verified_findings = parsed.get("verified_findings", [])

    print(f"[Verification Agent] Status: {status} | Reason: {reason}")

    update = {
        "verification_status": status,
        "verification_reason": reason,
    }

    # Only accumulate findings into all_findings if verification passed
    if status == "passed":
        existing = state.get("all_findings", [])
        update["all_findings"] = existing + verified_findings

    return update