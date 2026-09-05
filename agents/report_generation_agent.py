from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
)

def report_generation_agent(state):
    topic = state["topic"]
    summary = state.get("summary", [])
    findings = state.get("all_findings", [])
    verification_status = state.get("verification_status", "unknown")

    summary_text = "\n".join(f"- {point}" for point in summary)

    sources_text = "\n".join(
        f"- [{f.get('title', 'Untitled')}]({f.get('url', '')})"
        for f in findings
    )

    research_failed = verification_status == "failed" or not findings

    failure_notice = """
IMPORTANT: Research verification FAILED for this topic — no credible, relevant 
sources were found after multiple search attempts. This likely means the topic 
is fictional, too obscure, too recent to have coverage, or was misinterpreted. 
Every section, including the Introduction and Conclusion, MUST honestly reflect 
this. Do NOT frame the topic as real, significant, "emerging," or "under-researched" 
unless the sources actually support that. Simply and plainly state that reliable 
information could not be found, and suggest the user verify the topic or rephrase it.
""" if research_failed else ""

    prompt = f"""You are a research report generation agent. Produce a well-structured Markdown report.

Topic: {topic}
{failure_notice}
Key Insights (from Summarization Agent):
{summary_text if summary_text else "No verified insights available."}

Available Sources:
{sources_text if sources_text else "No sources available."}

Write a Markdown report with these sections, in this order:
## Introduction
A 2-3 sentence framing of the topic. If research failed (see notice above), 
state plainly that reliable sources could not be found for this topic — do not 
speculate on the topic's importance or framing.

## Key Findings
Present the key insights as a clear bullet list. If no insights are available, 
state that clearly and honestly — do not fabricate findings.

## Research Gaps & Open Questions
If findings exist, identify 2-4 specific things not well-covered or contradicted 
across sources. If research failed entirely, state that no gap analysis is 
possible without verified sources — do not invent gaps about the topic itself.

## Sources
List all sources as a Markdown link list, or state none are available.

## Conclusion
2-3 sentence wrap-up. If research failed, plainly recommend the user verify or 
rephrase the topic — do not summarize as if the topic was validly explored.

Do not include any text outside these five sections. Do not fabricate facts, 
sources, or claims about the topic's significance not supported by the material above.
"""

    response = llm.invoke(prompt)
    return {"final_report": response.content.strip()}