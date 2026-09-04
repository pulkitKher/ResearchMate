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

    summary_text = "\n".join(f"- {point}" for point in summary)

    sources_text = "\n".join(
        f"- [{f.get('title', 'Untitled')}]({f.get('url', '')})"
        for f in findings
    )

    prompt = f"""You are a research report generation agent. Produce a well-structured Markdown report.

Topic: {topic}

Key Insights (from Summarization Agent):
{summary_text if summary_text else "No verified insights available."}

Available Sources:
{sources_text if sources_text else "No sources available."}

Write a Markdown report with these sections, in this order:
## Introduction
A 2-3 sentence framing of the topic and why it matters.

## Key Findings
Present the key insights as a clear bullet list, grouped logically if there are natural clusters. If no insights are available, state that clearly and honestly — do not fabricate findings.

## Research Gaps & Open Questions
Based on the key insights and sources above, identify 2-4 specific things that appear NOT to be well-covered, unresolved, or contradicted across sources. If findings are insufficient to identify real gaps, say so honestly instead of inventing gaps.

## Sources
List all sources as a Markdown link list.

## Conclusion
2-3 sentence wrap-up.

Do not include any text outside these five sections. Do not fabricate facts, sources, or gaps not supported by the material above.
"""

    response = llm.invoke(prompt)
    return {"final_report": response.content.strip()}