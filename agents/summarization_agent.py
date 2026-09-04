from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
)

def summarization_agent(state):
    findings = state["all_findings"]

    findings_text = "\n\n".join(
        f"Source: {f.get('source', 'unknown')}\nContent: {f.get('content', '')}"
        for f in findings
    )

    prompt = f"""You are a research summarization assistant.

Below are verified research findings on the topic: "{state['topic']}"

{findings_text}

Extract the key insights as a list of concise bullet points. Each bullet should:
- Capture ONE distinct insight (no overlap between bullets)
- Be factual and specific (include numbers/dates where present in the findings)
- Avoid restating the topic itself

Return ONLY the bullet points, one per line, no numbering, no preamble.
"""

    response = llm.invoke(prompt)
    bullets = [
        line.strip("-• ").strip()
        for line in response.content.split("\n")
        if line.strip()
    ]

    return {"summary": bullets}