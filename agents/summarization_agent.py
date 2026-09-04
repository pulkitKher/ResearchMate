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
    if not findings:
        return  {"summary": ["No verified findings were available for this topic after research and verification."]}

    findings_text = "\n\n".join(
        f"Title: {f.get('title', 'unknown')}\n"
        f"URL: {f.get('url', 'unknown')}\n"
        f"Key Point: {f.get('key_point', '')}"
        for f in findings
    )

    prompt = f"""You are a research summarization assistant.

Below are verified key points from research on the topic: "{state['topic']}"

{findings_text}

Extract the key insights as a list of concise bullet points. Each bullet should:
- Capture ONE distinct insight (merge overlapping key points from different sources into one bullet where they agree)
- Be factual and specific
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