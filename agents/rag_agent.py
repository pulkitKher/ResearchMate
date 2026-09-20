"""
agents/rag_agent.py
Retrieves relevant chunks for a user's document question and generates
a grounded answer, citing the page(s) the answer came from.
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ingest import load_faiss_index

load_dotenv()

TOP_K = 4

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

RAG_PROMPT_TEMPLATE = """You are answering a question using ONLY the document excerpts below.
Each excerpt has a chunk_id. Do not use outside knowledge. If the excerpts don't
contain enough information to answer, say so clearly — do not guess or fabricate.

Question: {question}

Document excerpts:
{excerpts}

Respond ONLY in this JSON format, no markdown, no preamble:
{{
  "answer": "<your answer, grounded only in the excerpts above>",
  "used_chunk_ids": [<chunk_id ints of excerpts you actually relied on>],
  "sufficient_context": <true or false — was there enough info to answer confidently>
}}
"""


def rag_agent_node(state: dict, vectorstore=None) -> dict:
    question = state["doc_question"]

    if vectorstore is None:
        vectorstore = load_faiss_index()

    retrieved_docs = vectorstore.similarity_search(question, k=TOP_K)
    # ... rest unchanged

    retrieved_chunks = [
        {
            "chunk_id": doc.metadata["chunk_id"],
            "page": doc.metadata["page"],
            "text": doc.page_content,
        }
        for doc in retrieved_docs
    ]

    excerpts_text = "\n\n".join(
        f"[chunk_id: {c['chunk_id']} | page {c['page']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = RAG_PROMPT_TEMPLATE.format(question=question, excerpts=excerpts_text)
    response = llm.invoke(prompt)

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        parsed = json.loads(raw)
        answer = parsed.get("answer", "")
        used_ids = parsed.get("used_chunk_ids", [])
        sufficient = parsed.get("sufficient_context", False)
    except (json.JSONDecodeError, AttributeError):
        answer = "Could not parse a grounded answer from the model response."
        used_ids = []
        sufficient = False

    cited_pages = sorted({
        c["page"] for c in retrieved_chunks if c["chunk_id"] in used_ids
    }) if sufficient else []

    state["doc_answer"] = answer
    state["doc_citations"] = cited_pages
    return state


if __name__ == "__main__":
    test_state = {"doc_question": "Who is responsible for maintaining the property?"}
    result = rag_agent_node(test_state)
    print("Answer:", result["doc_answer"])
    print("Cited pages:", result["doc_citations"])