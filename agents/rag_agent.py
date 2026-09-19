"""
agents/rag_agent.py
Retrieves relevant chunks for a user's document question and generates
a grounded answer, citing the page(s) the answer came from.
"""

import json
from langchain_openai import ChatOpenAI
from ingest import load_faiss_index

TOP_K = 4

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
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


def rag_agent_node(state: dict) -> dict:
    """
    LangGraph-compatible node. Expects state['doc_question'] and
    state['document_chunks'] (from ingest.load_and_chunk, already in state
    or reloaded via FAISS index for retrieval).
    """
    question = state["doc_question"]
    all_chunks = state["document_chunks"]  # list of {chunk_id, page, text}

    vectorstore = load_faiss_index()
    retrieved_docs = vectorstore.similarity_search(question, k=TOP_K)

    # Map retrieved langchain Documents back to our chunk dicts by matching text
    # (simplest reliable join given how FAISS.from_documents preserved page_content)
    retrieved_chunks = []
    for doc in retrieved_docs:
        for chunk in all_chunks:
            if chunk["text"] == doc.page_content:
                retrieved_chunks.append(chunk)
                break

    excerpts_text = "\n\n".join(
        f"[chunk_id: {c['chunk_id']} | page {c['page']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = RAG_PROMPT_TEMPLATE.format(question=question, excerpts=excerpts_text)
    response = llm.invoke(prompt)

    # Defensive parsing — same pattern as your verification_agent
    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        parsed = json.loads(raw)
        answer = parsed.get("answer", "")
        used_ids = parsed.get("used_chunk_ids", [])
    except (json.JSONDecodeError, AttributeError):
        answer = "Could not parse a grounded answer from the model response."
        used_ids = []

    # Map used chunk_ids back to page numbers for citation
    cited_pages = sorted({
        c["page"] for c in retrieved_chunks if c["chunk_id"] in used_ids
    })

    state["doc_answer"] = answer
    state["doc_citations"] = cited_pages
    return state