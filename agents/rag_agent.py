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

EXTRACTION_FIELDS = {
    "methodology": "What methodology, approach, or model architecture does this paper use?",
    "dataset": "What dataset(s) were used for training or evaluation in this paper?",
    "results": "What were the key results, findings, or performance metrics reported?",
}

EXTRACTION_PROMPT_TEMPLATE = """You are extracting a specific piece of information from a research paper
using ONLY the excerpts below. Do not use outside knowledge.

Field to extract: {field_label}
Guiding question: {field_question}

Document excerpts:
{excerpts}

Respond ONLY in this JSON format, no markdown, no preamble:
{{
  "value": "<concise extracted answer for this field, or 'Not found in document' if the excerpts don't cover it>",
  "used_chunk_ids": [<chunk_id ints actually relied on>],
  "sufficient_context": <true or false>
}}
"""


def _extract_field(field_label: str, field_question: str, vectorstore) -> dict:
    """Run one targeted retrieval + extraction for a single literature-review field."""
    retrieved_docs = vectorstore.similarity_search(field_question, k=TOP_K)

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

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        field_label=field_label,
        field_question=field_question,
        excerpts=excerpts_text,
    )
    response = llm.invoke(prompt)

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        parsed = json.loads(raw)
        value = parsed.get("value", "Not found in document")
        used_ids = parsed.get("used_chunk_ids", [])
        sufficient = parsed.get("sufficient_context", False)
    except (json.JSONDecodeError, AttributeError):
        value = "Could not parse extraction result."
        used_ids = []
        sufficient = False

    cited_pages = sorted({
        c["page"] for c in retrieved_chunks if c["chunk_id"] in used_ids
    }) if sufficient else []

    return {"value": value, "citations": cited_pages}


def literature_review_node(state: dict, vectorstore=None) -> dict:
    """
    LangGraph-compatible node. Extracts structured literature-review fields
    (methodology, dataset, results) from the currently loaded document.
    """
    if vectorstore is None:
        vectorstore = load_faiss_index()

    extraction = {}
    for field_label, field_question in EXTRACTION_FIELDS.items():
        extraction[field_label] = _extract_field(field_label, field_question, vectorstore)

    state["literature_review"] = extraction
    return state

CONTRADICTION_PROMPT_TEMPLATE = """You are comparing a claim from an uploaded document against findings
gathered from independent web research, to check for contradictions.

Document claim: {claim}

Web research findings:
{web_findings}

Does the web research agree with, contradict, or say nothing relevant to this claim?
Respond ONLY in this JSON format, no markdown, no preamble:
{{
  "verdict": "agrees" | "contradicts" | "unrelated",
  "explanation": "<one sentence explaining the verdict, citing which finding if contradicts>",
  "conflicting_finding_title": "<title of the specific web finding that conflicts, or null if not 'contradicts'>"
}}
"""


def _check_claim_against_web(claim: str, all_findings: list) -> dict:
    """Ask the LLM whether a single document claim agrees/contradicts/is unrelated
    to the web findings already gathered by the Research+Verification agents."""
    if not all_findings:
        return {"claim": claim, "verdict": "unrelated", "explanation": "No web findings available to compare.", "conflicting_finding_title": None}

    findings_text = "\n".join(
        f"- {f.get('title', 'Untitled')}: {f.get('key_point', '')} (source: {f.get('url', 'N/A')})"
        for f in all_findings
    )

    prompt = CONTRADICTION_PROMPT_TEMPLATE.format(claim=claim, web_findings=findings_text)
    response = llm.invoke(prompt)

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        parsed = json.loads(raw)
    except (json.JSONDecodeError, AttributeError):
        parsed = {"verdict": "unrelated", "explanation": "Could not parse comparison result.", "conflicting_finding_title": None}

    parsed["claim"] = claim
    return parsed


def contradiction_check_node(state: dict, vectorstore=None) -> dict:
    """
    LangGraph-compatible node. Extracts key claims from the document (reusing
    literature-review extraction as the claim source) and cross-checks each
    against state['all_findings'] from the web-research path.
    """
    all_findings = state.get("all_findings", [])

    # Reuse literature_review if already run this session, else compute it fresh
    lit_review = state.get("literature_review")
    if lit_review is None:
        state = literature_review_node(state, vectorstore=vectorstore)
        lit_review = state["literature_review"]

    # Treat each extracted field's value as one claim to check
    claims = [
        data["value"] for data in lit_review.values()
        if data["value"] and "not found" not in data["value"].lower()
    ]

    results = [_check_claim_against_web(claim, all_findings) for claim in claims]

    contradictions = [r for r in results if r["verdict"] == "contradicts"]

    state["contradiction_check"] = {
        "all_results": results,
        "contradictions": contradictions,
    }
    return state






if __name__ == "__main__":
    test_state = {"doc_question": "Who is responsible for maintaining the property?"}
    result = rag_agent_node(test_state)
    print("Answer:", result["doc_answer"])
    print("Cited pages:", result["doc_citations"])