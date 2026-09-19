from typing import TypedDict,List,Optional

class ResearchState(TypedDict):
    topic: str                      # the user's research query
    search_results: List[dict]      # raw results from Tavily (this round)
    all_findings: List[dict]        # accumulated verified findings across rounds
    verification_status: str        # "pending" | "passed" | "failed"
    verification_reason: Optional[str]  # why it failed, if it did
    retry_count: int                # how many re-search attempts so far
    max_retries: int                # cap, so we don't loop forever
    summary: list          # NEW — bullet-point key insights from Summarization Agent
    final_report: str      # NEW — compiled report from Report Generation Agent
    # --- RAG / Document path (Phase 3) ---
    document_path: str          # local path to the uploaded PDF
    document_chunks: list        # list of dicts: {"text": ..., "page": ..., "chunk_id": ...}
    doc_question: str            # user's question about the document
    doc_answer: str               # generated answer
    doc_citations: list           # list of page numbers / chunk_ids actually used in the answer