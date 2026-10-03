"""
main_rag.py
End-to-end runner for the RAG document path (Phase 3).
Ingests a PDF once, then answers repeated questions against it.
"""

import os
from ingest import load_and_chunk, build_faiss_index
from agents.rag_agent import rag_agent_node
from ingest import load_and_chunk, build_faiss_index, load_faiss_index
from agents.rag_agent import rag_agent_node, literature_review_node , contradiction_check_node


def ingest_document(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No file found at: {pdf_path}")

    print(f"Ingesting: {pdf_path}")
    structured_chunks, lc_docs = load_and_chunk(pdf_path)
    print(f"  -> {len(structured_chunks)} chunks created.")

    build_faiss_index(lc_docs)
    print("  -> FAISS index built and saved to ./faiss_index/\n")

    return structured_chunks


def ask_loop():
    vectorstore = load_faiss_index()
    print("Document loaded. Commands: ask a question, 'lit-review' for structured extraction, 'exit' to quit.\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() == "lit-review":
            result = literature_review_node({}, vectorstore=vectorstore)
            print("\n--- Literature Review Extraction ---")
            for field, data in result["literature_review"].items():
                print(f"\n{field.upper()}:")
                print(f"  {data['value']}")
                if data["citations"]:
                    pages = ", ".join(str(p) for p in data["citations"])
                    print(f"  (Source: page(s) {pages})")
            print()
            continue
        
# inside ask_loop, add another branch:
        if question.lower() == "check-contradictions":
            # NOTE: all_findings must come from a prior web-research run (Phase 1/2).
            # For standalone RAG testing, we fake a small sample set here.
            sample_web_findings = [
                {"title": "NeurIPS Transformer Follow-up", "key_point": "Later work showed the base Transformer's WMT14 En-De BLEU score as 27.3, not 28.4, due to differing evaluation scripts.", "url": "https://example.com/followup"},
                {"title": "MT Survey 2023", "key_point": "The Transformer architecture uses only self-attention, no recurrence or convolution.", "url": "https://example.com/survey"},
            ]
            state = {"all_findings": sample_web_findings}
            result = contradiction_check_node(state, vectorstore=vectorstore)
            print("\n--- Contradiction Check ---")
            for r in result["contradiction_check"]["all_results"]:
                print(f"\nClaim: {r['claim'][:100]}...")
                print(f"  Verdict: {r['verdict']}")
                print(f"  {r['explanation']}")
            print()
            continue
        if not question:
            continue

        state = {"doc_question": question}
        result = rag_agent_node(state, vectorstore=vectorstore)

        print(f"\nA: {result['doc_answer']}")
        if result["doc_citations"]:
            pages = ", ".join(str(p) for p in result["doc_citations"])
            print(f"   (Source: page(s) {pages})\n")
        else:
            print("   (No specific page citation — model may lack sufficient context)\n")

if __name__ == "__main__":
    pdf_path = input("Path to PDF to ingest: ").strip().strip('"')
    ingest_document(pdf_path)
    ask_loop()