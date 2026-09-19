"""
main_rag.py
End-to-end runner for the RAG document path (Phase 3).
Ingests a PDF once, then answers repeated questions against it.
"""

import os
from ingest import load_and_chunk, build_faiss_index
from agents.rag_agent import rag_agent_node


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
    print("Document loaded. Ask questions (type 'exit' to quit).\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        state = {"doc_question": question}
        result = rag_agent_node(state)

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