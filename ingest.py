"""
ingest.py
Loads a PDF, splits it into overlapping chunks with page metadata,
embeds them locally (MiniLM), and builds/persists a FAISS index.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_and_chunk(pdf_path: str):
    """Load a PDF and split into chunks, preserving page numbers."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # one Document per page, page.metadata["page"] = page index (0-based)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(pages)

    # Normalize into our own chunk dict shape (page as 1-based, human-friendly)
    structured_chunks = []
    for i, chunk in enumerate(chunks):
        structured_chunks.append({
            "chunk_id": i,
            "page": chunk.metadata.get("page", 0) + 1,
            "text": chunk.page_content,
        })

    return structured_chunks, chunks  # return both: our dicts for state, langchain docs for indexing


def build_faiss_index(chunks_lc_docs, persist_dir: str = "faiss_index"):
    """Embed chunks with a local model and persist a FAISS index to disk."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks_lc_docs, embeddings)
    vectorstore.save_local(persist_dir)
    return vectorstore


def load_faiss_index(persist_dir: str = "faiss_index"):
    """Load a previously persisted FAISS index."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    # Quick manual test — run this file directly to sanity-check ingestion
    test_pdf = "sample.pdf"  # put a test PDF at project root with this name
    structured, lc_docs = load_and_chunk(test_pdf)
    print(f"Chunked into {len(structured)} pieces. Example:")
    print(structured[0])

    vs = build_faiss_index(lc_docs)
    print("FAISS index built and saved to ./faiss_index/")