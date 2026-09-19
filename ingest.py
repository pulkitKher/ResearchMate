"""
ingest.py
Loads a PDF, splits it into overlapping chunks with page + chunk_id metadata,
embeds them locally (MiniLM), and builds/persists a FAISS index.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_and_chunk(pdf_path: str):
    """Load a PDF, split into chunks, and tag each chunk with page + chunk_id
    directly in its metadata (so retrieval never needs a separate lookup)."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # one Document per page, page.metadata["page"] = 0-based index

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(pages)

    structured_chunks = []
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1  # 1-based for humans

        structured_chunks.append({
            "chunk_id": i,
            "page": chunk.metadata["page"],
            "text": chunk.page_content,
        })

    return structured_chunks, chunks  # dicts for state, langchain Documents (with metadata) for indexing


def build_faiss_index(chunks_lc_docs, persist_dir: str = "faiss_index"):
    """Embed chunks with a local model and persist a FAISS index to disk.
    metadata (chunk_id, page) travels with each Document automatically."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks_lc_docs, embeddings)
    vectorstore.save_local(persist_dir)
    return vectorstore


def load_faiss_index(persist_dir: str = "faiss_index"):
    """Load a previously persisted FAISS index."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    test_pdf = "sample.pdf"
    structured, lc_docs = load_and_chunk(test_pdf)
    print(f"Chunked into {len(structured)} pieces. Example:")
    print(structured[0])

    vs = build_faiss_index(lc_docs)
    print("FAISS index built and saved to ./faiss_index/")