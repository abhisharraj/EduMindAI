import fitz  # PyMuPDF
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load embedding model only once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def process_pdf(pdf_path):
    """
    Reads PDF, creates chunks, embeddings and FAISS index.
    """

    # -------------------------
    # Read PDF
    # -------------------------

    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    total_pages = len(doc)

    # -------------------------
    # Chunking
    # -------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    # -------------------------
    # Embeddings
    # -------------------------

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    # -------------------------
    # FAISS
    # -------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    # -------------------------
    # Return everything
    # -------------------------

    return {
        "index": index,
        "chunks": chunks,
        "pages": total_pages,
        "chunk_count": len(chunks)
    }