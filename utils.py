import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def process_pdf(pdf_path):
    # Read PDF
    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        page_text = page.get_text()
        if page_text.strip():
            text += page_text + "\n"

    total_pages = len(doc)

    if not text.strip():
        raise ValueError("No text could be extracted from the PDF.")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    if len(chunks) == 0:
        raise ValueError("No chunks were created from the PDF.")

    # Embeddings
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    ).astype(np.float32)

    print("Text length:", len(text))
    print("Chunks:", len(chunks))
    print("Embeddings shape:", embeddings.shape)

    if len(embeddings.shape) != 2:
        raise ValueError(f"Invalid embedding shape: {embeddings.shape}")

    # FAISS
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return {
        "index": index,
        "chunks": chunks,
        "pages": total_pages,
        "chunk_count": len(chunks)
    }