from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Read PDF
reader = PdfReader("uploads/notes.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print("Chunks Created:", len(chunks))

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("FAISS Index Created Successfully")
print("Total Vectors:", index.ntotal)

# -----------------------
# SAVE INDEX
# -----------------------
faiss.write_index(index, "faiss_index.index")

import pickle

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index saved!")
print("Chunks saved!")