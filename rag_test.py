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

# Create chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

# FAISS
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# User Question
query = "What is supervised learning?"  #kya phuchana hai wahi batayega pura notes nahi dega

query_embedding = model.encode([query])

distances, indices = index.search(
    np.array(query_embedding),
    k=3
)

print("\nTop Relevant Chunks:\n")

for idx in indices[0]:
    print(chunks[idx])
    print("\n" + "="*50 + "\n")