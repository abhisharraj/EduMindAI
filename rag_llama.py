from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import requests
import faiss
import numpy as np


# ======================
# STEP 1: Read PDF
# ======================

reader = PdfReader("uploads/notes.pdf")

text = ""

for page in reader.pages:

    page_text = page.extract_text()

    if page_text:

        text += page_text


# ======================
# STEP 2: Create Chunks
# ======================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)


# ======================
# STEP 3: Create Embeddings
# ======================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(chunks)


# ======================
# STEP 4: Create FAISS Index
# ======================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))


# ======================
# STEP 5: Ask User Question
# ======================

query = input("Ask your question: ")


# ======================
# STEP 6: Search PDF
# ======================

query_embedding = embedding_model.encode([query])

distances, indices = index.search(
    np.array(query_embedding),
    k=3
)

context = ""

for idx in indices[0]:
    context += chunks[idx] + "\n"


# ======================
# STEP 7: Get Best Context
# ======================

print("\nRetrieved Context:\n")

print(context)

print("\n" + "=" * 50 + "\n")


# ======================
# STEP 8: Build Prompt
# ======================

prompt = f"""
You are a study assistant for Indian college students.

Use ONLY the context below.

Context:
{context}

Question:
{query}

Instructions:

- Explain in easy Hinglish.
- Use simple words.
- Keep technical terms in English.
- Do NOT use difficult Hindi words.
- Explain in 4-6 bullet points.
- Imagine you are teaching a beginner student.
- Do not repeat information.
- If information is unavailable, say 'Yeh information PDF mein available nahi hai.'

Answer:
"""


# ======================
# STEP 9: Ask Llama via REST API
# ======================

API_URL = "http://172.17.0.1:11434/api/generate"

payload = {
    "model": "llama3.2:1b",
    "prompt": prompt,
    "stream": False,
    "options": {
        "num_ctx": 1024,
        "num_gpu": 1,
        "use_mmap": True
    }
}

try:
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        answer = response.json()["response"]
    else:
        answer = f"Error {response.status_code}: {response.text}"

except requests.exceptions.RequestException as e:
    answer = f"Connection Error: {e}"


# ======================
# STEP 10: Print Answer
# ======================

print("\nAnswer:\n")
print(answer)