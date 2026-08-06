import tempfile
import requests
import streamlit as st

from sentence_transformers import SentenceTransformer
from utils import process_pdf


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="EduMindAI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# EMBEDDING MODEL
# ----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedder = load_embedder()

# ----------------------------
# CHAT HISTORY
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown(
    """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
    background: linear-gradient(135deg,#eef5ff,#ffffff);
}

.main-title{
    text-align:center;
    font-size:52px;
    font-weight:800;
    color:#1565C0;
}

.sub-title{
    text-align:center;
    color:#666666;
    font-size:20px;
    margin-bottom:30px;
}

.stButton>button{
    width:100%;
    height:48px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
    background:#1565C0;
    color:white;
}

.stButton>button:hover{
    background:#0D47A1;
}
            
.answer-card{
    background:#eef6ff;
    border-left:6px solid #1565C0;
    padding:20px;
    border-radius:12px;
    margin-top:10px;
    color:#222;
    box-shadow:0 2px 8px rgba(0,0,0,.08);
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.title("📚 EduMindAI")
    st.success("Offline AI Assistant")

    st.markdown("---")
    st.subheader("Features")
    st.markdown(
        """
- 🤖 Llama 3.1 (Offline API)
- 📄 PDF Question Answering
- 🔍 FAISS Semantic Search
- 🧠 RAG Architecture
- 🇮🇳 Hinglish Answers
"""
    )

    st.markdown("---")
    st.info("Made by team MMMUT")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------
# HEADER
# ----------------------------
st.markdown(
    "<div class='main-title'>📚 EduMindAI</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='sub-title'>Offline Hinglish Study Assistant using Llama 3.1 + RAG</div>",
    unsafe_allow_html=True,
)

# ----------------------------
# LAYOUT
# ----------------------------
left, right = st.columns([1, 2])

with left:
    st.markdown("### 📄 Upload PDF")

    uploaded_pdf = st.file_uploader(
        "Choose your study notes",
        type=["pdf"],
    )

    if uploaded_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name

        with st.spinner("📖 Processing PDF..."):
            data = process_pdf(pdf_path)

            st.session_state["index"] = data["index"]
            st.session_state["chunks"] = data["chunks"]
            st.session_state["pages"] = data["pages"]
            st.session_state["chunk_count"] = data["chunk_count"]
            st.session_state["pdf_name"] = uploaded_pdf.name

        st.success("✅ PDF Processed Successfully!")

        st.markdown("---")
        st.markdown("### 📊 PDF Statistics")

        st.metric("📄 PDF", st.session_state["pdf_name"])
        st.metric("📑 Pages", st.session_state["pages"])
        st.metric("📚 Chunks", st.session_state["chunk_count"])
        st.metric("🤖 Model", "Llama 3.2 : 1B")
    else:
        st.info("Upload a PDF to begin.")

with right:
    st.markdown("### 💬 Chat with EduMindAI")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything from your uploaded PDF...")


# ----------------------------
# REST API HELPER
# ----------------------------
def ask_llama(prompt: str) -> str:
    API_URL = "http://172.17.0.1:11434/api/generate"

    payload = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(API_URL, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


# ----------------------------
# QUESTION ANSWERING
# ----------------------------
if question:
    if "index" not in st.session_state:
        st.warning("⚠️ Please upload a PDF first.")
        st.stop()

    if question.strip() == "":
        st.warning("⚠️ Please enter a question.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.spinner("🔍 Searching your notes..."):
        index = st.session_state["index"]
        chunks = st.session_state["chunks"]

        query_embedding = embedder.encode(
            [question],
            convert_to_numpy=True,
        ).astype("float32")

        _, I = index.search(query_embedding, k=3)

        context = ""
        for idx in I[0]:
            context += chunks[idx] + "\n\n"

    prompt = f"""
You are EduMindAI.

Answer ONLY using the PDF context below.

Rules:
- Answer in simple Hinglish.
- Keep technical words in English.
- Explain like a teacher.
- Use bullet points.
- Don't use outside knowledge.
- If answer is not present, reply exactly:

Ye information uploaded PDF mein available nahi hai.

PDF Context:

{context}

Question:

{question}

Answer:
"""

    with st.spinner("🤖 Generating Answer..."):
        try:
            answer = ask_llama(prompt)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to Llama API: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(
            f"""
            <div class="answer-card">
                <h4>📖 EduMindAI Answer</h4>
                {answer}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("📚 Retrieved Chunks"):
        for i, idx in enumerate(I[0]):
            st.markdown(f"### Chunk {i + 1}")
            st.info(chunks[idx])
            st.divider()

st.markdown("---")
st.markdown(
    """
<div style='text-align:center;color:gray;padding:10px'>
<b>EduMindAI</b><br>
Offline AI Study Assistant<br>
Powered by Llama 3.1 • FAISS • Sentence Transformers • Streamlit
</div>
""",
    unsafe_allow_html=True,
)