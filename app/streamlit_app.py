import sys
import os
# Fix import path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import streamlit as st
import faiss
import numpy as np

from src.models.model_manager import get_embedding_model, get_llm
from src.retrieval.vector_store import search_index
# from src.retrieval.reranker import rerank_chunks
from src.models.llm_interface import generate_answer
from src.indexing.build_index import build_index_for_paper
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="ResearchLens",
    page_icon="📄",
    layout="wide"
)

st.title("📄 ResearchLens")
st.markdown("Chat with research papers using AI")


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider("FAISS Top-K", 3, 10, 5)
    rerank_k = st.slider("Final Chunks", 1, 5, 3)
    max_tokens = st.slider("Answer Length", 100, 600, 300)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.write(
        "RAG Pipeline:\n"
        "- FAISS retrieval\n"
        "- Cross-encoder reranking\n"
        "- LLM generation"
    )


# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    embedding_model = get_embedding_model()
    tokenizer, llm_model = get_llm()
    return embedding_model, tokenizer, llm_model


embedding_model, tokenizer, llm_model = load_models()


# -------------------- SOURCE SELECTOR --------------------
st.subheader("📂 Choose Data Source")

# Upload first
uploaded_file = st.file_uploader("📄 Upload your own PDF", type="pdf")

# Radio selector
if uploaded_file:
    source_option = st.radio(
        "Select source",
        ["Uploaded PDF","Default Papers"]
    )
else:
    source_option = "Default Papers"

# -------------------- DEFAULT PAPERS --------------------
paper_option = None

if source_option == "Default Papers":
    st.subheader("📚 Choose a Paper")

    paper_option = st.selectbox(
        "Select a research paper",
        ["Transformer", "ResNet"]
    )

    if paper_option == "Transformer":
        st.caption("Attention Is All You Need (Vaswani et al., 2017)")
    elif paper_option == "ResNet":
        st.caption("Deep Residual Learning for Image Recognition (He et al., 2015)")


# -------------------- PATH MAPPING --------------------
def get_paper_paths(option):
    if option == "Transformer":
        return (
            "data/papers/transformer/faiss.index",
            "data/papers/transformer/chunks.npy"
        )
    elif option == "ResNet":
        return (
            "data/papers/resnet/faiss.index",
            "data/papers/resnet/chunks.npy"
        )


# -------------------- LOAD DATA --------------------
@st.cache_resource
def load_data(index_path, chunks_path):
    index = faiss.read_index(index_path)
    chunks = np.load(chunks_path, allow_pickle=True).tolist()
    return index, chunks


# -------------------- PROCESS UPLOADED PDF --------------------
@st.cache_resource
def process_uploaded_pdf(upload_bytes):
    upload_path = "data/temp_uploaded.pdf"
    save_dir = "data/temp_index"

    with open(upload_path, "wb") as f:
        f.write(upload_bytes)

    build_index_for_paper(upload_path, save_dir)

    return f"{save_dir}/faiss.index", f"{save_dir}/chunks.npy"


uploaded_index = None
uploaded_chunks = None

if uploaded_file:
    st.info("Processing uploaded PDF...")

    index_path, chunks_path = process_uploaded_pdf(uploaded_file.read())
    uploaded_index, uploaded_chunks = load_data(index_path, chunks_path)

    st.success("PDF indexed successfully!")


# -------------------- SELECT ACTIVE DATA --------------------
if source_option == "Uploaded PDF" and uploaded_index is not None:
    index, chunks = uploaded_index, uploaded_chunks
    st.caption("Using uploaded document")

elif source_option == "Uploaded PDF":
    st.warning("Please upload a PDF first.")
    st.stop()

else:
    index_path, chunks_path = get_paper_paths(paper_option)
    index, chunks = load_data(index_path, chunks_path)
    st.caption(f"Using: {paper_option}")


# -------------------- RESET CHAT ON SOURCE CHANGE --------------------
if "last_source" not in st.session_state:
    st.session_state.last_source = source_option

if st.session_state.last_source != source_option:
    st.session_state.messages = []
    st.session_state.last_source = source_option


# -------------------- CHAT STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------- DISPLAY CHAT --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])



# -------------------- INPUT --------------------
user_input = st.chat_input("Ask a question...")


# -------------------- MAIN PIPELINE --------------------
if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking... 🤔"):

            # Convert chunks → documents (once per run ideally)
            documents = [
                {
                    "text": chunk,
                    "doc_id": f"doc_{i}",
                    "chunk_id": i
                }
                for i, chunk in enumerate(chunks)
            ]

            # Initialize retriever (you can cache this later)
            bm25 = BM25Retriever(documents)
            retriever = HybridRetriever(index, bm25)

            # Hybrid search
            docs = retriever.hybrid_search(user_input, k=top_k)

            final_chunks = [doc["text"] for doc in docs[:rerank_k]]

            # Generate answer
            answer = generate_answer(
                user_input,
                final_chunks,
                tokenizer,
                llm_model,
                max_tokens=max_tokens
            )

        st.write(answer)

        st.caption(f"Retrieved {len(final_chunks)} relevant chunks")
        st.caption(f"📊 Based on top {len(final_chunks)} retrieved chunks")
        # confidence = len(final_chunks) / top_k
        # st.caption(f"📊 Confidence Score: {confidence:.2f}")  #this will always give 0.6 for k=3 

        st.markdown("### 📚 Sources")
        # for i, chunk in enumerate(final_chunks):
        #     with st.expander(f"Source {i+1}"):
        #         st.write(chunk)
        for i, chunk in enumerate(final_chunks):
            with st.expander(f"📄 Source [{i+1}]"):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(chunk[:500] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})


# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<center>RAG Pipeline</center>",
    unsafe_allow_html=True
)