import sys
import os

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Fix import path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import streamlit as st
import faiss
import numpy as np

from src.models.model_manager import get_embedding_model, get_llm
from src.retrieval.vector_store import search_index
from src.models.llm_interface import generate_answer
from src.indexing.build_index import build_index_for_paper
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.highlighting import highlight_relevant_sentence
from src.utils.roles import ROLE_ACCESS
from src.agent.langgraph_workflow import get_corplens_graph


# -------------------- RBAC FILTER --------------------
def filter_docs_by_role(docs, role):
    allowed = ROLE_ACCESS.get(role, [])
    if "all" in allowed:
        return docs
    return [
        doc for doc in docs
        if doc.get("department", "general") in allowed
    ]


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="CorpLens | Enterprise Policy & Compliance AI Assistant",
    layout="wide"
)

st.title("CorpLens")
st.caption("Enterprise AI Policy & Compliance Assistant powered by LangGraph, Hybrid Search & Neural Reranking")


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("Agent Controls")
    use_agent = st.toggle(
        "Enable LangGraph Agent",
        value=True,
        help="Enables stateful intent routing, Corrective RAG (CRAG) self-evaluation, and multi-document reasoning."
    )

    st.markdown("---")
    st.header("Retrieval Parameters")
    top_k = st.slider("FAISS Top-K", 3, 10, 5)
    rerank_k = st.slider("Final Passages (Reranked)", 1, 5, 3)
    max_tokens = st.slider("Answer Length", 100, 600, 300)

    st.markdown("---")
    st.header("Access Control (RBAC)")
    role = st.selectbox(
        "Select User Role",
        ["intern", "employee", "manager", "legal", "admin"]
    )

    st.markdown("---")
    st.markdown("### Architecture")
    st.write(
        "- **Agent Framework**: LangGraph\n"
        "- **Vector Search**: FAISS (L2)\n"
        "- **Lexical Search**: BM25Okapi\n"
        "- **Reranker**: Cross-Encoder MiniLM\n"
        "- **LLM**: FLAN-T5 / TinyLlama"
    )


# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    embedding_model = get_embedding_model()
    tokenizer, llm_model = get_llm()
    return embedding_model, tokenizer, llm_model


embedding_model, tokenizer, llm_model = load_models()


# -------------------- SOURCE SELECTOR --------------------
st.subheader("Corporate Knowledge Base")

uploaded_file = st.file_uploader("Upload Company Document (PDF)", type="pdf")

if uploaded_file:
    source_option = st.radio(
        "Select document source",
        ["Uploaded PDF", "Corporate Policy Repository"],
        horizontal=True
    )
else:
    source_option = "Corporate Policy Repository"

policy_option = None
doc_dept = "general"

if source_option == "Corporate Policy Repository":
    policy_option = st.selectbox(
        "Select an active corporate policy document",
        [
            "Employee Workplace Handbook",
            "Information Security & IT Policy",
            "Financial Procurement & Travel Governance",
            "Software Engineering & Architecture Standards"
        ]
    )

    if policy_option == "Employee Workplace Handbook":
        st.caption("Acme Global Code of Conduct, Hybrid Work, PTO & Health Benefits (Department: HR / General)")
        doc_dept = "general"
    elif policy_option == "Information Security & IT Policy":
        st.caption("Acme Global InfoSec, Data Classification, MFA, AI Rules & Key Management (Department: Legal / General)")
        doc_dept = "legal"
    elif policy_option == "Financial Procurement & Travel Governance":
        st.caption("Acme Global Purchasing Approvals, Travel Per Diems, Gifts & Anti-Bribery (Department: Finance)")
        doc_dept = "finance"
    elif policy_option == "Software Engineering & Architecture Standards":
        st.caption("Acme Engineering SDLC, PR SLAs, Incident Severity Levels & Deployment Policies (Department: Engineering)")
        doc_dept = "engineering"


# -------------------- PATH MAPPING --------------------
def get_policy_paths(option):
    if option == "Employee Workplace Handbook":
        return (
            "data/corporate_docs/employee_handbook/faiss.index",
            "data/corporate_docs/employee_handbook/chunks.npy"
        )
    elif option == "Information Security & IT Policy":
        return (
            "data/corporate_docs/infosec_policy/faiss.index",
            "data/corporate_docs/infosec_policy/chunks.npy"
        )
    elif option == "Financial Procurement & Travel Governance":
        return (
            "data/corporate_docs/procurement_policy/faiss.index",
            "data/corporate_docs/procurement_policy/chunks.npy"
        )
    elif option == "Software Engineering & Architecture Standards":
        return (
            "data/corporate_docs/engineering_standards/faiss.index",
            "data/corporate_docs/engineering_standards/chunks.npy"
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
    st.info("Processing & indexing uploaded document with FAISS...")
    index_path, chunks_path = process_uploaded_pdf(uploaded_file.read())
    uploaded_index, uploaded_chunks = load_data(index_path, chunks_path)
    st.success("Document indexed successfully.")


# -------------------- SELECT ACTIVE DATA --------------------
if source_option == "Uploaded PDF" and uploaded_index is not None:
    index, chunks = uploaded_index, uploaded_chunks
    st.caption("Active Document: Uploaded Company Document")
    doc_dept = "general"

elif source_option == "Uploaded PDF":
    st.warning("Please upload a PDF document first.")
    st.stop()

else:
    index_path, chunks_path = get_policy_paths(policy_option)
    index, chunks = load_data(index_path, chunks_path)
    st.caption(f"Active Document: {policy_option}")


# -------------------- RESET CHAT ON SOURCE CHANGE --------------------
if "last_source" not in st.session_state:
    st.session_state.last_source = source_option

if st.session_state.last_source != source_option:
    st.session_state.messages = []
    st.session_state.last_source = source_option


# -------------------- CHAT STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "trace" in msg and msg["trace"]:
            with st.expander("View LangGraph Execution Trace"):
                for step in msg["trace"]:
                    st.markdown(f"**[{step['node']}]**: {step['description']}")


# -------------------- USER INPUT --------------------
user_input = st.chat_input("Ask a question about corporate policies, procedures, or compliance...")


# -------------------- MAIN PIPELINE --------------------
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        # Wrap document chunks with metadata and department tag for RBAC
        documents = [
            {
                "text": chunk,
                "doc_id": f"doc_{i}",
                "chunk_id": i,
                "department": doc_dept
            }
            for i, chunk in enumerate(chunks)
        ]

        # Pre-check RBAC permissions for the active document
        allowed_depts = ROLE_ACCESS.get(role, [])
        if "all" not in allowed_depts and doc_dept not in allowed_depts:
            denied_msg = f"**Access Denied**: Your role (`{role}`) does not have authorization to view `{doc_dept.upper()}` documents."
            st.warning(denied_msg)
            st.session_state.messages.append({"role": "assistant", "content": denied_msg})
            st.stop()

        # Initialize hybrid retriever
        bm25 = BM25Retriever(documents)
        retriever = HybridRetriever(index, bm25)

        answer_text = ""
        final_chunks = []
        execution_trace = []

        if use_agent:
            # -------------------- LANGGRAPH AGENT WORKFLOW --------------------
            with st.spinner("LangGraph Agent executing workflow..."):
                initial_state = {
                    "query": user_input,
                    "intent": "",
                    "retrieved_chunks": [],
                    "rewritten_query": None,
                    "retry_count": 0,
                    "answer": "",
                    "steps": [],
                    "tokenizer": tokenizer,
                    "llm_model": llm_model,
                    "embedding_model": embedding_model,
                    "retriever": retriever,
                    "chunks": chunks,
                    "top_k": top_k,
                    "rerank_k": rerank_k,
                    "max_tokens": max_tokens
                }

                graph = get_corplens_graph()
                agent_output = graph.invoke(initial_state)

                answer_text = agent_output.get("answer", "")
                final_chunks = agent_output.get("retrieved_chunks", [])
                execution_trace = agent_output.get("steps", [])

            # Display interactive LangGraph execution trace
            with st.status("LangGraph StateGraph Execution", expanded=False) as status:
                for step in execution_trace:
                    st.markdown(f"**[{step['node']}]**: {step['description']}")
                status.update(label="Policy reasoning completed", state="complete")

            st.write(answer_text)

        else:
            # -------------------- DIRECT RAG --------------------
            with st.spinner("Searching policy guidelines..."):
                docs = retriever.hybrid_search(user_input, k=top_k)
                docs = filter_docs_by_role(docs, role)

                if len(docs) == 0:
                    st.warning("No accessible policy passages found for your role.")
                    st.stop()

                final_chunks = [doc["text"] for doc in docs[:rerank_k]]
                answer_text = generate_answer(
                    user_input,
                    final_chunks,
                    tokenizer,
                    llm_model,
                    max_tokens=max_tokens
                )

            st.write(answer_text)

        # -------------------- SOURCE CITATIONS & HIGHLIGHTING --------------------
        if final_chunks:
            st.caption(f"Grounded on {len(final_chunks)} verified policy sections")
            st.markdown("### Policy Citations & Evidence")

            for i, chunk in enumerate(final_chunks):
                with st.expander(f"Policy Source [{i+1}]"):
                    highlight = highlight_relevant_sentence(chunk, user_input)
                    st.markdown("**Key Ruling / Evidence:**")
                    st.markdown(f"**{highlight}**")
                    st.markdown("---")
                    st.markdown("**Full Policy Text:**")
                    st.write(chunk[:500] + ("..." if len(chunk) > 500 else ""))

    # Save to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "trace": execution_trace
    })


# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<center><small>CorpLens | Enterprise AI Policy & Compliance Assistant | LangGraph StateGraph | FAISS + BM25</small></center>",
    unsafe_allow_html=True
)
