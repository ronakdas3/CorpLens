"""
LangGraph Workflow for CorpLens Enterprise RAG System.
Implements a stateful, cyclic agent with policy routing, corrective retrieval (CRAG), and multi-document compliance reasoning.
"""

from typing import TypedDict, List, Dict, Any, Optional
import os
import faiss
import numpy as np

from langgraph.graph import StateGraph, START, END

from src.agent.router import route_query
from src.agent.rewriter import evaluate_retrieval_relevance, rewrite_query
from src.retrieval.reranker import rerank_chunks
from src.models.llm_interface import generate_answer


# -------------------- AGENT STATE --------------------
class CorpLensState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[str]
    rewritten_query: Optional[str]
    retry_count: int
    answer: str
    steps: List[Dict[str, str]]
    # Dynamic runtime references (injected via runner)
    tokenizer: Any
    llm_model: Any
    embedding_model: Any
    retriever: Any
    chunks: List[str]
    top_k: int
    rerank_k: int
    max_tokens: int


# -------------------- GRAPH NODES --------------------

def route_node(state: CorpLensState) -> Dict[str, Any]:
    """Node: Classifies corporate query intent."""
    query = state["query"]
    tokenizer = state.get("tokenizer")
    llm_model = state.get("llm_model")

    routing = route_query(query, tokenizer, llm_model)
    intent = routing["intent"]

    step = {
        "node": "Route Intent",
        "description": f"Router classified request as '{intent}' ({routing['reasoning']})"
    }
    return {
        "intent": intent,
        "steps": state.get("steps", []) + [step]
    }


def greeting_node(state: CorpLensState) -> Dict[str, Any]:
    """Node: Handles conversational greetings without document retrieval."""
    greeting = (
        "Hello, I am **CorpLens**, your Enterprise AI Knowledge & Compliance Assistant powered by LangGraph. "
        "I can answer questions regarding corporate policies, employee benefits, information security guidelines, "
        "and procurement rules with verified source citations. What policy or procedure would you like to explore?"
    )
    step = {
        "node": "Greeting Agent",
        "description": "Provided conversational greeting without triggering document retrieval."
    }
    return {
        "answer": greeting,
        "retrieved_chunks": [],
        "steps": state.get("steps", []) + [step]
    }


def summarize_node(state: CorpLensState) -> Dict[str, Any]:
    """Node: Fetches the foundational policy scope and overview sections for summarization."""
    all_chunks = state.get("chunks", [])
    summary_chunks = all_chunks[:4] if len(all_chunks) >= 4 else all_chunks

    step = {
        "node": "Policy Summarizer",
        "description": f"Loaded top {len(summary_chunks)} introductory/scope passages for executive summary."
    }
    return {
        "retrieved_chunks": summary_chunks,
        "steps": state.get("steps", []) + [step]
    }


def compare_node(state: CorpLensState) -> Dict[str, Any]:
    """Node: Retrieves and pools context from multiple corporate policy documents."""
    query = state["query"]
    embedding_model = state.get("embedding_model")
    combined_chunks = []

    policy_sources = [
        ("Employee Handbook", "data/corporate_docs/employee_handbook"),
        ("InfoSec Policy", "data/corporate_docs/infosec_policy"),
        ("Procurement Policy", "data/corporate_docs/procurement_policy"),
        ("Engineering Standards", "data/corporate_docs/engineering_standards")
    ]

    for doc_name, doc_dir in policy_sources:
        idx_path = os.path.join(doc_dir, "faiss.index")
        chk_path = os.path.join(doc_dir, "chunks.npy")
        if os.path.exists(idx_path) and os.path.exists(chk_path):
            index = faiss.read_index(idx_path)
            chunks = np.load(chk_path, allow_pickle=True).tolist()
            q_emb = embedding_model.encode([query])
            _, ids = index.search(np.array(q_emb), 2)
            for idx in ids[0]:
                combined_chunks.append(f"[{doc_name}] {chunks[idx]}")

    reranked = rerank_chunks(query, combined_chunks)[:4] if combined_chunks else []
    step = {
        "node": "Cross-Policy Comparator",
        "description": f"Retrieved and merged {len(reranked)} cross-document passages across corporate policies."
    }
    return {
        "retrieved_chunks": reranked,
        "steps": state.get("steps", []) + [step]
    }


def retrieve_node(state: CorpLensState) -> Dict[str, Any]:
    """Node: Hybrid search (FAISS + BM25) and neural reranking over active policy."""
    search_query = state.get("rewritten_query") or state["query"]
    retriever = state["retriever"]
    top_k = state.get("top_k", 5)
    rerank_k = state.get("rerank_k", 3)

    docs = retriever.hybrid_search(search_query, k=top_k)
    candidates = [d["text"] for d in docs]
    reranked = rerank_chunks(search_query, candidates)[:rerank_k]

    step = {
        "node": "Hybrid Retrieve & Rerank",
        "description": f"Retrieved top candidates for '{search_query}'. Selected top {len(reranked)} reranked passages."
    }
    return {
        "retrieved_chunks": reranked,
        "steps": state.get("steps", []) + [step]
    }


def grade_relevance_node(state: CorpLensState) -> Dict[str, Any]:
    """Node: Corrective RAG evaluator to assess policy passage relevance."""
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    is_relevant = evaluate_retrieval_relevance(query, chunks)

    step = {
        "node": "Policy Evaluator (CRAG)",
        "description": f"Relevance assessment: {'HIGH CONFIDENCE' if is_relevant else 'LOW RELEVANCE - TRIGGER REWRITE'}."
    }
    return {
        "steps": state.get("steps", []) + [step]
    }


def rewrite_node(state: CorpLensState) -> Dict[str, Any]:
    """Node: Reformulates query using corporate search terms when relevance is insufficient."""
    query = state["query"]
    tokenizer = state.get("tokenizer")
    llm_model = state.get("llm_model")

    new_query = rewrite_query(query, tokenizer, llm_model)
    step = {
        "node": "Query Reformulator",
        "description": f"Reformulated query: '{query}' -> '{new_query}' for higher recall."
    }
    return {
        "rewritten_query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "steps": state.get("steps", []) + [step]
    }


def generate_node(state: CorpLensState) -> Dict[str, Any]:
    """Node: Synthesizes final corporate policy answer with verified citations."""
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    tokenizer = state["tokenizer"]
    llm_model = state["llm_model"]
    max_tokens = state.get("max_tokens", 300)

    answer = generate_answer(query, chunks, tokenizer, llm_model, max_tokens=max_tokens)
    step = {
        "node": "LLM Policy Synthesis",
        "description": f"Generated compliance answer backed by {len(chunks)} verified policy citations."
    }
    return {
        "answer": answer,
        "steps": state.get("steps", []) + [step]
    }


# -------------------- CONDITIONAL ROUTING --------------------

def route_decision(state: CorpLensState) -> str:
    intent = state.get("intent", "DOCUMENT_QUERY")
    if intent == "GREETING":
        return "greeting"
    elif intent == "SUMMARIZE":
        return "summarize"
    elif intent == "COMPARE":
        return "compare"
    return "retrieve"


def grade_decision(state: CorpLensState) -> str:
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    retries = state.get("retry_count", 0)

    # If relevant or already retried once, proceed to generate
    if evaluate_retrieval_relevance(query, chunks) or retries >= 1:
        return "generate"
    return "rewrite"


# -------------------- COMPILE LANGGRAPH WORKFLOW --------------------

def build_corplens_graph():
    """
    Constructs and compiles the CorpLens StateGraph.
    """
    workflow = StateGraph(CorpLensState)

    # Add Nodes
    workflow.add_node("route_node", route_node)
    workflow.add_node("greeting_node", greeting_node)
    workflow.add_node("summarize_node", summarize_node)
    workflow.add_node("compare_node", compare_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("grade_node", grade_relevance_node)
    workflow.add_node("rewrite_node", rewrite_node)
    workflow.add_node("generate_node", generate_node)

    # Add Edges
    workflow.add_edge(START, "route_node")

    workflow.add_conditional_edges(
        "route_node",
        route_decision,
        {
            "greeting": "greeting_node",
            "summarize": "summarize_node",
            "compare": "compare_node",
            "retrieve": "retrieve_node"
        }
    )

    workflow.add_edge("greeting_node", END)
    workflow.add_edge("summarize_node", "generate_node")
    workflow.add_edge("compare_node", "generate_node")

    # Retrieval -> Grading -> (Rewrite -> Retrieval) OR (Generate -> END)
    workflow.add_edge("retrieve_node", "grade_node")

    workflow.add_conditional_edges(
        "grade_node",
        grade_decision,
        {
            "generate": "generate_node",
            "rewrite": "rewrite_node"
        }
    )

    workflow.add_edge("rewrite_node", "retrieve_node")
    workflow.add_edge("generate_node", END)

    return workflow.compile()


# Singleton compiled graph
_app_graph = None

def get_corplens_graph():
    global _app_graph
    if _app_graph is None:
        _app_graph = build_corplens_graph()
    return _app_graph
