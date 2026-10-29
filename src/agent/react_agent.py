"""
Native ReAct (Reasoning + Acting) Agent for CorpLens.
Operates without LangChain / LangGraph using pure Python control loops.
"""

import os
import faiss
import numpy as np
from src.agent.router import route_query
from src.agent.rewriter import evaluate_retrieval_relevance, rewrite_query
from src.retrieval.reranker import rerank_chunks
from src.models.llm_interface import generate_answer, generate_answer_stream, generate_simple


class CorpLensAgent:
    def __init__(self, tokenizer, llm_model, embedding_model):
        self.tokenizer = tokenizer
        self.llm_model = llm_model
        self.embedding_model = embedding_model

    def run(self, query: str, context_state: dict, stream: bool = True):
        """
        Orchestrates agent reasoning, tool execution, and answer synthesis.
        
        Args:
            query (str): User question
            context_state (dict): Holds current index, chunks, retriever, role, etc.
            stream (bool): Whether to stream final generation
            
        Returns:
            dict: {
                "trace": list of reasoning steps,
                "answer": str (if stream=False) or generator (if stream=True),
                "chunks": list of retrieved context chunks,
                "intent": str
            }
        """
        trace = []

        # ---------------- 1. ROUTING STEP ----------------
        routing = route_query(query, self.tokenizer, self.llm_model)
        intent = routing["intent"]

        trace.append({
            "stage": "Intent Routing",
            "thought": routing["reasoning"],
            "action": f"Route to {intent}",
            "observation": f"Identified query type as '{intent}' (Confidence: {routing['confidence']})"
        })

        # ---------------- 2. HANDLE GREETINGS ----------------
        if intent == "GREETING":
            greeting_response = (
                "Hello, I am **CorpLens**, your Enterprise AI Knowledge Assistant. "
                "I can analyze company policies, explain workplace benefits, verify procurement rules, "
                "and cross-reference compliance handbooks with source citations. "
                "How can I help you today?"
            )
            trace.append({
                "stage": "Execution",
                "thought": "Providing direct conversational greeting without triggering retrieval.",
                "action": "direct_reply",
                "observation": "Greeting dispatched."
            })
            
            def stream_text(text):
                for word in text.split(" "):
                    yield word + " "

            return {
                "trace": trace,
                "answer": stream_text(greeting_response) if stream else greeting_response,
                "chunks": [],
                "intent": intent
            }

        # ---------------- 3. HANDLE SUMMARIZATION ----------------
        if intent == "SUMMARIZE":
            trace.append({
                "stage": "Tool Execution",
                "thought": "User requested an overview. Fetching the foundational policy scope sections (first 4 passages).",
                "action": "get_document_overview",
                "observation": "Extracted foundational sections of the document."
            })
            
            all_chunks = context_state.get("chunks", [])
            summary_chunks = all_chunks[:4] if len(all_chunks) >= 4 else all_chunks

            if stream:
                answer = generate_answer_stream(
                    "Provide a comprehensive summary of this corporate policy document based on the overview.",
                    summary_chunks,
                    self.tokenizer,
                    self.llm_model,
                    max_tokens=context_state.get("max_tokens", 350)
                )
            else:
                answer = generate_answer(
                    "Provide a comprehensive summary of this corporate policy document based on the overview.",
                    summary_chunks,
                    self.tokenizer,
                    self.llm_model,
                    max_tokens=context_state.get("max_tokens", 350)
                )

            return {
                "trace": trace,
                "answer": answer,
                "chunks": summary_chunks,
                "intent": intent
            }

        # ---------------- 4. HANDLE MULTI-POLICY COMPARISON ----------------
        if intent == "COMPARE":
            trace.append({
                "stage": "Multi-Document Tool",
                "thought": "Query involves comparison. Activating cross-document retrieval across corporate policies.",
                "action": "compare_documents",
                "observation": "Querying corporate policy stores simultaneously."
            })

            combined_chunks = []
            for doc_name, doc_dir in [
                ("Employee Handbook", "data/corporate_docs/employee_handbook"),
                ("InfoSec Policy", "data/corporate_docs/infosec_policy"),
                ("Procurement Policy", "data/corporate_docs/procurement_policy")
            ]:
                idx_path = os.path.join(doc_dir, "faiss.index")
                chk_path = os.path.join(doc_dir, "chunks.npy")
                if os.path.exists(idx_path) and os.path.exists(chk_path):
                    p_index = faiss.read_index(idx_path)
                    p_chunks = np.load(chk_path, allow_pickle=True).tolist()
                    q_emb = self.embedding_model.encode([query])
                    _, p_ids = p_index.search(np.array(q_emb), 2)
                    for idx in p_ids[0]:
                        combined_chunks.append(f"[{doc_name}] {p_chunks[idx]}")

            if combined_chunks:
                reranked = rerank_chunks(query, combined_chunks)[:4]
                trace.append({
                    "stage": "Synthesis",
                    "thought": f"Retrieved {len(reranked)} cross-document evidence chunks. Generating comparative analysis.",
                    "action": "generate_comparison",
                    "observation": "Context assembled from multiple policy documents."
                })
                
                gen_func = generate_answer_stream if stream else generate_answer
                answer = gen_func(
                    query,
                    reranked,
                    self.tokenizer,
                    self.llm_model,
                    max_tokens=context_state.get("max_tokens", 400)
                )
                return {
                    "trace": trace,
                    "answer": answer,
                    "chunks": reranked,
                    "intent": intent
                }

        # ---------------- 5. STANDARD DOCUMENT QUERY (CRAG + RERANKING) ----------------
        retriever = context_state.get("retriever")
        top_k = context_state.get("top_k", 5)
        rerank_k = context_state.get("rerank_k", 3)

        trace.append({
            "stage": "Retrieval Tool",
            "thought": "Executing hybrid dense (FAISS) and sparse (BM25) search for factual evidence.",
            "action": f"hybrid_search(k={top_k})",
            "observation": f"Retrieved top {top_k} candidate chunks."
        })

        docs = retriever.hybrid_search(query, k=top_k)
        candidate_texts = [d["text"] for d in docs]

        # Corrective RAG Evaluation
        is_relevant = evaluate_retrieval_relevance(query, candidate_texts)
        if not is_relevant:
            trace.append({
                "stage": "Corrective RAG",
                "thought": "Initial candidates show low term overlap. Formulating an expanded query with synonyms.",
                "action": "rewrite_query()",
                "observation": "Re-running retrieval with reformulated query."
            })
            rewritten_q = rewrite_query(query, self.tokenizer, self.llm_model)
            more_docs = retriever.hybrid_search(rewritten_q, k=top_k)
            # Merge unique candidates
            seen = set(candidate_texts)
            for d in more_docs:
                if d["text"] not in seen:
                    candidate_texts.append(d["text"])
                    seen.add(d["text"])

        # Neural Reranking
        trace.append({
            "stage": "Neural Reranker",
            "thought": "Cross-Encoder scoring query-context pairs to isolate top semantic chunks.",
            "action": f"rerank_chunks(final_k={rerank_k})",
            "observation": f"Filtered to top {rerank_k} most relevant passages."
        })

        final_chunks = rerank_chunks(query, candidate_texts)[:rerank_k]

        trace.append({
            "stage": "Generation",
            "thought": "Synthesizing factual answer with citation identifiers grounded in context.",
            "action": "generate_answer_stream" if stream else "generate_answer",
            "observation": "Generating streaming response."
        })

        gen_func = generate_answer_stream if stream else generate_answer
        answer = gen_func(
            query,
            final_chunks,
            self.tokenizer,
            self.llm_model,
            max_tokens=context_state.get("max_tokens", 300)
        )

        return {
            "trace": trace,
            "answer": answer,
            "chunks": final_chunks,
            "intent": intent
        }
