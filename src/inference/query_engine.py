import os
import numpy as np
import faiss

from src.models.model_manager import get_embedding_model, get_llm
from src.retrieval.reranker import rerank_chunks
from src.retrieval.vector_store import search_index
from src.models.llm_interface import generate_answer


def retrieve_relevant_chunks(query, chunks, index, embedding_model, k=5):
    """
    Retrieve the most relevant document chunks for a query.
    """
    query_embedding = embedding_model.encode([query])
    distances, indices = search_index(index, np.array(query_embedding), k)
    results = [chunks[i] for i in indices[0]]
    return results


if __name__ == "__main__":
    embedding_model = get_embedding_model()
    tokenizer, llm_model = get_llm()

    index_path = "data/corporate_docs/employee_handbook/faiss.index"
    chunks_path = "data/corporate_docs/employee_handbook/chunks.npy"

    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found. Run scripts/generate_corporate_docs.py first.")
        exit(1)

    index = faiss.read_index(index_path)
    chunks = np.load(chunks_path, allow_pickle=True).tolist()

    query = "What is the policy for remote work and home office stipend?"

    results = retrieve_relevant_chunks(query, chunks, index, embedding_model, k=5)
    results = rerank_chunks(query, results)
    results = results[:3]

    answer = generate_answer(query, results, tokenizer, llm_model)

    print("\n-------------------------------------------")
    print("Corporate Question:", query)
    print("-------------------------------------------")
    print("\nVerified Answer:\n", answer)
    print("\nPolicy Citations:")
    for i, chunk in enumerate(results):
        print(f"\n[{i+1}] {chunk[:250]}...")
