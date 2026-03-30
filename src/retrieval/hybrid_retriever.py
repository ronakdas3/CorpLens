import numpy as np
from src.models.model_manager import get_embedding_model
from src.retrieval.vector_store import search_index


class HybridRetriever:
    def __init__(self, faiss_index, bm25_retriever):
        self.index = faiss_index
        self.bm25 = bm25_retriever
        self.embedding_model = get_embedding_model()

    def faiss_search(self, query, k=5):
        query_embedding = self.embedding_model.encode([query])
        distances, indices = search_index(
            self.index,
            np.array(query_embedding),
            k
        )

        results = []
        for i in indices[0]:
            results.append(self.bm25.documents[i])

        return results

    def hybrid_search(self, query, k=5, alpha=0.5):
        dense = self.faiss_search(query, k=k)
        sparse = self.bm25.search(query, k=k)

        scores = {}

        # Dense scores
        for rank, doc in enumerate(dense):
            key = doc["text"]
            scores[key] = scores.get(key, 0) + (1 - alpha) * (k - rank)

        # Sparse scores
        for rank, doc in enumerate(sparse):
            key = doc["text"]
            scores[key] = scores.get(key, 0) + alpha * (k - rank)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        text_to_doc = {doc["text"]: doc for doc in dense + sparse}

        return [text_to_doc[text] for text, _ in sorted_docs[:k]]