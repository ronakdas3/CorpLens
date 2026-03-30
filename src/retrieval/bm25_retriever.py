from rank_bm25 import BM25Okapi
import re


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, documents):
        """
        documents: list of dicts
        {
            "text": "...",
            "doc_id": "...",
            "chunk_id": ...
        }
        """
        self.documents = documents
        self.tokenized_corpus = [
            tokenize(doc["text"]) for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, k=5):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        return [self.documents[i] for i in ranked_indices]