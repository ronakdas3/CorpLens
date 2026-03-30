from sentence_transformers import CrossEncoder


_reranker = None


def get_reranker():

    global _reranker

    if _reranker is None:
        print("Loading reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return _reranker


def rerank_chunks(query, chunks):

    reranker = get_reranker()

    pairs = [(query, chunk) for chunk in chunks]

    scores = reranker.predict(pairs)

    scored_chunks = list(zip(chunks, scores))

    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    reranked = [chunk for chunk, score in scored_chunks]

    return reranked