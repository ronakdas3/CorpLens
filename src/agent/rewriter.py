import re
from src.models.llm_interface import generate_simple


def evaluate_retrieval_relevance(query: str, chunks: list[str]) -> bool:
    """
    Heuristic check to determine if retrieved chunks contain relevant terms from the query.
    Returns True if at least one chunk has significant term overlap.
    """
    if not chunks:
        return False

    stopwords = {"what", "who", "where", "when", "how", "why", "is", "are", "the", "a", "an", "in", "of", "to", "for", "and", "or", "did", "does"}
    query_words = set(re.findall(r"\w+", query.lower())) - stopwords

    if not query_words:
        return True

    for chunk in chunks:
        chunk_words = set(re.findall(r"\w+", chunk.lower()))
        overlap = query_words & chunk_words
        # If at least half of the key terms or 2+ key terms are present
        if len(overlap) >= min(2, len(query_words)):
            return True

    return False


def rewrite_query(query: str, tokenizer, model) -> str:
    """
    Expands or reformulates a query to improve keyword and semantic search recall.
    """
    prompt = f"""Rewrite this question into a search engine query with relevant keywords and synonyms.
Original: "{query}"
Expanded Query:"""

    rewritten = generate_simple(prompt, tokenizer, model, max_tokens=40)
    
    # Strip any extra text or quotes
    rewritten = re.sub(r'^(Expanded Query:|Query:|"|\')', '', rewritten).strip(' ".\'\n')

    # Fallback to original if rewriting was degenerate
    if not rewritten or len(rewritten) < 3:
        return query

    return rewritten
