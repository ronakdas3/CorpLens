import re
from src.models.llm_interface import generate_simple

GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening))\b",
    r"^who\s+are\s+you\b",
    r"^what\s+(can\s+you\s+do|is\s+your\s+name|are\s+your\s+capabilities)\b",
    r"^help\b"
]

SUMMARIZE_PATTERNS = [
    r"\b(summarize|summary|overview|tldr|tl;dr|abstract|give\s+me\s+an\s+overview|what\s+is\s+this\s+paper\s+about)\b"
]

COMPARE_PATTERNS = [
    r"\b(compare|comparison|versus|vs\.?|difference\s+between|differences\s+between)\b"
]


def route_query(query: str, tokenizer=None, model=None) -> dict:
    """
    Classify user query intent to decide the optimal execution path.
    Returns:
        dict: {"intent": "GREETING" | "SUMMARIZE" | "COMPARE" | "DOCUMENT_QUERY", "confidence": float, "reasoning": str}
    """
    clean_q = query.strip().lower()

    # 1. Fast Pattern Matching (0 ms latency)
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, clean_q):
            return {
                "intent": "GREETING",
                "confidence": 0.95,
                "reasoning": "Detected conversational greeting or identity query."
            }

    for pattern in COMPARE_PATTERNS:
        if re.search(pattern, clean_q):
            return {
                "intent": "COMPARE",
                "confidence": 0.90,
                "reasoning": "Detected comparative request between documents or concepts."
            }

    for pattern in SUMMARIZE_PATTERNS:
        if re.search(pattern, clean_q):
            return {
                "intent": "SUMMARIZE",
                "confidence": 0.90,
                "reasoning": "Detected request for high-level document summary or overview."
            }

    # 2. LLM fallback if model provided and query is ambiguous
    if tokenizer is not None and model is not None and len(clean_q.split()) <= 4:
        prompt = (
            f"Classify the following query into exactly ONE category:\n"
            f"[GREETING, SUMMARIZE, COMPARE, QUESTION]\n\n"
            f"Query: \"{query}\"\n"
            f"Category:"
        )
        prediction = generate_simple(prompt, tokenizer, model, max_tokens=10).upper()
        if "GREET" in prediction:
            return {"intent": "GREETING", "confidence": 0.8, "reasoning": "LLM classified as greeting."}
        elif "SUMMAR" in prediction:
            return {"intent": "SUMMARIZE", "confidence": 0.8, "reasoning": "LLM classified as summarization."}
        elif "COMPARE" in prediction:
            return {"intent": "COMPARE", "confidence": 0.8, "reasoning": "LLM classified as comparison."}

    return {
        "intent": "DOCUMENT_QUERY",
        "confidence": 0.85,
        "reasoning": "Standard factual question requiring document retrieval."
    }
