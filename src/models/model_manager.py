from sentence_transformers import SentenceTransformer
from src.models.llm_interface import load_llm


_embedding_model = None
_tokenizer = None
_llm_model = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return _embedding_model


def get_llm():
    global _tokenizer, _llm_model

    if _llm_model is None:
        print("Loading LLM...")
        _tokenizer, _llm_model = load_llm()

    return _tokenizer, _llm_model