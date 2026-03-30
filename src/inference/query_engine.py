import numpy as np
import faiss
# from sentence_transformers import SentenceTransformer
from src.models.model_manager import get_embedding_model, get_llm
from src.retrieval.reranker import rerank_chunks

# from src.data.pdf_loader import extract_text_from_pdf
# from src.data.text_chunker import chunk_text
# from src.embeddings.embedding_generator import generate_embeddings
from src.retrieval.vector_store import build_vector_index, search_index

from src.models.llm_interface import load_llm, generate_answer

def retrieve_relevant_chunks(query, chunks, index, embedding_model, k=5):
    """
    Retrieve the most relevant document chunks for a query.
    """

    # query_embedding = model.encode([query])
    query_embedding = embedding_model.encode([query])

    distances, indices = search_index(index, np.array(query_embedding), k)

    results = [chunks[i] for i in indices[0]]

    return results


if __name__ == "__main__":

    # pdf_path = "data/raw/sample_paper.pdf"
    # text = extract_text_from_pdf(pdf_path)
    # chunks = chunk_text(text)
    # embeddings = generate_embeddings(chunks)
    # index = build_vector_index(embeddings)

    embedding_model = get_embedding_model()
    tokenizer, llm_model = get_llm()

    index = faiss.read_index("data/faiss.index")
    chunks = np.load("data/chunks.npy", allow_pickle=True).tolist()

    # model = SentenceTransformer("all-MiniLM-L6-v2")

    # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # query = "What is the main contribution of this paper?"

    # results = retrieve_relevant_chunks(query, chunks, index, model)

    # print("\nTop relevant chunks:\n")

    # for i, chunk in enumerate(results):
    #     print(f"\nResult {i+1}:\n")
    #     print(chunk[:500])

    

    # load LLM
    # generator = load_llm()
    # tokenizer, model = load_llm()

    query = "What is the main contribution of this paper?"
    # query = "Who are the Authors of this paper at the top of the page?"
    # query = "What is the transformer architecture?"
    # query = "what architecture does the paper propose?"
    # query = "what is attention"
    # query = "what is Scaled Dot-Product Attention"
    # query = "what GPUs were used"


    results = retrieve_relevant_chunks(query, chunks, index, embedding_model)

    results = rerank_chunks(query, results)

    results = results[:5]
    
    # answer = generate_answer(query, results, generator)
    answer = generate_answer(query, results, tokenizer, llm_model)

    print("\nThe question was\n")
    print(query)

    print("\nFinal Answer:\n")
    print(answer)

    print("\nSources:\n")
    for i, chunk in enumerate(results):
        print(f"[{i+1}] {chunk[:300]}\n")
