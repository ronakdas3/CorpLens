import sys
import os

import faiss
import numpy as np

from src.data.pdf_loader import extract_text_from_pdf
from src.data.text_chunker import chunk_text
from src.embeddings.embedding_generator import generate_embeddings


def build_vector_index(embeddings):
    """
    Build FAISS vector index from embeddings.
    """

    dimension = embeddings.shape[1]
    

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index


def search_index(index, query_embedding, k=5):
    """
    Search the FAISS index.
    """

    distances, indices = index.search(query_embedding, k)

    return distances, indices


if __name__ == "__main__":

    pdf_path = "data/raw/sample_paper.pdf"

    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(text)

    embeddings = generate_embeddings(chunks)

    index = build_vector_index(embeddings)

    print("Total vectors in index:", index.ntotal)
