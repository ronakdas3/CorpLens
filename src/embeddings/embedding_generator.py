from sentence_transformers import SentenceTransformer

from src.data.pdf_loader import extract_text_from_pdf
from src.data.text_chunker import chunk_text


def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Convert text chunks into embeddings.

    Args:
        chunks (list[str]): document chunks
        model_name (str): embedding model

    Returns:
        embeddings (list): vector embeddings
    """

    model = SentenceTransformer(model_name)

    embeddings = model.encode(chunks)

    return embeddings


if __name__ == "__main__":

    pdf_path = "data/raw/sample_paper.pdf"

    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(text)

    embeddings = generate_embeddings(chunks)

    print("Number of chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)