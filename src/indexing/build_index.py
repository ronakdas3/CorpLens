import os
import faiss
import numpy as np

from src.data.pdf_loader import extract_text_from_pdf
from src.data.text_cleaning import remove_references
from src.data.text_chunker import chunk_text
from src.models.model_manager import get_embedding_model


def build_index_for_paper(pdf_path, save_dir):
    """
    Build FAISS index for a single PDF document and save it.

    Args:
        pdf_path (str): path to input PDF
        save_dir (str): directory to save index + chunks
    """

    print(f"\nProcessing: {pdf_path}")

    # 1. Extract text
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted text length: {len(text)}")

    # 2. Clean text (remove references)
    text = remove_references(text)
    print("Removed references")

    # 3. Split into chunks
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    if len(chunks) == 0:
        raise ValueError("No chunks created. Check text extraction or splitting.")

    # 4. Load embedding model
    model = get_embedding_model()
    print("Loaded embedding model")

    # 5. Generate embeddings
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings)

    print(f"Embeddings shape: {embeddings.shape}")

    # 6. Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)
    print(f"Added {index.ntotal} vectors to FAISS index")

    # 7. Save outputs
    os.makedirs(save_dir, exist_ok=True)

    index_path = os.path.join(save_dir, "faiss.index")
    chunks_path = os.path.join(save_dir, "chunks.npy")

    faiss.write_index(index, index_path)
    np.save(chunks_path, chunks)

    print(f"Saved index to: {index_path}")
    print(f"Saved chunks to: {chunks_path}")


# -------------------- MAIN (OPTIONAL CLI USAGE) --------------------

if __name__ == "__main__":

    # Example corporate policy indexing
    build_index_for_paper(
        "data/corporate_docs/employee_handbook/document.pdf",
        "data/corporate_docs/employee_handbook"
    )

    build_index_for_paper(
        "data/corporate_docs/infosec_policy/document.pdf",
        "data/corporate_docs/infosec_policy"
    )

    build_index_for_paper(
        "data/corporate_docs/procurement_policy/document.pdf",
        "data/corporate_docs/procurement_policy"
    )

    build_index_for_paper(
        "data/corporate_docs/engineering_standards/document.pdf",
        "data/corporate_docs/engineering_standards"
    )
