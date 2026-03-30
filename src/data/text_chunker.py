def chunk_text(text, chunk_size=800, overlap=100):
    """
    Split text into overlapping chunks.

    Args:
        text (str): input document text
        chunk_size (int): characters per chunk
        overlap (int): overlap between chunks

    Returns:
        list[str]: list of text chunks
    """


    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    from pdf_loader import extract_text_from_pdf

    pdf_path = "data/raw/sample_paper.pdf"

    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(text)

    print("Number of chunks:", len(chunks))
    print("\nFirst chunk preview:\n")
    print(chunks[0])


    #why do we need/use overlap in chunking