import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text
    """

    document = fitz.open(pdf_path)
    text = ""

    for page in document:
        text += page.get_text()

    return text


if __name__ == "__main__":
    sample_pdf = "data/raw/sample_paper.pdf"
    text = extract_text_from_pdf(sample_pdf)

    # print("hello")

    print(text[:1000])  # preview first 1000 characters