import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


# Load environment variables
load_dotenv()

# OpenAI client
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# ChromaDB
chroma_client = chromadb.PersistentClient(
    path=str(VECTORSTORE_DIR)
)

collection = chroma_client.get_or_create_collection(
    name="business_documents"
)


def extract_text_from_pdf(pdf_path):
    """
    Extract text from all pages of a PDF.
    """

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_embeddings(texts):
    """
    Create OpenAI embeddings for text chunks.
    """

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [
        item.embedding
        for item in response.data
    ]


def ingest_pdf(pdf_path):
    """
    Read a PDF, split it into chunks,
    create embeddings, and store them in ChromaDB.
    """

    pdf_path = Path(pdf_path)

    print(f"\nProcessing: {pdf_path.name}")

    # Extract text
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        print("No readable text found in the PDF.")
        return

    print(f"Extracted {len(text)} characters.")

    # Split into chunks
    chunks = split_text(text)

    print(f"Created {len(chunks)} chunks.")

    # Create embeddings
    embeddings = create_embeddings(chunks)

    # Create unique IDs
    ids = [
        f"{pdf_path.stem}_{i}"
        for i in range(len(chunks))
    ]

    # Metadata
    metadatas = [
        {
            "source": pdf_path.name,
            "chunk": i
        }
        for i in range(len(chunks))
    ]

    # Store in ChromaDB
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Successfully stored {len(chunks)} chunks.")
    print("PDF ingestion completed!")


def ingest_all_pdfs():
    """
    Process every PDF inside the data folder.
    """

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("\nNo PDF files found.")
        print(f"Put your PDF files inside: {DATA_DIR}")
        return

    print(f"\nFound {len(pdf_files)} PDF file(s).")

    for pdf_file in pdf_files:
        ingest_pdf(pdf_file)


if __name__ == "__main__":
    ingest_all_pdfs()