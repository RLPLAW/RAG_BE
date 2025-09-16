import os
import fitz  # PyMuPDF
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings  # updated import
from langchain_chroma import Chroma 

# Load existing Chroma DB
def load_chroma():
    base_dir = Path(__file__).resolve().parent
    chroma_dir = base_dir.parents[2] / "chroma"

    if not os.path.exists(chroma_dir):
        print(f"Chroma directory '{chroma_dir}' does not exist")
        return None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"Loaded Chroma DB from {chroma_dir}")
    return Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)


# Extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text_parts = []
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(page_text)
        doc.close()
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return None


# Split into chunks
def split_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    text = ' '.join(text.split())  # normalize spaces
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        break_point = text.rfind('.', start, end)
        if break_point == -1:
            break_point = text.rfind(' ', start, end)
        if break_point == -1:
            break_point = end
        else:
            break_point += 1
        chunks.append(text[start:break_point].strip())
        start = max(break_point - overlap, start + 1)
        if start >= len(text):
            break
    return chunks


# Process all PDFs into ChromaDB
def process_pdfs_and_store():
    base_dir = Path(__file__).resolve().parent
    pdf_dir = base_dir.parents[2] / "AI_DOCS_TEST" / "Reports"
    chroma_dir = base_dir.parents[2] / "chroma"

    os.makedirs(chroma_dir, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)

    if not os.path.exists(pdf_dir):
        print(f"Error: PDF directory '{pdf_dir}' does not exist")
        return None

    pdf_files = [f for f in pdf_dir.iterdir() if f.suffix.lower() == ".pdf"]
    if not pdf_files:
        print(f"No PDF files found in '{pdf_dir}'")
        return None

    print(f"Found {len(pdf_files)} PDF files to process...")

    total_chunks = 0
    processed_files = 0

    for pdf_file in pdf_files:
        print(f"➡ Processing: {pdf_file.name}")
        text = extract_text_from_pdf(str(pdf_file))
        if text:
            chunks = split_text(text, chunk_size=1000, overlap=200)
            print(f"  → {len(chunks)} chunks extracted")
            if chunks:
                metadatas = [{"source": pdf_file.name, "chunk_id": i} for i in range(len(chunks))]
                db.add_texts(chunks, metadatas=metadatas)
                total_chunks += len(chunks)
                processed_files += 1
            else:
                print(f"  → No valid text in {pdf_file.name}")
        else:
            print(f"  → Failed to extract {pdf_file.name}")

    print(f"\nProcessing complete!")
    print(f"Summary: {processed_files}/{len(pdf_files)} files processed, {total_chunks} chunks added to ChromaDB")

    return db


# Get report chunks from DB
def get_report_chunks(db, report_name):
    # Retrieve ALL chunks for the given report
    return db.similarity_search(f"source:{report_name}", k=1000)  # k large enough to cover all


# Process large reports in batches to avoid Ollama crash
def process_report_in_batches(db, report_name, instruction, process_func, batch_size=5):
    results = get_report_chunks(db, report_name)
    if not results:
        print(f"No chunks found for '{report_name}'")
        return None

    modified_chunks = []
    for i in range(0, len(results), batch_size):
        batch_text = "\n".join(doc.page_content for doc in results[i:i + batch_size])
        prompt = f"""
Bạn là trợ lý pháp lý. Tôi sẽ cung cấp một phần của báo cáo trong CONTEXT.
Nhiệm vụ: {instruction}
Hãy giữ nguyên cấu trúc và định dạng phần này.

--- CONTEXT ---
{batch_text}
"""
        modified_text = process_func(prompt)
        modified_chunks.append(modified_text)

    return "\n".join(modified_chunks)
