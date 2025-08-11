import os
import fitz  # PyMuPDF
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Text extraction ---
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

# --- Chunking ---
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
        start = max(break_point - overlap, start + 1)  # Prevent infinite loop
        if start >= len(text):
            break
    return chunks

# --- Process PDFs and store in ChromaDB ---
def process_pdfs_and_store():
    # File path logic
    base_dir = Path(__file__).resolve().parent
    pdf_dir = base_dir.parents[2] / "AI_DOCS_TEST" / "Reports"
    chroma_dir = base_dir.parents[2] / "chroma"
    
    # Create directories if they don't exist
    os.makedirs(chroma_dir, exist_ok=True)
    
    # Initialize embeddings and database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)
    
    # Check if PDF directory exists
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
        print(f"Processing: {pdf_file.name}")
        
        text = extract_text_from_pdf(str(pdf_file))
        if text:
            chunks = split_text(text, chunk_size=1000, overlap=200)
            print(f"  → {len(chunks)} chunks extracted")
            
            if chunks:
                # Add metadata to track source file (fixed typo by removing "挖坑")
                metadatas = [{"source": pdf_file.name, "chunk_id": i} for i in range(len(chunks))]
                db.add_texts(chunks, metadatas=metadatas)
                total_chunks += len(chunks)
                processed_files += 1
            else:
                print(f"  → No text content found in {pdf_file.name}")
        else:
            print(f"  → Failed to extract text from {pdf_file.name}")
    
    # Persist the database
    db.persist()
    print(f"\nProcessing complete!")
    print(f"Summary: {processed_files}/{len(pdf_files)} files processed, {total_chunks} total chunks added to ChromaDB")
    
    return db