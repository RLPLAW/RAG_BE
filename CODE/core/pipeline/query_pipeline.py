from vector.chroma_client import process_pdfs_and_store
from llm.ollama_client import chat_with_ollama

def get_full_report_from_db(db, report_name):
    # Tìm tất cả chunk có metadata source = report_name
    results = db.similarity_search(f"source:{report_name}", k=50)
    full_text = "\n".join([doc.page_content for doc in results])
    return full_text

def answer_query(db, report_name, modification_instruction):
    if not db:
        raise ValueError("Database is empty. Please process PDFs first.")

    # Lấy toàn bộ báo cáo từ DB
    # Limit context size
    report_text = get_full_report_from_db(db, report_name)
    if not report_text:
        raise ValueError(f"Không tìm thấy báo cáo: {report_name}")

    # Prompt cho Ollama
    prompt = f"""
Bạn là trợ lý pháp lý. Tôi sẽ cung cấp toàn bộ báo cáo trong phần CONTEXT.
Nhiệm vụ: {modification_instruction}
Hãy giữ nguyên cấu trúc và định dạng báo cáo gốc.

--- CONTEXT ---
{report_text}
"""

    # Gửi prompt tới Ollama
    return chat_with_ollama(prompt)

def init_db():
    return process_pdfs_and_store()
