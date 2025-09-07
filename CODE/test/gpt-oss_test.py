import ollama
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path

# --- Config ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
MODEL = "gpt-oss:20b"  # Use gpt-oss:20b for Mac compatibility
EMBED_MODEL = "nomic-embed-text:latest"

# --- Font setup for PDF (support Vietnamese) ---
# FONT_PATH = "/System/Library/Fonts/Supplemental/DejaVuSans.ttf"
# if not Path(FONT_PATH).exists():
#     raise FileNotFoundError(f"Font file not found at {FONT_PATH}. Please install DejaVuSans or specify another font.")
# pdfmetrics.registerFont(TTFont('DejaVuSans', FONT_PATH))

# --- Check Ollama server status ---
def check_ollama_status():
    try:
        ollama.list()  # Simple API call to check if server is running
        return True
    except Exception as e:
        print(f"Ollama server not running or inaccessible: {e}")
        return False

# --- Get local embedding from Ollama ---
def get_embedding(text, model=EMBED_MODEL):
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# --- Chat with Ollama using retrieved context ---
def chat_with_ollama(prompt, model=MODEL):
    if not check_ollama_status():
        print("Lỗi: Ollama server không chạy. Vui lòng chạy 'ollama serve'.")
        return None
    try:
        response = ollama.generate(model=model, prompt=prompt, stream=False)
        return response.get("response", "").strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return None

# --- Export to .txt file ---
def export_to_txt(content, output_path):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Đã lưu file văn bản: {output_path}")
    except Exception as e:
        print(f"Lỗi khi lưu file .txt: {e}")

# --- Export to PDF file ---
def export_to_pdf(content, output_path):
    try:
        c = canvas.Canvas(output_path, pagesize=A4)
        c.setFont("Helvetica", 12)
        text_object = c.beginText(40, 800)
        text_object.setFont("Helvetica", 12)
        y_position = 800
        for line in content.split("\n"):
            if y_position < 40:
                c.drawText(text_object)
                c.showPage()
                c.setFont("Helvetica", 12)
                text_object = c.beginText(40, 800)
                y_position = 800
            text_object.textLine(line)
            y_position -= 14
        c.drawText(text_object)
        c.showPage()
        c.save()
        print(f"Đã lưu file PDF: {output_path}")
    except Exception as e:
        print(f"Lỗi khi lưu file PDF: {e}")

# --- Update report content ---
def update_report(chunks, change_from="tháng 4 năm 2025", change_to="tháng 5 năm 2025"):
    # Filter chunks containing the target text to reduce input size
    relevant_chunks = [chunk for chunk in chunks if change_from in chunk]
    if not relevant_chunks:
        print("Không tìm thấy chuỗi cần thay thế trong báo cáo.")
        return "\n".join(chunks)
    
    # Create prompt for text editing
    report_content = "\n".join(relevant_chunks)
    prompt = f"""
    Bạn được cung cấp nội dung báo cáo sau:
    "{report_content}"

    Nhiệm vụ: Thay thế tất cả các lần xuất hiện của "{change_from}" thành "{change_to}" trong nội dung báo cáo.
    - Giữ nguyên toàn bộ số liệu, cấu trúc, và định dạng của báo cáo.
    - Không thêm bất kỳ bình luận, câu hỏi, hoặc nội dung không liên quan.
    - Chỉ trả về nội dung báo cáo đã chỉnh sửa, không bao gồm bất kỳ văn bản giải thích nào.
    """
    
    updated_content = chat_with_ollama(prompt)
    if updated_content:
        # Reconstruct full report, replacing only relevant chunks
        result = []
        updated_index = 0
        for chunk in chunks:
            if chunk in relevant_chunks:
                # Extract the corresponding updated chunk
                chunk_length = len(chunk)
                updated_chunk = updated_content[updated_index:updated_index + chunk_length]
                result.append(updated_chunk)
                updated_index += chunk_length
            else:
                result.append(chunk)
        return "\n".join(result)
    else:
        # Fallback: Simple string replacement
        print("Ollama thất bại, sử dụng thay thế chuỗi đơn giản...")
        return "\n".join(chunk.replace(change_from, change_to) for chunk in chunks)

# --- Example usage ---
def main():
    # Simulate chunks from ChromaDB
    sample_chunks = [
        "Báo cáo Kết quả công tác thi hành án dân sự tháng 4 năm 2025",
        "Số liệu: 1000 vụ việc, 500 giải quyết.",
        "Kế hoạch tháng 4 năm 2025 đã hoàn thành 80%.",
        "Báo cáo khác không chứa tháng 4 năm 2025."
    ]
    
    # Update report
    updated_text = update_report(sample_chunks)
    
    # Export to files
    filename = "676-BÁO CÁO Kết quả công tác thi hành án dân sự tháng 4 năm 2025.pdf"
    export_to_txt(updated_text, f"updated_{filename}.txt")
    export_to_pdf(updated_text, f"updated_{filename}.pdf")

if __name__ == "__main__":
    main()