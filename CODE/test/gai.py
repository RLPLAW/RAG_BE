import subprocess
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Hàm chia nhỏ văn bản thành nhiều chunk
def chunk_text(text, max_chars=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # tìm điểm ngắt hợp lý (dấu chấm hoặc khoảng trắng)
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

# Đọc PDF gốc
reader = PdfReader("report.pdf")
full_text = ""
for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        full_text += extracted + "\n"

#Chia nhỏ thành chunks
chunks = chunk_text(full_text, max_chars=1500, overlap=200)
print(f"Tổng số chunks: {len(chunks)}")

#Gửi từng chunk cho Ollama để dịch
translated_chunks = []
for i, chunk in enumerate(chunks):
    prompt = f"Hãy dịch nội dung sau sang tiếng Việt và giữ nguyên định dạng cơ bản:\n\n{chunk}"
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    output = result.stdout.decode("utf-8").strip()
    translated_chunks.append(output)
    print(f"Chunk {i+1}/{len(chunks)} dịch xong")

# Xuất ra PDF mới
c = canvas.Canvas("report_translated.pdf", pagesize=letter)
text_obj = c.beginText(40, 750)
for chunk in translated_chunks:
    for line in chunk.split("\n"):
        text_obj.textLine(line)
c.drawText(text_obj)
c.save()

print("File report_translated.pdf đã tạo xong!")
