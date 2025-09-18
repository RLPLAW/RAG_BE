676-BÁO CÁO Kết quả công tác thi hành án dân sự tháng 4 năm 2025.pdf
Đổi tháng từ 4/2025 thành 5/2025, giữ nguyên toàn bộ số liệu và cấu trúc
source .venv/bin/activate

### NOT TESTED ON **MAC**

# RAG-Law

A Retrieval-Augmented Generation (RAG) system for processing and searching legal documents.

## Overview

This project processes PDF legal documents, stores their text embeddings in ChromaDB, and uses Ollama for answering queries based on the stored data. It is designed for Windows (not tested on macOS) and uses Python 3.13.4, Ollama 0.9.0, ChromaDB, and Redis.

## Main Requirements

- **Install Python**:

  - Download Python 3.13.4 from the [official website](https://www.python.org/downloads/).
  - Ensure Python is added to your system PATH during installation.
  - This project is currently using version **3.13.4** for **Windows**.

- **Install Ollama**:

  - Download Ollama 0.9.0 from [here](https://ollama.com/download).
  - **Optional**: Install in a custom directory:
    ```bash
    cd <directory_with_ollama_setup>
    OllamaSetup.exe /DIR=[CUSTOM_DIRECTORY]
    ```
  - **Optional**: Customize where Ollama stores models when running `ollama pull [MODEL]`:
    - Open `Environment Variables` (Windows: `Win + S` → search "environment variables" → select "Edit the system environment variables" → "Environment Variables...").
    - Add a new system variable:
      - **Variable name**: `OLLAMA_MODELS`
      - **Variable value**: `<custom_directory_for_models>`
    - This project is using version **0.9.0** for **Windows**.

- **Chroma for storing vectors**:

  - Requires Docker. Install [Docker](https://www.docker.com/) if not already installed.
  - Run ChromaDB container:
    ```bash
    docker run -d --name chroma-vectors -p 8000:8000 chromadb/chroma:latest
    ```

- **Redis**:
  - Requires Docker. Uses port `6379` for database access and `8001` for the Redis web UI.
  - Run Redis container:
    ```bash
    docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
    ```

## Prerequisites

Install the required Python packages by running the following in your terminal:

```bash
pip install sentence-transformers python-docx PyPDF2 redis langchain langchain-ollama python-dotenv langchain-community chromadb
```

Pull the required Ollama models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

Alternatively, install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If using a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Project Structure

```
RAG-Law/
├── code/
│   ├── core/
│   │   └── vector/
│   │       ├── chroma_client.py
│   │       ├── ollama_interaction.py
│   │       └── main.py
│   └── ...
├── AI_DOCS_TEST/
│   └── Reports/   # PDF documents go here
├── chroma/         # ChromaDB storage
├── requirements.txt
└── README.md
```

- **chroma_client.py**: Handles PDF text extraction, chunking, and storing embeddings in ChromaDB.
- **ollama_interaction.py**: Manages embedding generation and query answering with Ollama.
- **main.py**: Orchestrates the RAG process by calling functions from the above scripts.
- **AI_DOCS_TEST/Reports/**: Directory for input PDF documents.
- **chroma/**: Directory for ChromaDB persistent storage.

## Usage

1. **Prepare Documents**:

   - Place PDF documents in `RAG-Law/AI_DOCS_TEST/Reports/`.

2. **Verify ChromaDB**:

   - Check if ChromaDB is running by visiting `localhost:8000/api/v2` in a web browser.
   - To inspect stored vectors, use ChromaDB's API or client tools.

3. **Run the Client**:
   - Ensure your virtual environment is activated (if used).
   - Run the main script:
     ```bash
     python code/core/vector/main.py
     ```
   - The script processes PDFs, stores embeddings in ChromaDB, and prompts for a query (e.g., "What is AI?").

## Bugs and Fixes

### 1. `ollama serve` Error

**Error**: `listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address (protocol/network address/port) is normally permitted.`

- **Check which application is using port 11434**:

  ```bash
  for /f "tokens=5" %a in ('netstat -aon ^| findstr :11434') do tasklist /FI "PID eq %a"
  ```

- **Fix**:
  - **Temporary**: Set a different port in the terminal:
    ```bash
    set OLLAMA_HOST=127.0.0.1:11435
    ```
  - **Permanent**: Add to environment variables:
    - **Variable name**: `OLLAMA_HOST`
    - **Variable value**: `127.0.0.1:11435`
    - Follow the steps in the Ollama installation section to add environment variables.

### 2. `ModuleNotFoundError: No module named 'langchain_community'`

- **Cause**: Some LangChain modules have been split into separate packages.
- **Solution**:
  ```bash
  pip install langchain-community
  ```
  Or update your import according to the new structure.

### 3. `MemoryError` when splitting large PDF text

- **Cause**: The `split_text` function loads too much data into memory at once.
- **Solution**:
  - Process PDFs in smaller chunks.
  - Use streaming text splitters like `RecursiveCharacterTextSplitter` from LangChain.
  - Increase available RAM or run on a machine with more memory.

### 4. Deprecation Warning for `Chroma`

- **Cause**: `Chroma` in `langchain_community.vectorstores` is deprecated since LangChain 0.2.9.
- **Solution**:
  ```bash
  pip install -U langchain-chroma
  ```
  Update the import in your scripts:
  ```python
  from langchain_chroma import Chroma
  ```

### 5. `Error: PDF directory '../AI_DOCS_TEST/Reports' does not exist`

- **Cause**: The path to the PDF directory is relative and does not match your folder structure.
- **Solution**:
  - **Option 1**: Use an absolute path in `chroma_client.py`:
    ```python
    pdf_dir = r"D:\Bu\Projects\RAG-Law\AI_DOCS_TEST\Reports"
    ```
  - **Option 2**: Use `Path` to navigate relative to the script location (already implemented in `chroma_client.py`):
    ```python
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    pdf_dir = base_dir.parents[2] / "AI_DOCS_TEST" / "Reports"
    ```

### 6. Segmentation fault when running `chroma_client.py`

- **Cause**: Often due to version conflicts between `chromadb`, `langchain`, and the embedding model.
- **Solution**:
  - Ensure a clean environment:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # Windows
    source .venv/bin/activate  # Linux/macOS
    ```
  - Reinstall dependencies:
    ```bash
    pip install --upgrade --force-reinstall -r requirements.txt
    ```
  - Test with a minimal script to load embeddings.

## Documents

### ChromaDB

- **Check if Chroma is running**:
  - Visit `localhost:8000/api/v2` in a web browser.
- **Inspect stored vectors**:
  - Use ChromaDB's API or client tools to view the contents of the vector database.

## Notes

- Ensure your document folder structure (`RAG-Law/AI_DOCS_TEST/Reports/`) is correct before running the script.
- Always activate your virtual environment before installing or running scripts.
- For large datasets, consider using persistent vector storage with ChromaDB to optimize performance.
