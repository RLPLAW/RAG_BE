import os
import docx
import PyPDF2
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME')

# Test the Ollama model
def test_ollama_model(input_text) -> str:
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL_NAME)
        response = llm.invoke(input_text)
        return response
    except Exception as e:
        print(f"Model failed: {e}")
        return None

# Read content from a .txt file
def read_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

# Read content from a .docx file
def read_docx_file(file_path):
    try:
        doc = docx.Document(file_path)
        content = [paragraph.text for paragraph in doc.paragraphs]
        return '\n'.join(content)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

# Read content from a .pdf file
def read_pdf_file(file_path):
    try:
        content = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    content.append(text)
        return '\n'.join(content)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

# Read all supported documents from a folder and its subfolders
def read_documents_from_folder(root_path):
    documents = {}
    supported_extensions = {'.txt', '.docx', '.pdf'}
    
    if not os.path.exists(root_path):
        print(f"Root folder {root_path} does not exist!")
        return documents
    
    # Walk through all folders and subfolders
    for root, _, files in os.walk(root_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # Get file extension
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            
            if ext not in supported_extensions:
                continue
                
            print(f"Reading: {os.path.relpath(file_path, root_path)}")
            
            # Read based on file type
            content = ""
            if ext == '.txt':
                content = read_txt_file(file_path)
            elif ext == '.docx':
                content = read_docx_file(file_path)
            elif ext == '.pdf':
                content = read_pdf_file(file_path)
            
            if content.strip():  # Only add if content is not empty
                # Use relative path as key to distinguish files from different subfolders
                relative_path = os.path.relpath(file_path, root_path)
                documents[relative_path] = content
            else:
                print(f"Warning: No content extracted from {relative_path}")
    
    return documents

# Create a RAG-style prompt with document context
def create_rag_prompt(user_query, documents):
    # Combine all document contents
    document_text = ""
    for filename, content in documents.items():
        document_text += f"\n--- Document: {filename} ---\n"
        document_text += content
        document_text += "\n" + "="*50 + "\n"
    
    # Create the full prompt
    rag_prompt = f"""You are an AI assistant that answers questions based solely on the provided documents.

IMPORTANT INSTRUCTIONS:
- Only use information from the documents provided below
- If the requested information is not present in the documents, only answer "Answer not found"
- Do not use your general knowledge or make assumptions
- Be precise and cite which document you're referencing when possible

DOCUMENTS:
{document_text}

USER QUESTION: {user_query}

ANSWER:"""
    
    return rag_prompt

# Main function to perform RAG-like querying
def simple_rag_query(root_path, user_query, ollama_model_name):
    global OLLAMA_MODEL_NAME
    OLLAMA_MODEL_NAME = ollama_model_name
    
    print(f"Reading documents from: {root_path}")
    documents = read_documents_from_folder(root_path)
    
    if not documents:
        return "No documents found or readable in the specified folder or its subfolders."
    
    print(f"Successfully loaded {len(documents)} documents:")
    for filename in documents.keys():
        print(f"  - {filename}")
    
    # Create RAG prompt
    rag_prompt = create_rag_prompt(user_query, documents)
    
    print("\nQuerying the model...")
    response = test_ollama_model(rag_prompt)
    
    return response

# Example usage
if __name__ == "__main__":
    # Configuration
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'AI_DOCS_TEST'))  # Change this to your root document folder path
    
    print("=== Simple RAG Demo ===\n")
    
    # Interactive mode
    while True:
        user_query = input("\nEnter your question (or 'quit' to exit): ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_query:
            continue
            
        try:
            response = simple_rag_query(ROOT_PATH, user_query, OLLAMA_MODEL_NAME)
            print(f"\nResponse: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")