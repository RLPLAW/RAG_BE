import os
import json
import docx
import PyPDF2
from langchain.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import redis
import logging
import shutil
import time
import gc

# Load environment variables
load_dotenv()
OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', 'llama3')  # Default fallback
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL_NAME', 'nomic-embed-text')  # Default fallback

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize text splitter with better parameters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Smaller chunks for better retrieval
    chunk_overlap=50,  # Reduced overlap
    length_function=len,
)

# Test the Ollama model
def test_ollama_model(input_text) -> str:
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL_NAME)
        response = llm.invoke(input_text)
        return response
    except Exception as e:
        logger.error("Model failed: %s", e)
        return None

# Read content from a .txt file
def read_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            logger.error("Error reading %s: %s", file_path, e)
            return ""
    except Exception as e:
        logger.error("Error reading %s: %s", file_path, e)
        return ""

# Read content from a .docx file
def read_docx_file(file_path):
    try:
        doc = docx.Document(file_path)
        content = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        return '\n'.join(content)
    except Exception as e:
        logger.error("Error reading %s: %s", file_path, e)
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
        logger.error("Error reading %s: %s", file_path, e)
        return ""

# Clean up vector store and Redis cache
def cleanup_resources():
    """Clean up vector store and Redis cache"""
    try:
        # Clean up vector store
        persist_dir = "./chroma_db"
        if os.path.exists(persist_dir):
            # Force close any file handles
            gc.collect()
            time.sleep(0.5)  # Brief pause
            
            # Try multiple times if needed
            for attempt in range(3):
                try:
                    shutil.rmtree(persist_dir)
                    logger.info("Cleaned up vector store directory")
                    break
                except PermissionError as e:
                    if attempt < 2:
                        logger.warning(f"Attempt {attempt + 1} failed to clean vector store: {e}")
                        time.sleep(1)
                    else:
                        logger.error(f"Failed to clean vector store after 3 attempts: {e}")
        
        # Clean up Redis cache
        try:
            # Get all keys that match our pattern
            keys = redis_client.keys("documents:*")
            if keys:
                redis_client.delete(*keys)
                logger.info(f"Cleaned up {len(keys)} Redis cache entries")
        except Exception as e:
            logger.error(f"Error cleaning Redis cache: {e}")
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Modified process_documents function with controlled folder scanning
def process_documents(root_path, subfolder=None):
    documents = []
    supported_extensions = {'.txt', '.docx', '.pdf'}
    
    if not os.path.exists(root_path):
        logger.error("Root folder %s does not exist!", root_path)
        return documents
    
    # Construct the target path for the subfolder
    target_path = root_path
    if subfolder:
        target_path = os.path.join(root_path, subfolder)
        if not os.path.exists(target_path):
            logger.error("Subfolder %s does not exist!", target_path)
            return documents
    
    # Use a unique cache key based on root_path and subfolder
    cache_key = f"documents:{root_path}:{subfolder or 'all'}"
    
    # Skip cache for now to ensure fresh processing
    logger.info("Processing documents fresh (no cache)")
    
    # Determine scanning behavior based on subfolder parameter
    if subfolder:
        # If subfolder is specified, only scan that specific subfolder (no recursion)
        logger.info("Scanning specific subfolder: %s", target_path)
        files = []
        try:
            for filename in os.listdir(target_path):
                file_path = os.path.join(target_path, filename)
                if os.path.isfile(file_path):
                    files.append((target_path, filename))
        except Exception as e:
            logger.error("Error listing files in %s: %s", target_path, e)
            return documents
    else:
        # If no subfolder specified, scan all files and subfolders recursively
        logger.info("Scanning all folders recursively from: %s", target_path)
        files = []
        for root, _, filenames in os.walk(target_path):
            for filename in filenames:
                files.append((root, filename))
    
    # Process the collected files
    for root, filename in files:
        file_path = os.path.join(root, filename)
        logger.info("Scanning: %s", os.path.relpath(file_path, root_path))
        
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        if ext not in supported_extensions:
            continue
            
        logger.info("Processing: %s", os.path.relpath(file_path, root_path))
        
        content = ""
        if ext == '.txt':
            content = read_txt_file(file_path)
        elif ext == '.docx':
            content = read_docx_file(file_path)
        elif ext == '.pdf':
            content = read_pdf_file(file_path)
        
        if content.strip():
            chunks = text_splitter.split_text(content)
            logger.info("Created %d chunks from %s", len(chunks), filename)
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    documents.append({
                        'content': chunk,
                        'metadata': {
                            'source': os.path.relpath(file_path, root_path),
                            'chunk_id': i,
                            'filename': filename
                        }
                    })
        else:
            logger.warning("No content extracted from %s", os.path.relpath(file_path, root_path))
    
    logger.info("Total documents processed: %d", len(documents))
    return documents

# This function is now integrated directly into simple_rag_query for better isolation

# Enhanced RAG prompt with better instructions
def create_rag_prompt(user_query, retrieved_docs):
    if not retrieved_docs:
        return f"""You are an AI assistant. The user asked: "{user_query}"

No relevant documents were found to answer this question.

Please respond with: "Answer not found"
"""
    
    context = ""
    for i, doc in enumerate(retrieved_docs, 1):
        content = doc.page_content
        source = doc.metadata.get('source', 'Unknown')
        context += f"\n--- Document {i}: {source} ---\n"
        context += content
        context += "\n" + "="*50 + "\n"
    
    rag_prompt = f"""You are an AI assistant that answers questions based on provided documents.

INSTRUCTIONS:
- Answer the question using ONLY the information from the documents below
- If the information is not in the documents, respond with "Answer not found"
- Be specific and cite which document contains the information
- Provide a clear, direct answer

DOCUMENTS:
{context}

QUESTION: {user_query}

ANSWER:"""
    
    return rag_prompt

# Enhanced RAG query function with complete isolation
def simple_rag_query(root_path, user_query, ollama_model_name, subfolder=None):
    global OLLAMA_MODEL_NAME
    OLLAMA_MODEL_NAME = ollama_model_name
    
    # Clean up at the start of each query
    cleanup_resources()
    
    # Force garbage collection to ensure clean state
    gc.collect()
    
    logger.info("=== NEW QUERY SESSION ===")
    logger.info("Processing documents from: %s%s", root_path, f", subfolder: {subfolder}" if subfolder else "")
    
    # Process documents for THIS query only
    documents = process_documents(root_path, subfolder)
    
    if not documents:
        return f"No documents found or readable in the specified folder{subfolder and f'/{subfolder}' or ''} or its subfolders."
    
    logger.info("Successfully processed %d document chunks for this query", len(documents))
    
    # Debug: Show what sources we're using for THIS query
    sources = set()
    for doc in documents:
        sources.add(doc['metadata']['source'])
    logger.info("Sources for this query: %s", list(sources))
    
    # Debug: Show sample document content
    if documents:
        sample_doc = documents[0]
        logger.info("Sample document content (first 200 chars): %s...", sample_doc['content'][:200])
    
    # Create a completely fresh vector store for THIS query only
    logger.info("Creating fresh vector store for this query...")
    vector_store = None
    
    try:
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        logger.info("Creating embeddings for %d text chunks", len(texts))
        
        # Create a completely new embedding instance
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Create fresh vector store with unique collection name
        import uuid
        collection_name = f"temp_collection_{uuid.uuid4().hex[:8]}"
        
        vector_store = Chroma.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            collection_name=collection_name
        )
        
        logger.info("Fresh vector store created successfully with collection: %s", collection_name)
        
        # Retrieve relevant chunks
        logger.info("Searching for query: %s", user_query)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        retrieved_docs = retriever.invoke(user_query)
        
        logger.info("Retrieved %d relevant chunks", len(retrieved_docs))
        
        # Debug: Show retrieved content and verify sources
        retrieved_sources = set()
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', 'Unknown')
            retrieved_sources.add(source)
            logger.info("Retrieved doc %d from '%s' (first 100 chars): %s...", 
                       i+1, source, doc.page_content[:100])
        
        logger.info("Retrieved documents are from sources: %s", list(retrieved_sources))
        
        # Verify that retrieved sources match our current document sources
        if not retrieved_sources.issubset(sources):
            logger.error("ERROR: Retrieved sources don't match current document sources!")
            logger.error("Current sources: %s", list(sources))
            logger.error("Retrieved sources: %s", list(retrieved_sources))
            return "Internal error: Retrieved documents from wrong sources!"
        
        # Create RAG prompt
        rag_prompt = create_rag_prompt(user_query, retrieved_docs)
        
        # Debug: Show prompt length
        logger.info("RAG prompt length: %d characters", len(rag_prompt))
        
        logger.info("Querying the model...")
        response = test_ollama_model(rag_prompt)
        
        result = response if response else "Failed to generate response."
        
    except Exception as e:
        logger.error("Error during retrieval/generation: %s", e)
        result = "Failed to retrieve relevant documents or generate response."
    
    finally:
        # Aggressive cleanup after each query
        try:
            if vector_store is not None:
                # Try to delete the collection if possible
                try:
                    vector_store.delete_collection()
                    logger.info("Deleted vector store collection")
                except:
                    pass
                del vector_store
            
            # Clear any remaining references
            if 'embedding' in locals():
                del embedding
            if 'texts' in locals():
                del texts
            if 'metadatas' in locals():
                del metadatas
            if 'retrieved_docs' in locals():
                del retrieved_docs
                
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)
                
            logger.info("=== QUERY SESSION CLEANED UP ===")
            
        except Exception as e:
            logger.error("Error during final cleanup: %s", e)
    
    return result

# Test function to verify setup
def test_setup():
    """Test if Ollama models are working"""
    logger.info("Testing Ollama setup...")
    
    # Test LLM
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL_NAME)
        test_response = llm.invoke("Say 'Hello, I am working!'")
        logger.info("LLM test response: %s", test_response)
    except Exception as e:
        logger.error("LLM test failed: %s", e)
        return False
    
    # Test embeddings
    try:
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
        test_embedding = embedding.embed_query("test query")
        logger.info("Embedding test successful, dimension: %d", len(test_embedding))
    except Exception as e:
        logger.error("Embedding test failed: %s", e)
        return False
    
    return True

# Updated example usage
if __name__ == "__main__":
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'AI_DOCS_TEST'))
    
    print("=== Vector RAG Demo ===\n")
    
    # Test setup first
    if not test_setup():
        print("Setup test failed. Please check your Ollama installation and models.")
        exit(1)
    
    print("Setup test passed!")
    print(f"Root path: {ROOT_PATH}")
    print("\nFolder scanning behavior:")
    print("- Leave subfolder empty: scans ALL folders and subfolders recursively")
    print("- Enter subfolder name: scans ONLY that specific subfolder (no subfolders)")
    
    # Interactive mode
    while True:
        user_query = input("\nEnter your question (or 'quit' to exit): ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_query:
            continue
            
        subfolder = input("Enter subfolder to process (or press Enter for all folders): ").strip()
        subfolder = subfolder if subfolder else None
            
        try:
            response = simple_rag_query(ROOT_PATH, user_query, OLLAMA_MODEL_NAME, subfolder)
            print(f"\nResponse: {response}")
        except Exception as e:
            logger.error("Error: %s", e)
    
    # Final cleanup
    cleanup_resources()
    print("Goodbye!")