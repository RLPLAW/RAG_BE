from vector.chroma_client import process_pdfs_and_store
from llm.ollama_client import get_embedding, chat_with_ollama

def answer_query(query, db):
    if not db:
        print("No database available. Please process PDFs first.")
        return
    
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    if not query_embedding:
        print("Failed to generate query embedding.")
        return

    # Search for similar document chunks
    results = db.similarity_search_with_score(query, k=5)
    context_chunks = [
        {"text": doc.page_content, "similarity": score, "metadata": doc.metadata}
        for doc, score in results
    ]

    # Answer the query using Ollama
    response = chat_with_ollama(query, context_chunks)
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    # Step 1: Process PDFs and store in ChromaDB
    db = process_pdfs_and_store()
    
    # Step 2: Allow user to input a query
    user_query = input("Enter your query (e.g., 'What is AI?'): ") or "What is AI?"
    answer_query(user_query, db)