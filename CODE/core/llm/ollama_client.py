import ollama
import os
import requests

# --- Config ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
OLLAMA_API_URL = f"http://{OLLAMA_HOST}/api/generate"
OLLAMA_EMBED_URL = f"http://{OLLAMA_HOST}/api/embeddings"

# --- Get local embedding from Ollama ---
def get_embedding(text, model="nomic-embed-text:latest"):
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# --- Chat with Ollama using retrieved context ---
def chat_with_ollama(prompt, model="llama3:latest"):
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return None
