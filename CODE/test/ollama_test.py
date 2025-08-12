import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.llm.ollama_client import get_embedding, chat_with_ollama

# Test embedding
text = "Vietnam is a country in Southeast Asia."
embedding = get_embedding(text)

if embedding:
    print("Embedding generated successfully.")
    print(f"Length: {len(embedding)} | First 5 values: {embedding[:5]}")
else:
    print("Failed to generate embedding.")

# Test chat
query = "What is the capital of Vietnam?"
context = [{'text': "Vietnam's capital is Hanoi.", 'similarity': 0.95}]

response = chat_with_ollama(query, context)

print("\nChat Response:\n", response)
