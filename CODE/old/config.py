import sys
sys.path.insert(0, r'D:\Softwares\Projects\VSCode\src\misc_projects\RAG\lib')
import ollama
import redis
import json
import os
import re
from pymongo import MongoClient
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import discord
from discord.ext import commands
import io
import asyncio
from dotenv import load_dotenv
import fitz  # Using PyMuPDF as primary PDF library
import time
import requests

# Load environment
load_dotenv()

# Configuration
GOOGLE_CREDS = os.getenv('GOOGLE_DRIVE_CREDENTIALS_FILE')
FILE_ID = os.getenv('GOOGLE_DRIVE_FILE_ID')
MONGO_URI = os.getenv('MONGO_URI')
MONGO_DB = os.getenv('MONGO_DB', 'angler')
MONGO_COLL = os.getenv('MONGO_COLLECTION', 'Vector')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASS = os.getenv('REDIS_PASSWORD', '')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
DISCORD_CHANNEL = int(os.getenv('DISCORD_CHANNEL', 0))
MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')  # Ensures 768D embeddings
LANGUAGE_MODEL = os.getenv('LANGUAGE_MODEL', 'mistral')  # Default for chat
DOCUMENT_READY = False

# Set Ollama host
OLLAMA_HOST = os.getenv('OLLAMA_HOST', '127.0.0.1:11434')
os.environ['OLLAMA_HOST'] = OLLAMA_HOST
print(f"Using Ollama host: {OLLAMA_HOST}")
print(f"{GOOGLE_CREDS=}, {FILE_ID=}, {MONGO_URI=}, {MONGO_DB=}, {MONGO_COLL=}, {DISCORD_TOKEN=}, {DISCORD_CHANNEL=}, {REDIS_HOST=}")

# Validate environment variables
if not all([GOOGLE_CREDS, FILE_ID, MONGO_URI, MONGO_DB, DISCORD_TOKEN, DISCORD_CHANNEL, REDIS_HOST]):
    raise ValueError("Missing required environment variables")

# Helper function to check Ollama server status
def check_ollama_server():
    try:
        response = requests.get(f"http://{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            print(f"Ollama server is running at {OLLAMA_HOST}")
            return True
        else:
            print(f"Ollama server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"Failed to connect to Ollama server at {OLLAMA_HOST}: {e}")
        return False

# 1. Google Drive download
def download_from_drive(file_id):
    print(f"[INFO] Starting download for file_id: {file_id}")
    
    try:
        # Verify credentials file
        if not os.path.exists(GOOGLE_CREDS):
            print(f"[ERROR] Credentials file not found at: {GOOGLE_CREDS}")
            return None
        
        print(f"[DEBUG] Loading service account credentials from: {GOOGLE_CREDS}")
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_CREDS, scopes=['https://www.googleapis.com/auth/drive']
        )
        print("[DEBUG] Building Google Drive API service")
        service = build('drive', 'v3', credentials=creds)
        
        # Get file metadata
        print(f"[DEBUG] Fetching metadata for file_id: {file_id}")
        file_metadata = service.files().get(fileId=file_id, fields='id,name,mimeType,permissions').execute()
        file_name = file_metadata.get('name', 'Unknown')
        mime_type = file_metadata.get('mimeType', 'Unknown')
        print(f"[INFO] File metadata - Name: {file_name}, MIME: {mime_type}, Permissions: {file_metadata.get('permissions', [])}")
        
        # Download based on MIME type
        if mime_type == 'application/pdf':
            print("[DEBUG] Downloading PDF file")
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"[DEBUG] Download progress: {int(status.progress() * 100)}%")
            print(f"[INFO] Successfully downloaded PDF: {file_name}")
            # Extract text from PDF
            print("[DEBUG] Extracting text from PDF")
            pdf_document = fitz.open(stream=fh.getvalue(), filetype="pdf")
            text = ""
            for page in pdf_document:
                text += page.get_text()
            pdf_document.close()
            print(f"[INFO] Extracted text from PDF: {file_name}")
            return text
        
        elif 'document' in mime_type or 'text' in mime_type:
            print(f"[DEBUG] Handling document/text file with MIME: {mime_type}")
            if 'google-apps.document' in mime_type:
                print("[DEBUG] Exporting Google Docs as text/plain")
                request = service.files().export_media(fileId=file_id, mimeType='text/plain')
            else:
                print("[DEBUG] Downloading text file directly")
                request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"[DEBUG] Download progress: {int(status.progress() * 100)}%")
            text = fh.getvalue().decode('utf-8', errors='ignore')
            print(f"[INFO] Successfully downloaded text: {file_name}")
            return text
        
        else:
            print(f"[WARNING] Unsupported file type: {mime_type}")
            return None
    
    except Exception as e:
        print(f"[ERROR] Error downloading file_id {file_id}: {str(e)}")
        return None

# 1.2. PDF text extraction
def extract_text_from_pdf(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
            text += "\n\n"
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None

# 1.3. Improved text splitting
def split_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    
    text = text.strip()
    text = ' '.join(text.split())  # Normalize whitespace
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at sentence end
        break_point = text.rfind('.', start, end)
        if break_point == -1:
            # Try to break at word boundary
            break_point = text.rfind(' ', start, end)
        if break_point == -1:
            # Force break at chunk_size
            break_point = end
        else:
            break_point += 1  # Include the delimiter
        
        chunks.append(text[start:break_point].strip())
        start = break_point - overlap
        if start < 0:
            start = 0
    
    return [chunk for chunk in chunks if chunk.strip()]

# 1.4. Generate embeddings with retry
def get_embedding(text, max_retries=3, retry_delay=2):
    try:
        text = text.strip()
        if not text:
            return None
        
        for attempt in range(max_retries):
            try:
                if not check_ollama_server():
                    print(f"Ollama server not available, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
                
                response = ollama.embeddings(model=MODEL, prompt=text)
                embedding = response.get('embedding') or response.get('embeddings')
                
                if not embedding:
                    raise ValueError(f"Model '{MODEL}' returned no embedding data")
                
                # Check if it's actually 768 dimensions
                if len(embedding) != 768:
                    print(f"Warning: Expected 768 dimensions, got {len(embedding)} for model {MODEL}")
                    # You might want to adjust this based on your actual model
                    return embedding  # Return anyway for now
                
                return embedding
                
            except Exception as e:
                print(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
        return None
    except Exception as e:
        print(f"Error generating embedding after {max_retries} attempts: {e}")
        return None

# 1.5. Store to MongoDB
def store_to_mongo(embeddings_chunks, collection_name=MONGO_COLL):
    client = None
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]
        collection.delete_many({})
        successful_inserts = 0
        for chunk, vector in embeddings_chunks:
            if vector and chunk.strip():
                collection.insert_one({"text": chunk, "embedding": vector})
                successful_inserts += 1
        print(f"Stored {successful_inserts} documents to MongoDB")
    except Exception as e:
        print(f"Error storing to MongoDB: {e}")
    finally:
        if client:
            client.close()

# 1.6. Initialize document
async def initialize_document():
    global DOCUMENT_READY
    if DOCUMENT_READY:
        return

    print("Initializing document...")

    text = download_from_drive(FILE_ID)
    if not text:
        print("Failed to download/extract document.")
        return

    print(f"Extracted {len(text)} characters")

    chunks = split_text(text)
    embeddings_chunks = []
    
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding:
            embeddings_chunks.append((chunk, embedding))
        if (i + 1) % 10 == 0 or i + 1 == len(chunks):
            print(f"Processed {i + 1}/{len(chunks)} chunks")

    print(f"Generated {len(embeddings_chunks)} embeddings")
    store_to_mongo(embeddings_chunks)
    DOCUMENT_READY = True
    print("Document initialization complete!")

# 6. Check search index
def check_search_index(collection_name=MONGO_COLL, index_name='default'):
    client = None
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]
        
        search_indexes = list(collection.aggregate([{"$listSearchIndexes": {}}]))
        for index in search_indexes:
            if index['name'] == index_name and index['type'] == 'vectorSearch':
                return True
        
        print(f"Vector search index '{index_name}' not found.")
        return False
    except Exception as e:
        print(f"Error checking index: {e}")
        return False
    finally:
        if client:
            client.close()

# 7. Search MongoDB
def search_mongo(query_embedding, collection_name=MONGO_COLL, index_name='default', field_name='embedding', limit=3):
    if not query_embedding:
        return []
    if not check_search_index(collection_name, index_name):
        print(f"Vector search index '{index_name}' not configured.")
        return []
    
    client = None
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[collection_name]
        
        pipeline = [
            {
                '$vectorSearch': {
                    'index': index_name,
                    'path': field_name,
                    'queryVector': query_embedding,
                    'numCandidates': 50,
                    'limit': limit
                }
            }
        ]
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        print(f"Error searching MongoDB: {e}")
        return []
    finally:
        if client:
            client.close()

# 8. Cosine similarity
def cosine_similarity(a, b):
    try:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0

# 9. Redis storage
def store_to_redis(key, value, ttl=300):
    r = None
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS, decode_responses=True)
        r.setex(key, ttl, json.dumps(value, default=str))  # Added default=str for serialization
        print(f"Stored in Redis - Key: {key}, TTL: {ttl}")
    except Exception as e:
        print(f"Error storing to Redis: {e}")
    finally:
        if r:
            r.close()

# 10. Document processing
async def process_document(file_id, query, user_id):
    print(f"Processing query: {query}")
    
    if not DOCUMENT_READY:
        return [{"text": "Document not ready.", "similarity": 0}]
    
    query_embedding = get_embedding(query)
    if not query_embedding:
        return [{"text": "Failed to generate query embedding", "similarity": 0}]
    
    results = search_mongo(query_embedding, limit=3)
    similarities = [
        {"text": r['text'], "similarity": cosine_similarity(query_embedding, r['embedding'])}
        for r in results if 'embedding' in r
    ]
    
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    high_similarities = [s for s in similarities if s['similarity'] > 0.3]  # Lowered threshold
    if high_similarities:
        store_to_redis(f"chat:{user_id}", high_similarities)
    
    return high_similarities if high_similarities else similarities[:3]  # Return top 3 even if low similarity

# 10.1 Chat response with retry
async def get_chat_response(query, context_chunks=None, user_id=None):
    max_retries = 3
    retry_delay = 2
    try:
        instruction_prompt = (
            "You are a helpful assistant that answers questions based on provided document context. "
            "Use the following context to provide accurate and concise answers. If the answer is not in the context, "
            "state 'The document does not contain this information' and provide a general answer if possible. "
            "Summarize the context and make a clear and concise answer with a clear explanation or evaluation. "
            "Do not mention anything about the context or similarity scores in your final answer.\n\n"
            "Context:\n"
        )
        
        if context_chunks:
            for i, chunk in enumerate(context_chunks, 1):
                similarity = chunk.get('similarity', 0)
                text_preview = chunk['text'][:800] if len(chunk['text']) > 800 else chunk['text']
                instruction_prompt += f"Chunk {i}:\n{text_preview}\n\n"
        else:
            instruction_prompt += "No relevant document chunks found.\n\n"
        
        instruction_prompt += f"User Query: {query}\n\nAnswer:"
        
        for attempt in range(max_retries):
            try:
                if not check_ollama_server():
                    print(f"Ollama server not available, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    return "Ollama server is not available. Please try again later."
                
                print(f"Sending chat request to Ollama at {OLLAMA_HOST}, attempt {attempt + 1}")
                
                response = ollama.chat(
                    model=LANGUAGE_MODEL,
                    messages=[
                        {'role': 'system', 'content': instruction_prompt},
                        {'role': 'user', 'content': query},
                    ],
                    stream=False,  # Changed to False for simpler handling
                )
                
                if response and 'message' in response and 'content' in response['message']:
                    answer = response['message']['content'].strip()
                    
                    if user_id:
                        store_to_redis(f"chat:{user_id}", [{"text": answer, "similarity": 0}])
                    
                    return answer if answer else "No response generated by the model."
                else:
                    print(f"Unexpected response structure: {response}")
                    return "Unexpected response from the model."
                    
            except Exception as e:
                print(f"Chat attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise
        
        return "Failed to get response after multiple attempts."
        
    except Exception as e:
        print(f"Ollama chat error after {max_retries} attempts: {type(e).__name__}: {e}")
        return "Sorry, I couldn't generate a response due to a server connection issue. Please try again later."

# 11. Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    if check_ollama_server():
        await initialize_document()
    else:
        print("Ollama server not available, skipping document initialization.")

@bot.command(name='search')
async def search_command(ctx, *, query):
    if ctx.channel.id != DISCORD_CHANNEL:
        return
    
    await ctx.send(f"Searching for: '{query}'...")
    
    if not DOCUMENT_READY:
        await ctx.send("Document not ready. Please wait for initialization to complete.")
        return

    try:
        similarities = await process_document(FILE_ID, query, str(ctx.author.id))
    
        if similarities and similarities[0]['similarity'] > 0.2:  # Lowered threshold
            response = f"**Top Results for: '{query}'**\n\n"
            for i, result in enumerate(similarities[:3], 1):
                similarity_score = result['similarity']
                text_preview = result['text'][:500] + "..." if len(result['text']) > 500 else result['text']
                response += f"**{i}. Match (Similarity: {similarity_score:.3f})**\n{text_preview}\n\n"
        else:
            response = "No results found with sufficient similarity."
        
        if len(response) > 2000:
            response = response[:1997] + "..."
        
        await ctx.send(response)
        
    except Exception as e:
        await ctx.send(f"Error processing request: {str(e)}")

@bot.command(name='chat')
async def chat_command(ctx, *, query):
    if ctx.channel.id != DISCORD_CHANNEL:
        return
    
    await ctx.send(f"Processing query: '{query}'...")
    
    if not DOCUMENT_READY:
        await ctx.send("Document not ready. Please wait for initialization to complete.")
        return

    try:
        # Get top matches
        similarities = await process_document(FILE_ID, query, str(ctx.author.id))
        
        # Get Ollama response with context
        ollama_response = await get_chat_response(query, similarities[:3], str(ctx.author.id))
        
        # Send only the Ollama response
        if len(ollama_response) > 2000:
            # Split long responses
            for i in range(0, len(ollama_response), 2000):
                chunk = ollama_response[i:i+2000]
                await ctx.send(chunk)
        else:
            await ctx.send(ollama_response)
        
    except Exception as e:
        await ctx.send(f"Error processing chat request: {str(e)}")

@bot.command(name='info')
async def info_command(ctx):
    if ctx.channel.id != DISCORD_CHANNEL:
        return
    
    status = "Ready" if DOCUMENT_READY else "Not Ready"
    server_status = "Online" if check_ollama_server() else "Offline"
    
    await ctx.send(
        f"**PDF Search Bot**\n"
        f"Document Status: {status}\n"
        f"Ollama Server: {server_status}\n"
        f"Commands: `!search <query>`, `!chat <query>`, `!info`, `!stop`\n"
        f"Using {MODEL} for embeddings and {LANGUAGE_MODEL} for chat"
    )

@bot.command(name='stop')
async def stop_command(ctx):
    if ctx.channel.id != DISCORD_CHANNEL:
        return
    
    await ctx.send("Shutting down the bot...")
    try:
        await bot.close()
        print(f"Bot stopped by {ctx.author.name}")
    except Exception as e:
        await ctx.send(f"Error stopping the bot: {str(e)}")

# Main
if __name__ == "__main__":
    try:
        bot.run(DISCORD_TOKEN)
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Bot crashed: {e}")