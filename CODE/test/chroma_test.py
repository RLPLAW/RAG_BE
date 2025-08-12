import chromadb
from chromadb.config import Settings

# Tạo Chroma client local với mặc định
client = chromadb.PersistentClient(path="./chroma_db")


# Tạo collection
collection = client.get_or_create_collection(name="my_test_collection")

# Thêm documents + embedding
collection.add(
    documents=["hello world", "chroma test", "openai is great"],
    ids=["id1", "id2", "id3"],
    metadatas=[{"source": "a"}, {"source": "b"}, {"source": "c"}]
)

# Tìm gần giống
results = collection.query(
    query_texts=["hello chroma"],
    n_results=2
)

print(results)
