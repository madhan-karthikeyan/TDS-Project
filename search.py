import httpx
from qdrant_client import QdrantClient

API_URL = "https://aipipe.org/openai/v1/embeddings"
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDE2NDZAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.M8zMXIxTHMjOL9vmzn41xrEaOi1XM8rgpRY_--NmK50"  # If required by the proxy; otherwise remove
COLLECTION_NAME = "forum_posts"

qdrant_client = QdrantClient(url="http://localhost:6333")

def get_embedding_via_proxy(text: str) -> list[float]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"

    }
    # If your proxy requires API key authentication, include it here:
    # headers["Authorization"] = f"Bearer {API_KEY}"

    json_payload = {
        "model": "text-embedding-ada-002",
        "input": text
    }

    with httpx.Client() as client:
        response = client.post(API_URL, json=json_payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    embedding = data["data"][0]["embedding"]
    return embedding

def search_similar_documents(query: str, top_k: int = 5):
    print("📥 Embedding query via proxy...")
    embedding = get_embedding_via_proxy(query)

    print("🔍 Searching Qdrant collection...")
    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k
    )

    print(f"✅ Found {len(hits)} hits.")
    return hits


if __name__ == "__main__":
    query = "What to do if peer has not allowed access and the deadline is over for peer review in Project 2"
    results = search_similar_documents(query)

    for hit in results:
        print(f"Score: {hit.score}, Payload: {hit.payload}")
