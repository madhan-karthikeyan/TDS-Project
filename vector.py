import os
import json
import httpx
import numpy as np
from typing import List
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma

class CustomEmbeddings(Embeddings):
    def __init__(self, api_url: str, bearer_token: str, model: str = "text-embedding-ada-002", batch_size: int = 20):
        self.api_url = api_url
        self.bearer_token = bearer_token
        self.model = model
        self.batch_size = batch_size
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with batching"""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size} ({len(batch)} texts)")
            
            headers = {
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": batch,
                "model": self.model
            }
            
            max_retries = 3
            for retry in range(max_retries):
                try:
                    with httpx.Client(timeout=60.0) as client:  # 60 second timeout
                        response = client.post(self.api_url, headers=headers, json=payload)
                        response.raise_for_status()
                        
                        data = response.json()
                        batch_embeddings = [item["embedding"] for item in data["data"]]
                        all_embeddings.extend(batch_embeddings)
                        break
                        
                except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    print(f"Timeout on batch {i//self.batch_size + 1}, retry {retry + 1}/{max_retries}")
                    if retry == max_retries - 1:
                        raise e
                    import time
                    time.sleep(2 ** retry)  # Exponential backoff
                    
                except httpx.HTTPStatusError as e:
                    print(f"HTTP error: {e.response.status_code} - {e.response.text}")
                    raise e
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]

# Configuration - set your bearer token here
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6Im1hZGhhbmtydGhpa0BnbWFpbC5jb20ifQ.ko-Vbxf1Bc9WaDxUvr7CdcRdzh-RGroru_t8LmE02eA" # Set this environment variable
if not BEARER_TOKEN:
    raise ValueError("Please set AIPIPE_BEARER_TOKEN environment variable")

# Step 1: Load markdown files
loader = DirectoryLoader("markdown_files", glob="**/*.md")
markdown_docs = loader.load()

# Step 2: Parse frontmatter metadata
for doc in markdown_docs:
    lines = doc.page_content.splitlines()
    metadata = {}
    
    # Check for frontmatter with --- delimiters
    if lines and lines[0].strip() == "---":
        i = 1
        while i < len(lines) and lines[i].strip() != "---":
            line = lines[i].strip()
            if ":" in line:
                # Handle lines that might have multiple colons (like URLs)
                parts = line.split(":", 1)  # Split only on first colon
                key = parts[0].strip()
                val = parts[1].strip().strip('"').strip("'")  # Remove quotes if present
                metadata[key] = val
                print(f"DEBUG: Found metadata - {key}: {val}")
            i += 1
        
        # Update document metadata
        doc.metadata.update(metadata)
        
        # Remove frontmatter from content
        if i < len(lines):
            doc.page_content = "\n".join(lines[i+1:])
        else:
            doc.page_content = ""
    else:
        # Handle frontmatter without --- delimiters (appears to be your case)
        # Look for lines that start with "title:" or "original_url:" at the beginning
        content_start = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith(("title:", "original_url:", "downloaded_at:")):
                if ":" in line:
                    parts = line.split(":", 1)
                    key = parts[0].strip()
                    val = parts[1].strip().strip('"').strip("'")
                    metadata[key] = val
                    print(f"DEBUG: Found metadata - {key}: {val}")
                    content_start = i + 1
            elif line and not line.startswith(("title:", "original_url:", "downloaded_at:")):
                # Found content, stop looking for metadata
                break
        
        # Update document metadata
        doc.metadata.update(metadata)
        
        # Remove metadata lines from content
        if content_start > 0:
            doc.page_content = "\n".join(lines[content_start:])
            
    print(f"DEBUG: Document {doc.metadata.get('source', 'unknown')} metadata: {doc.metadata}")

print(f"DEBUG: Processed {len(markdown_docs)} markdown documents")

# Step 3: Load discourse JSON
with open("downloaded_threads/discourse_posts.json", "r") as f:
    posts = json.load(f)

discourse_docs = []
for post in posts:
    metadata = {
        "url": post["url"],
        "title": post.get("topic_title", ""),
        "author": post.get("author"),
        "created_at": post.get("created_at"),
    }
    discourse_docs.append(Document(page_content=post["content"], metadata=metadata))

# Step 4: Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_chunks = splitter.split_documents(markdown_docs + discourse_docs)

print(f"Total chunks to embed: {len(all_chunks)}")

# Step 5: Custom embedding using httpx with batching
embedding = CustomEmbeddings(
    api_url="https://aipipe.org/openai/v1/embeddings",
    bearer_token=BEARER_TOKEN,
    batch_size=10  # Process 10 documents at a time
)

# Step 6: Store in ChromaDB
db = Chroma.from_documents(all_chunks, embedding, persist_directory="vector_db")
db.persist()

print("✅ Vector DB built successfully.")