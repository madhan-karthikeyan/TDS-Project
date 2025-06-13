import os
import json
import httpx
import numpy as np
from typing import List, Dict, Any
import re
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma

class CustomEmbeddings(Embeddings):
    def __init__(self, api_url: str, bearer_token: str, model: str = "text-embedding-3-small", batch_size: int = 15):
        self.api_url = api_url
        self.bearer_token = bearer_token
        self.model = model
        self.batch_size = batch_size
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with batching and retry logic"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i//self.batch_size + 1
            total_batches = (len(texts) + self.batch_size - 1)//self.batch_size
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
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
                    with httpx.Client(timeout=60.0) as client:
                        response = client.post(self.api_url, headers=headers, json=payload)
                        response.raise_for_status()
                        
                        data = response.json()
                        batch_embeddings = [item["embedding"] for item in data["data"]]
                        all_embeddings.extend(batch_embeddings)
                        break
                        
                except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    print(f"Timeout on batch {batch_num}, retry {retry + 1}/{max_retries}")
                    if retry == max_retries - 1:
                        raise e
                    import time
                    time.sleep(2 ** retry)
                    
                except httpx.HTTPStatusError as e:
                    print(f"HTTP error: {e.response.status_code} - {e.response.text}")
                    raise e
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    if retry == max_retries - 1:
                        raise e
                    import time
                    time.sleep(2 ** retry)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]

def parse_frontmatter_regex(content: str) -> tuple[Dict[str, str], str]:
    """Parse YAML frontmatter using regex fallback"""
    metadata = {}
    pattern = r"---\s*\n(.*?)\n---\s*\n(.*)"
    match = re.match(pattern, content, re.DOTALL)

    if match:
        frontmatter, body = match.groups()
        for line in frontmatter.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip('"').strip("'")
        return metadata, body

    # Try extracting metadata from inline text if frontmatter not found
    # Handle case where title and URL are on the same line
    title_url_match = re.search(r"title:\s*(.*?)\s+original_url:\s*(.*?)(?:\s+downloaded_at:.*)?(?:\n|$)", content)
    
    if title_url_match:
        title_part = title_url_match.group(1).strip()
        url_part = title_url_match.group(2).strip()
        
        metadata["title"] = title_part
        metadata["original_url"] = clean_url(url_part)
        
        # Remove the entire metadata line from content
        content = re.sub(r"title:.*?(?:downloaded_at:.*?)?(?:\n|$)", "", content, flags=re.DOTALL)
    else:
        # Fallback to individual extraction
        fallback_title = re.search(r"title:\s*(.*?)(?:\s+original_url:|$)", content)
        fallback_url = re.search(r"original_url:\s*(.*?)(?:\s+downloaded_at:|$)", content)
        
        if fallback_title:
            metadata["title"] = fallback_title.group(1).strip()
        else:
            metadata["title"] = "Untitled"
        
        if fallback_url:
            metadata["original_url"] = clean_url(fallback_url.group(1).strip())
        else:
            metadata["original_url"] = ""
        
        # Remove metadata lines from content
        content = re.sub(r"title:\s*.*?(?=\n|$)", "", content)
        content = re.sub(r"original_url:\s*.*?(?=\n|$)", "", content)
    
    # Remove other metadata lines like downloaded_at
    content = re.sub(r"downloaded_at:\s*.*?(?=\n|$)", "", content)
    
    return metadata, content

def clean_url(url: str) -> str:
    """Clean and shorten URL by removing query parameters and downloaded_at info"""
    # Remove downloaded_at info if it's part of the URL string
    url = re.sub(r"\s+downloaded_at:.*", "", url)
    
    # Remove query parameters starting with ?id=
    url = re.sub(r"\?id=.*", "", url)
    
    # Remove other common query parameters if needed
    url = re.sub(r"\?.*", "", url)
    
    return url.strip()

def clean_markdown(content: str) -> str:
    content = re.sub(r"Copy to clipboardErrorCopied\\n?", "", content)
    content = re.sub(r"\\n\s*\\n\s*\\n", "\n\n", content)
    return content.strip()

def chunk_markdown(content: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )
    return [chunk.page_content for chunk in splitter.create_documents([content])]

# Configuration
BEARER_TOKEN = os.getenv("OPENAI_API_KEY")
if not BEARER_TOKEN:
    raise ValueError("BEARER_TOKEN environment variable not set")

print("🔄 Loading markdown files...")
loader = DirectoryLoader("markdown_files", glob="**/*.md")
markdown_docs = loader.load()

# Process markdown files and create JSON payloads
payloads = []
course_chunks = []

for doc in markdown_docs:
    print(f"Processing: {doc.metadata.get('source', 'unknown')}")
    metadata, body = parse_frontmatter_regex(doc.page_content)
    cleaned = clean_markdown(body)
    chunks = chunk_markdown(cleaned)

    # Create payloads for JSON output
    for chunk in chunks:
        payload = {
            "title": metadata.get("title", "Untitled"),
            "url": metadata.get("original_url", ""),
            "content": chunk
        }
        print(payload)
        payloads.append(payload)
    
    # Create Document objects for vector DB
    for chunk in chunks:
        chunk_metadata = {
            "title": metadata.get("title", "Untitled"),
            "url": metadata.get("original_url", ""),
            "content_type": "course_material",
            "source": doc.metadata.get('source', 'unknown')
        }
        course_chunks.append(Document(
            page_content=chunk,
            metadata=chunk_metadata
        ))

print(f"✅ Created {len(payloads)} content payloads from markdown files")

# Save JSON payloads
with open("course_chunks_payload.json", "w") as f:
    json.dump(payloads, f, indent=2)

print("📄 Payloads saved to course_chunks_payload.json")

print("🔄 Loading discourse posts...")

# Load discourse JSON (keep as-is)
discourse_chunks = []
try:
    with open("downloaded_threads/discourse_posts.json", "r") as f:
        posts = json.load(f)
    
    for post in posts:
        if post.get("content", "").strip():
            metadata = {
                "url": post.get("url", ""),
                "title": post.get("topic_title", ""),
                "author": post.get("author", ""),
                "created_at": post.get("created_at", ""),
                "content_type": "forum_discussion",
                "source": "discourse"
            }
            discourse_chunks.append(Document(
                page_content=post["content"].strip(),
                metadata=metadata
            ))
    
    print(f"✅ Loaded {len(discourse_chunks)} discourse posts")
    
except FileNotFoundError:
    print("⚠️  discourse_posts.json not found, skipping forum content")
    discourse_chunks = []

# Combine all documents
all_documents = course_chunks + discourse_chunks

print(f"✅ Total documents: {len(all_documents)}")

# Create embeddings
print("🔄 Creating embeddings...")

embedding = CustomEmbeddings(
    api_url="https://aipipe.org/openai/v1/embeddings",
    bearer_token=BEARER_TOKEN,
    model="text-embedding-3-small",
    batch_size=10
)

# Build vector database
print("🔄 Building ChromaDB...")

# Remove existing db if it exists
if os.path.exists("vector_db"):
    import shutil
    shutil.rmtree("vector_db")

db = Chroma.from_documents(
    all_documents, 
    embedding, 
    persist_directory="vector_db",
    collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

db.persist()

# Print summary
course_count = len([c for c in all_documents if c.metadata.get('content_type') == 'course_material'])
forum_count = len([c for c in all_documents if c.metadata.get('content_type') == 'forum_discussion'])

print("✅ Vector DB built successfully!")
print(f"📊 Summary:")
print(f"  - Course material chunks: {course_count}")
print(f"  - Forum discussion chunks: {forum_count}")
print(f"  - Total chunks: {len(all_documents)}")
print(f"  - Database saved to: vector_db/")

# Test the database
print("\n🧪 Testing the database...")
test_query = "What is the LLM usage allowance for students?"
results = db.similarity_search(test_query, k=3)
print(f"Test query: '{test_query}'")
print(f"Found {len(results)} relevant chunks")
for i, result in enumerate(results[:2]):
    print(f"  Result {i+1}: {result.page_content[:100]}...")