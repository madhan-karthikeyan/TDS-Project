import os
import httpx
from typing import List, Optional, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
from PIL import Image
import pytesseract
import json
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class CustomEmbeddings(Embeddings):
    def __init__(self, api_url: str, bearer_token: str, model: str = "text-embedding-ada-002"):
        self.api_url = api_url
        self.bearer_token = bearer_token
        self.model = model
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": texts,
            "model": self.model
        }
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings
        except httpx.RequestError as e:
            raise Exception(f"Embedding API request failed: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"Embedding API returned error {e.response.status_code}: {e.response.text}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]

class CustomChatLLM(LLM):
    api_url: str
    bearer_token: str
    model: str = "gpt-3.5-turbo"
    
    def __init__(self, api_url: str, bearer_token: str, model: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(
            api_url=api_url,
            bearer_token=bearer_token,
            model=model,
            **kwargs
        )
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Call the custom chat API"""
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.RequestError as e:
            raise Exception(f"Chat API request failed: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"Chat API returned error {e.response.status_code}: {e.response.text}")
    
    @property
    def _llm_type(self) -> str:
        return "custom_chat_llm"

# Configuration - Use environment variable or fallback to hardcoded token
BEARER_TOKEN = os.getenv("OPENAI_API_KEY")
if not BEARER_TOKEN:
    raise ValueError("BEARER_TOKEN environment variable not set")

if not BEARER_TOKEN:
    raise ValueError("Please set AIPIPE_BEARER_TOKEN environment variable")

EMBEDDING_API_URL = "https://aipipe.org/openai/v1/embeddings"
CHAT_API_URL = "https://aipipe.org/openai/v1/chat/completions"

# Initialize FastAPI app
app = FastAPI(title="RAG API with OCR", description="A FastAPI application for RAG with OCR capabilities")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

# Global variables for embeddings and LLM
custom_embeddings = None
custom_llm = None
vectorstore = None

@app.on_event("startup")
async def startup_event():
    """Initialize embeddings, LLM, and vector store after app startup"""
    global custom_embeddings, custom_llm, vectorstore
    
    try:
        custom_embeddings = CustomEmbeddings(
            api_url=EMBEDDING_API_URL,
            bearer_token=BEARER_TOKEN
        )
        
        custom_llm = CustomChatLLM(
            api_url=CHAT_API_URL,
            bearer_token=BEARER_TOKEN
        )
        
        # Initialize vector store
        if os.path.exists("vector_db"):
            vectorstore = Chroma(
                persist_directory="vector_db", 
                embedding_function=custom_embeddings
            )
            print("Vector store loaded successfully")
        else:
            print("Warning: vector_db directory not found. Vector store will be empty.")
            vectorstore = Chroma(
                persist_directory="vector_db", 
                embedding_function=custom_embeddings
            )
            
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG API with OCR",
        "endpoints": {
            "POST /api/": "Submit a question with optional image for OCR",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "vectorstore_initialized": vectorstore is not None}

@app.get("/debug/db-contents")
async def debug_db_contents():
    """Debug endpoint to check what's in the vector database"""
    if not vectorstore:
        return {"error": "Vector store not initialized"}
    
    try:
        # Get all documents (or a sample)
        all_docs = vectorstore.get()
        
        total_docs = len(all_docs['ids']) if all_docs['ids'] else 0
        
        # Check metadata of first few documents
        sample_metadata = []
        if all_docs['metadatas']:
            for i, metadata in enumerate(all_docs['metadatas'][:10]):  # First 10
                sample_metadata.append({
                    'index': i,
                    'metadata': metadata,
                    'has_url': 'url' in metadata if metadata else False,
                    'content_preview': all_docs['documents'][i][:100] if all_docs['documents'] else "No content"
                })
        
        # Count documents with URLs
        url_count = 0
        discourse_count = 0
        if all_docs['metadatas']:
            for metadata in all_docs['metadatas']:
                if metadata and 'url' in metadata:
                    url_count += 1
                if metadata and metadata.get('url', '').find('discourse') != -1:
                    discourse_count += 1
        
        return {
            "total_documents": total_docs,
            "documents_with_urls": url_count,
            "discourse_documents": discourse_count,
            "sample_metadata": sample_metadata
        }
    
    except Exception as e:
        return {"error": str(e)}
    
# Temporary debug version - replace your existing /api/ endpoint:

def enhance_question_for_retrieval(question: str, ocr_text: str = "") -> str:
    """Enhance the question with key terms for better retrieval"""
    enhanced = question
    
    # Add OCR text if available
    if ocr_text.strip():
        enhanced += f"\n\nAdditional context from image: {ocr_text}"
    
    # Extract key terms that might help retrieval
    key_terms = []
    question_lower = question.lower()
    
    if "gpt" in question_lower:
        key_terms.append("GPT model")
    if "api" in question_lower:
        key_terms.append("API")
    if "cost" in question_lower or "token" in question_lower:
        key_terms.append("cost calculation tokens")
    if "proxy" in question_lower:
        key_terms.append("AI proxy")
    
    if key_terms:
        enhanced += f"\n\nKey topics: {', '.join(key_terms)}"
    
    return enhanced


@app.api_route("/api/", methods=["GET", "POST", "OPTIONS"])
async def get_answer(query_request: QueryRequest):
    """Process question with optional OCR from image - DEBUG VERSION"""
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        question = query_request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        print(f"DEBUG: Original question: {question}")
        
        # Process OCR if image is provided
        if query_request.image:
            try:
                # Remove data URL prefix if present
                image_data = query_request.image
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                # Decode and process image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Perform OCR
                ocr_text = pytesseract.image_to_string(image).strip()
                print(f"DEBUG: OCR extracted text: {ocr_text}")
                if ocr_text:
                    question += f"\n\nExtracted text from image:\n{ocr_text}"
                else:
                    print("Warning: No text extracted from image")
                    
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"OCR processing failed: {str(e)}")

        print(f"DEBUG: Final question for retrieval: {question}")

        # Set up retriever with more documents
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}  # Increased from 10 to get more sources
        )
                # Get relevant documents manually first
        relevant_docs = retriever.get_relevant_documents(question)
        print(f"DEBUG: Retrieved {len(relevant_docs)} documents")
        
        for i, doc in enumerate(relevant_docs):
            print(f"DEBUG Doc {i}: {doc.page_content[:100]}...")
            print(f"DEBUG Doc {i} metadata: {doc.metadata}")
            print(f"DEBUG Doc {i} has URL: {'url' in doc.metadata}")
            
        # Continue with QA chain...
        prompt_template = """You are a helpful and precise teaching assistant for the IIT Madras Tools in Data Science (TDS) course.

        Use the following context to answer the student's question. The context includes:
        1. Official course content from https://tds.s-anand.net
        2. Forum discussions from the course Discourse site

        INSTRUCTIONS:
        - Always prioritize official course material (https://tds.s-anand.net) if relevant content is available
        - Use forum content only if the course material does not answer the question
        - If the student asks about scores, dashboard marks, or bonus marks:
        ➤ Convert to percentage out of 100 and apply bonus accordingly (e.g., 10/10 with bonus = 110)
        - If the answer requires referencing models, tools (e.g., Docker), scoring rules, or exams, base your answer strictly on what is stated in the context
        - If the answer is not clearly found in the provided context, say: **"I don't know based on the course content."**
        - NEVER guess or infer beyond what's in the context
        - DO NOT include URLs in the answer
        - Keep the answer brief, direct, and factually correct

        Context:
        {context}

        Student Question: {question}

        Answer:"""



        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )


        qa = RetrievalQA.from_chain_type(
            llm=custom_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        enhanced_question = enhance_question_for_retrieval(question, ocr_text if query_request.image else "")
        result = qa.invoke({"query": enhanced_question})
        sources = result["source_documents"]
        try:
            parsed_answer = json.loads(result["result"])
            if isinstance(parsed_answer, dict) and "answer" in parsed_answer:
                answer_text = parsed_answer["answer"]
                answer_links = parsed_answer.get("links", [])
            else:
                answer_text = result["result"]
                answer_links = []
        except json.JSONDecodeError:
            answer_text = result["result"]
            answer_links = []

        # Merge links from context with parsed links
        combined_links = answer_links.copy()
        for doc in sources:
            metadata = doc.metadata
            url = metadata.get("url") or metadata.get("original_url", "")
            title = metadata.get("title", "") or metadata.get("source", "").split("/")[-1].replace(".md", "")
            if url and not any(link["url"] == url for link in combined_links):
                combined_links.append({
                    "url": url,
                    "text": title or "Related content"
                })

        return {
            "answer": answer_text.strip(),
            "links": combined_links
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"DEBUG: Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/")
async def get_api_info():
    """GET endpoint for API information"""
    return {
        "message": "RAG API with OCR",
        "method": "POST",
        "body_format": {
            "question": "Your question here (required)",
            "image": "Base64 encoded image (optional)"
        },
        "example": {
            "question": "What is machine learning?",
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)