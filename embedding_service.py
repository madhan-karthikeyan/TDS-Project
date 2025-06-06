import httpx
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
import asyncio
from config import Config

class EmbeddingService:
    def __init__(self, use_openai: bool = True):
        self.use_openai = use_openai
        self.config = Config()
        
        if use_openai:
            self.setup_openai()
        else:
            self.setup_local()
    
    def setup_openai(self):
        """Setup OpenAI client with httpx"""
        self.api_key = self.config.OPENAI_API_KEY
        self.base_url = "https://aipipe.org/openai/v1/embeddings"
        self.embedding_dim = self.config.EMBEDDING_DIMENSION
        logging.info(f"Using OpenAI embeddings via {self.base_url}")
    
    def setup_local(self):
        """Setup local sentence transformer model"""
        self.model = SentenceTransformer(self.config.LOCAL_EMBEDDING_MODEL)
        self.embedding_dim = self.config.LOCAL_EMBEDDING_DIMENSION
        logging.info(f"Using local model: {self.config.LOCAL_EMBEDDING_MODEL}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self.use_openai:
            return self._generate_openai_embeddings(texts, batch_size)
        else:
            return self._generate_local_embeddings(texts)
    
    def _generate_openai_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using OpenAI API via httpx"""
        all_embeddings = []
        
        with httpx.Client(timeout=30.0) as client:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "model": self.config.EMBEDDING_MODEL,
                        "input": batch_texts
                    }
                    
                    response = client.post(
                        self.base_url,
                        json=payload,
                        headers=headers
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    batch_embeddings = [item['embedding'] for item in data['data']]
                    all_embeddings.extend(batch_embeddings)
                    
                    logging.info(f"Generated embeddings for batch {i//batch_size + 1}")
                    
                except httpx.HTTPError as e:
                    logging.error(f"HTTP error for batch {i//batch_size + 1}: {e}")
                    raise
                except Exception as e:
                    logging.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                    raise
        
        return np.array(all_embeddings)
    
    def _generate_local_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local model"""
        return self.model.encode(texts, show_progress_bar=True)
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if self.use_openai:
            with httpx.Client(timeout=30.0) as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.config.EMBEDDING_MODEL,
                    "input": [text]
                }
                
                response = client.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
                data = response.json()
                return data['data'][0]['embedding']
        else:
            return self.model.encode([text])[0].tolist()

    # Async version for better performance with large batches
    async def generate_embeddings_async(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings asynchronously for better performance"""
        if not self.use_openai:
            return self._generate_local_embeddings(texts)
        
        all_embeddings = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                task = self._generate_batch_async(client, batch_texts, i//batch_size + 1)
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks)
            
            for batch_embeddings in batch_results:
                all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    async def _generate_batch_async(self, client: httpx.AsyncClient, texts: List[str], batch_num: int) -> List[List[float]]:
        """Generate embeddings for a single batch asynchronously"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.EMBEDDING_MODEL,
            "input": texts
        }
        
        try:
            response = await client.post(
                self.base_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            batch_embeddings = [item['embedding'] for item in data['data']]
            
            logging.info(f"Generated embeddings for batch {batch_num}")
            return batch_embeddings
            
        except httpx.HTTPError as e:
            logging.error(f"HTTP error for batch {batch_num}: {e}")
            raise
        except Exception as e:
            logging.error(f"Error generating embeddings for batch {batch_num}: {e}")
            raise