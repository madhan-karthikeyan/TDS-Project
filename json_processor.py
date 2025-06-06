from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any
from config import Config
from embedding_service import EmbeddingService

class ForumPostProcessor:
    def __init__(self, use_openai_embeddings: bool = True):
        self.config = Config()
        self.embedding_service = EmbeddingService(use_openai=use_openai_embeddings)
        
        # Initialize Qdrant client
        if self.config.QDRANT_API_KEY:
            self.client = QdrantClient(
                url=f"https://your-cluster-url.qdrant.tech",
                api_key=self.config.QDRANT_API_KEY
            )
        else:
            self.client = QdrantClient(
                host=self.config.QDRANT_HOST,
                port=self.config.QDRANT_PORT
            )
        
        logging.info(f"Initialized with embedding dimension: {self.embedding_service.embedding_dim}")
    
    def create_collection(self, collection_name: str):
        """Create a new collection in Qdrant for forum posts"""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_service.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logging.info(f"Collection '{collection_name}' created successfully")
        except Exception as e:
            logging.warning(f"Collection might already exist: {e}")
    
    def extract_forum_posts(self, json_file_path: str) -> List[Dict]:
        """Extract forum posts from JSON file based on the provided schema"""
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Handle both single post and array of posts
        posts = data if isinstance(data, list) else [data]

        extracted_posts = []

        for post in posts:
            if not self._validate_post_schema(post):
                logging.warning(f"Skipping invalid post: {post.get('post_id', 'unknown')}")
                continue

            # Construct embedding text
            embedding_text = self._create_embedding_text(post)

            # Only the required fields + embedding_text
            processed_post = {
                'post_id': post['post_id'],
                'topic_id': post['topic_id'],
                'topic_title': post['topic_title'],
                'category_id': post['category_id'],
                'tags': post['tags'],
                'content': post['content'],
                'url': post['url'],
                'embedding_text': embedding_text
            }

            extracted_posts.append(processed_post)

        logging.info(f"Extracted {len(extracted_posts)} valid forum posts")
        return extracted_posts

    
    def _validate_post_schema(self, post: Dict) -> bool:
        """Validate post against the required schema fields"""
        required_fields = [
            'topic_id', 'topic_title', 'category_id', 'tags', 'post_id', 'url', 'content'
        ]
        
        for field in required_fields:
            if field not in post:
                logging.error(f"Missing required field: {field}")
                return False
        
        return True
    
    def _create_embedding_text(self, post: Dict) -> str:
        """Create a unified text string from all fields in post for embedding generation."""
        parts = [
            f"Post ID: {post['post_id']}",
            f"Topic ID: {post['topic_id']}",
            f"Topic Title: {post['topic_title']}",
            f"Category ID: {post['category_id']}",
            f"Tags: {', '.join(post['tags']) if isinstance(post['tags'], list) else post['tags']}",
            f"Content: {post['content']}",
            f"URL: {post['url']}"
        ]
        return " | ".join(parts)
    def upload_to_qdrant(self, posts: List[Dict[str, Any]]):
        """Generate embeddings and upload post data to Qdrant"""
        texts = [post['embedding_text'] for post in posts]
        embeddings = self.embedding_service.generate_embeddings(texts)

        # Prepare points
        points = []
        for i, post in enumerate(posts):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={
                    "post_id": post["post_id"],
                    "topic_id": post["topic_id"],
                    "topic_title": post["topic_title"],
                    "category_id": post["category_id"],
                    "tags": post["tags"],
                    "content": post["content"],
                    "url": post["url"]
                }
            )
            points.append(point)

        self.client.upsert(collection_name=self.config.COLLECTION_NAME, points=points)
        logging.info(f"Uploaded {len(points)} points to collection '{self.config.COLLECTION_NAME}'")


    def upload_posts_to_qdrant(self, collection_name: str, posts: List[Dict], batch_size: int = 0):
        """Upload forum posts with embeddings to Qdrant"""
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        logging.info(f"Generating embeddings for {len(posts)} forum posts...")
        
        # Extract texts for embedding
        texts = [post['embedding_text'] for post in posts]
        embeddings = self.embedding_service.generate_embeddings(texts)
        
        # Upload in batches
        total_batches = (len(posts) - 1) // batch_size + 1
        
        for i in range(0, len(posts), batch_size):
            batch_posts = posts[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            points = []
            for post, embedding in zip(batch_posts, batch_embeddings):
                point = PointStruct(
                    id=post['id'],
                    vector=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    payload={
                        'post_id': post['post_id'],
                        'topic_id': post['topic_id'],
                        'topic_title': post['topic_title'],
                        'category_id': post['category_id'],
                        'tags': post['tags'],
                        # 'author': post['author'],
                        'content': post['content'],  # Truncate for storage efficiency
                        # 'full_content': post['content'],  # Keep full content
                        # 'created_at': post['created_at'],
                        # 'updated_at': post['updated_at'],
                        'post_number': post['post_number'],
                        # 'is_reply': post['is_reply'],
                        # 'reply_to_post_number': post['reply_to_post_number'],
                        # 'reply_count': post['reply_count'],
                        # 'like_count': post['like_count'],
                        # 'is_accepted_answer': post['is_accepted_answer'],
                        'url': post['url'],
                        # 'processed_at': post['processed_at']
                    }
                )
                points.append(point)
            
            # Upload batch
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logging.info(f"Uploaded batch {i//batch_size + 1}/{total_batches}")
        
        logging.info(f"Successfully uploaded {len(posts)} forum posts to Qdrant!")
    
    def search_posts(self, collection_name: str, query: str, limit: int = 5, category_filter: int = None, include_replies: bool = True):
        """Search for similar forum posts"""
        # Generate embedding for query
        query_embedding = self.embedding_service.generate_single_embedding(query)
        
        # Build search filters
        search_filter = None
        if category_filter is not None or not include_replies:
            conditions = []
            
            if category_filter is not None:
                conditions.append({
                    "key": "category_id",
                    "match": {"value": category_filter}
                })
            
            if not include_replies:
                conditions.append({
                    "key": "is_reply",
                    "match": {"value": False}
                })
            
            if conditions:
                search_filter = {
                    "must": conditions
                }
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter,
            with_payload=True
        )
        
        return search_results
    
    def get_post_by_id(self, collection_name: str, post_id: int):
        """Get a specific post by its post_id"""
        search_results = self.client.scroll(
            collection_name=collection_name,
            scroll_filter={
                "must": [
                    {
                        "key": "post_id",
                        "match": {"value": post_id}
                    }
                ]
            },
            limit=1,
            with_payload=True
        )
        
        return search_results[0] if search_results[0] else None
    
    def get_collection_stats(self, collection_name: str):
        """Get statistics about the collection"""
        try:
            info = self.client.get_collection(collection_name)
            return {
                'points_count': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance_metric': info.config.params.vectors.distance
            }
        except Exception as e:
            logging.error(f"Error getting collection stats: {e}")
            return None