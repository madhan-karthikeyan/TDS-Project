import logging
import time
from json_processor import ForumPostProcessor

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()

    print("🔄 Initializing processor...")
    processor = ForumPostProcessor(use_openai_embeddings=True)  # Set False to use local embeddings

    print("📦 Creating collection in Qdrant (if not exists)...")
    processor.create_collection(collection_name="forum_posts")

    print("📁 Loading and processing forum posts from JSON...")
    posts = processor.extract_forum_posts("data/discourse_posts.json")
    print(f"✅ Loaded and processed {len(posts)} posts.")

    print("🚀 Uploading embeddings to Qdrant...")
    processor.upload_to_qdrant(posts)
    print("✅ Upload complete.")

    duration = time.time() - start_time
    print(f"🏁 All done in {duration:.2f} seconds.")
