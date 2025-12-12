"""
Upload Topics to Qdrant Vector Database - FIXED VERSION

Generates embeddings for topics and uploads them to a dedicated collection.
Memory-optimized with proper error handling.

USAGE:
    python scripts/upload_topics_to_qdrant.py
"""

import json
import os
import sys
import asyncio
import gc
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

load_dotenv()


class TopicQdrantUploader:
    """Handles topic embedding generation and Qdrant upload."""
    
    def __init__(self):
        """Initialize uploader with Qdrant client."""
        # Qdrant setup
        self.qdrant_url = os.getenv("QDRANT_HOST")
        self.qdrant_key = os.getenv("QDRANT_API_KEY")
        
        if not self.qdrant_url or not self.qdrant_key:
            raise ValueError("Missing QDRANT_HOST or QDRANT_API_KEY in .env")
        
        self.client = AsyncQdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_key
        )
        
        self.collection_name = "campus_topics"
        self.vector_size = 384
        
        # Model will be loaded lazily
        self.model = None
        print("✅ Qdrant client initialized!\n")
    
    def _load_model(self):
        """Lazy load embedding model to avoid memory issues."""
        if self.model is None:
            print("🔄 Loading embedding model (this may take a moment)...")
            try:
                # Force garbage collection first
                gc.collect()
                
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Model loaded!\n")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                print("\n💡 Try closing other applications to free memory")
                raise
    
    def load_topics(self, filepath: str = 'data/topics/computer_engineering.json') -> List[Dict]:
        """Load topics from JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Topic file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            topics = json.load(f)
        
        print(f"📂 Loaded {len(topics)} topics from {filepath}")
        return topics
    
    def create_embedding_text(self, topic: Dict) -> str:
        """Create rich text representation for embedding."""
        title = topic.get('title', '')
        description = topic.get('description', '')
        
        tags = topic.get('tags', topic.get('keywords', []))
        tags_text = ', '.join(tags) if tags else ''
        
        skills = topic.get('skills_required', [])
        skills_text = ', '.join(skills) if skills else ''
        
        option = topic.get('option', topic.get('category', ''))
        if isinstance(option, list):
            option_text = ', '.join(option)
        else:
            option_text = option
        
        department = topic.get('department', '')
        
        embedding_text = f"{title}. {description}."
        
        if tags_text:
            embedding_text += f" Tags: {tags_text}."
        
        if skills_text:
            embedding_text += f" Skills: {skills_text}."
        
        if option_text:
            embedding_text += f" Option: {option_text}."
        
        if department:
            embedding_text += f" Department: {department}."
        
        return embedding_text.strip()
    
    async def create_collection(self, force_recreate: bool = True):
        """Create topics collection in Qdrant."""
        exists = await self.client.collection_exists(self.collection_name)
        
        if exists and force_recreate:
            print(f"🗑️  Deleting existing '{self.collection_name}' collection...")
            await self.client.delete_collection(self.collection_name)
        
        if not exists or force_recreate:
            print(f"🆕 Creating '{self.collection_name}' collection...")
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print("✅ Collection created!\n")
        else:
            print(f"ℹ️  Collection '{self.collection_name}' already exists\n")
    
    async def upload_topics(self, topics: List[Dict], batch_size: int = 50):
        """Generate embeddings and upload topics to Qdrant."""
        # Load model now (lazy loading)
        self._load_model()
        
        print(f"🔄 Generating embeddings for {len(topics)} topics...")
        
        # Prepare embedding texts
        embedding_texts = [self.create_embedding_text(topic) for topic in topics]
        
        # Generate embeddings in smaller batches to avoid memory issues
        print("   Processing in batches to optimize memory...")
        embeddings = self.model.encode(
            embedding_texts,
            show_progress_bar=True,
            batch_size=16  # Smaller batch size for stability
        )
        
        print(f"✅ Embeddings generated!\n")
        
        # Free up memory
        gc.collect()
        
        # Prepare points for upload
        print(f"🔄 Preparing {len(topics)} points for upload...")
        points = []
        
        for idx, (topic, embedding) in enumerate(zip(topics, embeddings)):
            original_id = topic.get('id', f"TOPIC_{idx+1:03d}")
            
            # Use integer as Qdrant point ID
            qdrant_point_id = idx + 1
            
            payload = {
                'topic_id': original_id,
                'title': topic.get('title', ''),
                'description': topic.get('description', ''),
                'department': topic.get('department', ''),
                'option': topic.get('option', ''),
                'tags': topic.get('tags', []),
                'skills_required': topic.get('skills_required', []),
                'status': topic.get('status', 'Available'),
                'year': topic.get('year'),
                'type': 'topic',
                'embedding_text': self.create_embedding_text(topic)
            }
            
            point = PointStruct(
                id=qdrant_point_id,
                vector=embedding.tolist(),
                payload=payload
            )
            
            points.append(point)
        
        # Upload in batches
        print(f"☁️  Uploading to Qdrant (batch size: {batch_size})...")
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"   ✅ Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
        
        print(f"\n🎉 Successfully uploaded {len(points)} topics to Qdrant!")
    
    async def verify_upload(self):
        """Verify topics were uploaded correctly."""
        try:
            print("\n🔍 Verifying upload...")
            
            info = await self.client.get_collection(self.collection_name)
            
            print(f"✅ Collection info:")
            print(f"   Points count: {info.points_count}")
            print(f"   Status: {info.status}")
            
            # Simple verification - just check count
            if info.points_count > 0:
                print(f"\n✅ Verification successful! {info.points_count} topics in collection.")
            else:
                print(f"\n⚠️  Warning: Collection appears empty")
                
        except Exception as e:
            print(f"\n⚠️  Verification encountered an issue: {e}")
            print("   Your data is likely uploaded, but verification failed.")
    
    async def run(self, filepath: str = 'data/topics/computer_engineering.json'):
        """Main execution pipeline."""
        try:
            print("="*80)
            print("📚 TOPIC UPLOAD TO QDRANT")
            print("="*80 + "\n")
            
            # Load topics
            topics = self.load_topics(filepath)
            
            # Create collection
            await self.create_collection(force_recreate=True)
            
            # Upload topics
            await self.upload_topics(topics)
            
            # Verify
            await self.verify_upload()
            
            print("\n" + "="*80)
            print("✅ UPLOAD COMPLETE!")
            print("="*80 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Main entry point."""
    uploader = TopicQdrantUploader()
    success = await uploader.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())