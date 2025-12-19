"""
Script to check Qdrant 'topics' collection and sync data from MongoDB
"""
import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.mongodb_service import MongoDBService
from services.qdrant_service import QdrantService
from services.topic_service import TopicService

async def sync_topics():
    print("=" * 60)
    print("Checking Qdrant 'topics' Collection")
    print("=" * 60)
    
    # Initialize services
    # Note: TopicService uses its own sync connection now
    topic_service = TopicService()
    
    # Initialize Qdrant for 'topics'
    qdrant = QdrantService(collection_name="topics")
    
    # 1. Force recreate collection to ensure correct vector size
    print("   Forcing recreation of collection with correct vector size...")
    await qdrant.create_collection(force_recreate=True)
    print("   ✓ Collection recreated")

    # 2. Fetch all topics from MongoDB
    print("\nFetching topics from MongoDB...")
    topics = topic_service.list_topics(limit=1000)
    print(f"✓ Found {len(topics)} topics in MongoDB")
    
    if not topics:
        print("⚠ No topics to sync!")
        return

    # 3. Prepare for Qdrant
    points = []
    print("\nPreparing data for Qdrant...")
    
    for i, topic in enumerate(topics):
        if i % 100 == 0:
            print(f"   Processed {i}/{len(topics)} topics")
        # Create text for embedding
        # Combine title, description, and tags for better semantic search
        text = f"{topic.get('title', '')}. {topic.get('description', '')}. Tags: {', '.join(topic.get('tags', []))}"
        
        # Use topic_id as payload id
        t_id = topic.get('topic_id')
        
        # Qdrant requires point ID to be int or UUID. 
        # We generate a deterministic UUID from the topic_id string.
        import uuid
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(t_id)))
        
        points.append({
            "id": point_id,
            "name": text, # QdrantService uses 'name' to generate embedding
            "topic_id": t_id,
            "title": topic.get('title'),
            "department": topic.get('department'),
            "option": topic.get('option'),
            "year": topic.get('year'),
            "status": topic.get('status')
        })
        
    # 4. Upload to Qdrant
    print(f"Syncing {len(points)} topics to Qdrant...")
    await qdrant.upload_points(points)
    
    print("\n✓ Sync completed!")
    
    # 5. Verify count
    info = await qdrant.get_collection_info()
    print(f"Final Qdrant points count: {info.points_count}")

if __name__ == "__main__":
    asyncio.run(sync_topics())
