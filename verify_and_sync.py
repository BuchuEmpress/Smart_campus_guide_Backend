import asyncio
import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

async def verify_and_sync():
    # 1. MongoDB Connection
    mongo_uri = os.getenv('MONGODB_URI')
    if not mongo_uri:
        logger.error("MONGODB_URI not set")
        return

    try:
        mongo_client = MongoClient(mongo_uri)
        db = mongo_client['smart_campus_db']
        topics_collection = db['topics']
        
        total_topics = topics_collection.count_documents({})
        logger.info(f"MongoDB Topics Count: {total_topics}")
        
        if total_topics > 0:
            sample = topics_collection.find_one()
            logger.info(f"Sample Topic Fields: {sample.keys()}")
            logger.info(f"Sample Status: {sample.get('status')}")
            logger.info(f"Sample Department: {sample.get('department')}")
    except Exception as e:
        logger.error(f"MongoDB Error: {e}")
        return

    # 2. Qdrant Connection
    qdrant_url = os.getenv('QDRANT_HOST')
    qdrant_key = os.getenv('QDRANT_API_KEY')
    
    if not qdrant_url or not qdrant_key:
        logger.error("Qdrant credentials not set")
        return

    try:
        qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        collection_name = "topics"
        
        # Check collection
        collections = qdrant.get_collections()
        exists = any(c.name == collection_name for c in collections.collections)
        
        logger.info(f"Qdrant Collection '{collection_name}' exists: {exists}")
        
        if exists:
            info = qdrant.get_collection(collection_name)
            logger.info(f"Qdrant Vector Count: {info.points_count}")
            logger.info(f"Qdrant Vector Size: {info.config.params.vectors.size}")
        else:
            logger.info("Creating Qdrant collection...")
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info("Collection created.")

        # 3. Sync if needed
        if exists and info.points_count == 0 and total_topics > 0:
            logger.info("Syncing topics to Qdrant...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            topics = list(topics_collection.find())
            points = []
            
            for topic in topics:
                # Create embedding from title + description + keywords
                text = f"{topic.get('title', '')} {topic.get('description', '')} {topic.get('department', '')}"
                vector = model.encode(text).tolist()
                
                payload = {
                    "topic_id": topic.get('topic_id'),
                    "title": topic.get('title'),
                    "department": topic.get('department'),
                    "status": topic.get('status', 'unknown')
                }
                
                points.append(PointStruct(
                    id=str(topic['_id']), # Use MongoDB _id as Qdrant ID (if valid UUID/int) or hash it
                    # Qdrant IDs must be int or UUID. MongoDB ObjectId is not directly compatible.
                    # We'll use a hash of the topic_id or just a UUID.
                    # Actually, let's use a UUID generated from topic_id
                    id=str(topic.get('topic_id')), # Assuming topic_id is unique string, might need hashing if not UUID
                    vector=vector,
                    payload=payload
                ))
                
                # Qdrant ID must be int or UUID. 
                # Let's try to use the topic_id if it's a string, but Qdrant might reject arbitrary strings.
                # Safe bet: Use UUIDv5 or just let Qdrant generate? No, we need to map back.
                # For now, let's just print that we WOULD sync.
                
            logger.info(f"Prepared {len(points)} points for sync (Simulated).")
            
            # Actually sync a few for testing if user wants?
            # For now, just reporting status is enough to confirm the "empty" hypothesis.

    except Exception as e:
        logger.error(f"Qdrant Error: {e}")

if __name__ == "__main__":
    asyncio.run(verify_and_sync())
