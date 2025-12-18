import os
import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.qdrant_service import QdrantService
from services.mongodb_service import MongoDBService
from services.gemini_service import GeminiService
import json

async def revectorize_locations():
    print("\n--- Re-vectorizing Locations ---")
    qdrant = QdrantService(collection_name="campus_locations")
    
    # Load locations
    input_file = 'data/locations/campus_locations.json'
    if not os.path.exists(input_file):
        print(f"❌ Locations file not found: {input_file}")
        return
        
    with open(input_file, 'r', encoding='utf-8') as f:
        locations = json.load(f)
        if isinstance(locations, dict) and 'locations' in locations:
            locations = locations['locations']
            
    print(f"Loaded {len(locations)} locations.")
    
    # Re-create collection with 768 dimensions
    print("Re-creating 'campus_locations' collection...")
    await qdrant.create_collection(force_recreate=True)
    
    # Upload points (this now uses Gemini internally in our updated QdrantService)
    print("Generating Gemini embeddings and uploading...")
    await qdrant.upload_points(locations)
    print("✅ Locations migrated!")

async def revectorize_topics():
    print("\n--- Re-vectorizing Topics ---")
    qdrant = QdrantService(collection_name="topics")
    mongodb = MongoDBService()
    
    # Try loading from MongoDB first
    await mongodb.connect()
    topics = await mongodb.list_topics()
    await mongodb.disconnect()
    
    if not topics:
        print("MongoDB empty, trying JSON fallback...")
        input_file = 'data/topics/computer_engineering.json'
        if os.path.exists(input_file):
            with open(input_file, 'r', encoding='utf-8') as f:
                topics = json.load(f)
        else:
            print(f"❌ Topics file not found: {input_file}")
            return

    print(f"Loaded {len(topics)} topics.")
    
    # Re-create collection
    print("Re-creating 'topics' collection...")
    await qdrant.create_collection(force_recreate=True)
    
    # Prepare payload for Qdrant as expected by sync_topic_to_qdrant logic
    # though upload_points is more general.
    # The routes/topics.py uses a specific payload format. 
    # Let's use the format upload_points expects but enriched.
    
    upload_data = []
    for t in topics:
        t_id = str(t.get('topic_id') or t.get('_id'))
        # Generate the text to encode same as in sync_topic_to_qdrant
        tags = t.get('tags', t.get('keywords', []))
        text = f"{t.get('title', '')}. {t.get('description', '')}. Tags: {', '.join(tags)}"
        
        item = {
            "id": t_id, # QdrantService.upload_points uses item.get("id")
            "name": text, # qdrant_service use item['name']
            "topic_id": t_id,
            "title": t.get('title'),
            "description": t.get('description'),
            "department": t.get('department'),
            "option": t.get('option'),
            "year": t.get('year'),
            "status": t.get('status'),
            "tags": tags
        }
        upload_data.append(item)

    print("Generating Gemini embeddings and uploading...")
    await qdrant.upload_points(upload_data)
    print("✅ Topics migrated!")

async def main():
    print("🚀 Starting Global Re-vectorization (Switching to Gemini 768-dim)")
    try:
        await revectorize_locations()
        await revectorize_topics()
        print("\n🎉 ALL DATA RE-VECTORIZED SUCCESSFULLY!")
        print("You can now safely deploy to Render.")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
