"""
Upload Topics to MongoDB

Clean, modern script to upload topics to MongoDB with proper validation.
Handles duplicate checking and error recovery.

USAGE:
    python scripts/upload_topics_to_mongodb.py
    
OUTPUT:
    Uploads all topics to MongoDB 'topics' collection
"""

import json
import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv

load_dotenv()


class TopicMongoUploader:
    """Handles topic upload to MongoDB with validation."""
    
    def __init__(self):
        """Initialize MongoDB connection."""
        self.mongo_uri = os.getenv("MONGODB_URI")
        
        if not self.mongo_uri:
            raise ValueError("Missing MONGODB_URI in .env")
        
        self.client = AsyncIOMotorClient(self.mongo_uri)
        self.db = self.client['smart_campus_db']
        self.collection = self.db['topics']
        
        print("✅ MongoDB connection initialized\n")
    
    def load_topics(self, filepath: str = 'data/topics/computer_engineering.json') -> List[Dict]:
        """Load topics from JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Topic file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            topics = json.load(f)
        
        print(f"📂 Loaded {len(topics)} topics from {filepath}")
        return topics
    
    def normalize_topic(self, topic: Dict) -> Dict:
        """
        Normalize topic data to match expected schema.
        Handles field name variations and adds required fields.
        """
        # Create normalized topic
        normalized = {
            'topic_id': topic.get('id', ''),  # Keep 'id' as 'topic_id'
            'title': topic.get('title', ''),
            'description': topic.get('description', ''),
            'department': topic.get('department', ''),
            'option': topic.get('option', ''),  # Can be string or array
            'tags': topic.get('tags', []),
            'skills_required': topic.get('skills_required', []),
            'status': topic.get('status', 'Available'),
            'year': topic.get('year'),
            
            # Add timestamps
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            
            # Add stats
            'views': 0,
            'searches': 0
        }
        
        # Ensure arrays are arrays
        if not isinstance(normalized['tags'], list):
            normalized['tags'] = []
        
        if not isinstance(normalized['skills_required'], list):
            normalized['skills_required'] = []
        
        return normalized
    
    async def create_indexes(self):
        """Create database indexes for better query performance."""
        print("🔧 Creating database indexes...")
        
        try:
            # Create indexes
            await self.collection.create_index([("topic_id", 1)], unique=True)
            await self.collection.create_index([("department", 1)])
            await self.collection.create_index([("option", 1)])
            await self.collection.create_index([("year", -1)])
            await self.collection.create_index([("status", 1)])
            await self.collection.create_index([("title", "text")])
            
            print("✅ Indexes created!\n")
            
        except Exception as e:
            print(f"⚠️  Index creation warning: {str(e)}\n")
    
    async def check_duplicate(self, topic: Dict) -> Optional[Dict]:
        """Check if topic already exists (by topic_id or title)."""
        # Check by topic_id first
        existing = await self.collection.find_one({
            'topic_id': topic['topic_id']
        })
        
        if existing:
            return existing
        
        # Check by title (case-insensitive)
        existing = await self.collection.find_one({
            'title': {'$regex': f"^{topic['title']}$", '$options': 'i'}
        })
        
        return existing
    
    async def upload_topics(
        self, 
        topics: List[Dict], 
        skip_duplicates: bool = True,
        clear_existing: bool = False
    ):
        """
        Upload topics to MongoDB with duplicate handling.
        
        Args:
            topics: List of topic dictionaries
            skip_duplicates: If True, skip duplicates. If False, update them.
            clear_existing: If True, delete all existing topics first
        """
        print(f"🔄 Processing {len(topics)} topics...\n")
        
        # Clear existing if requested
        if clear_existing:
            print("⚠️  Clearing existing topics...")
            result = await self.collection.delete_many({})
            print(f"   Deleted {result.deleted_count} existing topics\n")
        
        # Counters
        inserted = 0
        updated = 0
        skipped = 0
        errors = 0
        
        # Process each topic
        for i, topic in enumerate(topics, 1):
            try:
                # Normalize topic data
                normalized = self.normalize_topic(topic)
                
                # Check for duplicate
                existing = await self.check_duplicate(normalized)
                
                if existing:
                    if skip_duplicates:
                        skipped += 1
                        print(f"   ⏭️  [{i}/{len(topics)}] Skipped (duplicate): {normalized['title']}")
                    else:
                        # Update existing
                        normalized['updated_at'] = datetime.utcnow()
                        await self.collection.update_one(
                            {'_id': existing['_id']},
                            {'$set': normalized}
                        )
                        updated += 1
                        print(f"   🔄 [{i}/{len(topics)}] Updated: {normalized['title']}")
                else:
                    # Insert new
                    await self.collection.insert_one(normalized)
                    inserted += 1
                    print(f"   ✅ [{i}/{len(topics)}] Inserted: {normalized['title']}")
            
            except DuplicateKeyError:
                skipped += 1
                print(f"   ⏭️  [{i}/{len(topics)}] Skipped (duplicate key): {topic.get('title', 'Unknown')}")
            
            except Exception as e:
                errors += 1
                print(f"   ❌ [{i}/{len(topics)}] Error: {topic.get('title', 'Unknown')} - {str(e)}")
        
        # Print summary
        print(f"\n" + "="*80)
        print("📊 UPLOAD SUMMARY")
        print("="*80)
        print(f"   ✅ Inserted: {inserted}")
        print(f"   🔄 Updated: {updated}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   ❌ Errors: {errors}")
        print(f"   📈 Success rate: {((inserted + updated) / len(topics) * 100):.1f}%")
        print("="*80 + "\n")
    
    async def verify_upload(self):
        """Verify topics were uploaded correctly."""
        print("🔍 Verifying upload...\n")
        
        # Count total
        total = await self.collection.count_documents({})
        print(f"   Total topics: {total}")
        
        # Count by department
        pipeline = [
            {'$group': {'_id': '$department', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]
        
        cursor = self.collection.aggregate(pipeline)
        by_department = await cursor.to_list(length=None)
        
        print(f"\n   Topics by department:")
        for dept in by_department:
            print(f"      - {dept['_id']}: {dept['count']}")
        
        # Count by status
        pipeline = [
            {'$group': {'_id': '$status', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]
        
        cursor = self.collection.aggregate(pipeline)
        by_status = await cursor.to_list(length=None)
        
        print(f"\n   Topics by status:")
        for status in by_status:
            print(f"      - {status['_id']}: {status['count']}")
        
        # Test search
        print(f"\n   Testing text search...")
        cursor = self.collection.find({'$text': {'$search': 'AI machine learning'}}).limit(3)
        results = await cursor.to_list(length=3)
        
        print(f"   Found {len(results)} results for 'AI machine learning':")
        for result in results:
            print(f"      - {result['title']}")
    
    async def run(
        self, 
        filepath: str = 'data/topics/computer_engineering.json',
        clear_existing: bool = False,
        skip_duplicates: bool = True
    ):
        """Main execution pipeline."""
        try:
            print("="*80)
            print("📚 TOPIC UPLOAD TO MONGODB")
            print("="*80 + "\n")
            
            # Load topics
            topics = self.load_topics(filepath)
            
            # Create indexes
            await self.create_indexes()
            
            # Upload topics
            await self.upload_topics(
                topics, 
                skip_duplicates=skip_duplicates,
                clear_existing=clear_existing
            )
            
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
        
        finally:
            # Close connection
            self.client.close()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload topics to MongoDB')
    parser.add_argument('--file', type=str, 
                       default='data/topics/computer_engineering.json',
                       help='Path to topics JSON file')
    parser.add_argument('--clear', action='store_true',
                       help='Clear existing topics before upload')
    parser.add_argument('--update-duplicates', action='store_true',
                       help='Update duplicates instead of skipping')
    
    args = parser.parse_args()
    
    uploader = TopicMongoUploader()
    success = await uploader.run(
        filepath=args.file,
        clear_existing=args.clear,
        skip_duplicates=not args.update_duplicates
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())