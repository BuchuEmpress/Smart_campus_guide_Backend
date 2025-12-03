"""
Migrate Topics from JSON to MongoDB

Migrates topic data from JSON files to MongoDB database.

USAGE:
    python scripts/migrate_to_mongodb.py data/topics/computer_engineering.json
    
    # Or migrate all JSON files in directory
    python scripts/migrate_to_mongodb.py data/topics/
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.mongodb_service import MongoDBService
from services.topic_service import TopicService


def migrate_json_file(file_path: str, topic_service: TopicService):
    """
    Migrate a single JSON file to MongoDB.
    
    Args:
        file_path: Path to JSON file
        topic_service: TopicService instance
    
    Returns:
        Tuple of (success_count, error_count)
    """
    print(f"\n📄 Processing: {file_path}")
    print("-" * 80)
    
    try:
        # Load JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        topics = []
        if isinstance(data, list):
            topics = data
        elif isinstance(data, dict):
            if 'topics' in data:
                topics = data['topics']
            else:
                topics = [data]
        
        print(f"  Found {len(topics)} topics in file")
        
        success_count = 0
        error_count = 0
        
        for i, topic in enumerate(topics, 1):
            try:
                # Check for duplicates
                existing = topic_service.check_duplicate(
                    title=topic.get('title', ''),
                    category=topic.get('category', '')
                )
                
                if existing:
                    print(f"  ⚠️  [{i}] Skipped (duplicate): {topic.get('title', 'Unknown')}")
                    continue
                
                # Add timestamps if missing
                current_time_iso = datetime.utcnow().isoformat() # Get ISO string
                
                # ✅ FIX: Assign ISO-formatted string instead of datetime object
                if 'created_at' not in topic:
                    topic['created_at'] = current_time_iso
                if 'updated_at' not in topic:
                    topic['updated_at'] = current_time_iso
                
                # Add stats if missing
                if 'views' not in topic:
                    topic['views'] = 0
                if 'searches' not in topic:
                    topic['searches'] = 0
                
                # Add to MongoDB
                topic_id = topic_service.add_topic(topic)
                
                if topic_id:
                    success_count += 1
                    print(f"  ✅ [{i}] Migrated: {topic.get('title', 'Unknown')}")
                else:
                    error_count += 1
                    print(f"  ❌ [{i}] Failed: {topic.get('title', 'Unknown')}")
                
            except Exception as e:
                error_count += 1
                print(f"  ❌ [{i}] Error: {str(e)}")
        
        print(f"\n  Summary: {success_count} success, {error_count} errors")
        return (success_count, error_count)
        
    except Exception as e:
        print(f"  ❌ Failed to process file: {str(e)}")
        return (0, 1)


def migrate_directory(dir_path: str, topic_service: TopicService):
    """
    Migrate all JSON files in a directory.
    
    Args:
        dir_path: Path to directory
        topic_service: TopicService instance
    
    Returns:
        Tuple of (total_success, total_errors)
    """
    print(f"\n📁 Scanning directory: {dir_path}")
    print("=" * 80)
    
    json_files = list(Path(dir_path).glob('*.json'))
    
    if not json_files:
        print("  ⚠️  No JSON files found")
        return (0, 0)
    
    print(f"  Found {len(json_files)} JSON files")
    
    total_success = 0
    total_errors = 0
    
    for json_file in json_files:
        success, errors = migrate_json_file(str(json_file), topic_service)
        total_success += success
        total_errors += errors
    
    return (total_success, total_errors)


def main():
    """Main migration function."""
    print("\n" + "="*80)
    print("🚀 MONGODB MIGRATION TOOL")
    print("="*80)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\n❌ ERROR: No input specified")
        print("\nUSAGE:")
        print("  python scripts/migrate_to_mongodb.py <file.json>")
        print("  python scripts/migrate_to_mongodb.py <directory/>")
        print("\nEXAMPLES:")
        print("  python scripts/migrate_to_mongodb.py data/topics/computer_engineering.json")
        print("  python scripts/migrate_to_mongodb.py data/topics/")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Check if path exists
    if not os.path.exists(input_path):
        print(f"\n❌ ERROR: Path not found: {input_path}")
        sys.exit(1)
    
    try:
        # Initialize services
        print("\n🔌 Connecting to MongoDB...")
        mongodb = MongoDBService()
        
        print("✅ Connected to MongoDB!")
        
        print("\n🔧 Initializing Topic Service...")
        topic_service = TopicService(mongodb)
        
        print("✅ Topic Service ready!")
        
        # Migrate based on input type
        if os.path.isfile(input_path):
            # Single file
            success, errors = migrate_json_file(input_path, topic_service)
        else:
            # Directory
            success, errors = migrate_directory(input_path, topic_service)
        
        # Print final summary
        print("\n" + "="*80)
        print("📊 MIGRATION SUMMARY")
        print("="*80)
        print(f"  ✅ Successfully migrated: {success} topics")
        print(f"  ❌ Errors: {errors}")
        print(f"  📈 Success rate: {(success/(success+errors)*100) if (success+errors) > 0 else 0:.1f}%")
        print("="*80 + "\n")
        
        # Show collection stats
        print("📊 MongoDB Collection Stats:")
        print("-" * 80)
        # Using list_topics with no filters (limit=1000) for approximate total count
        total = len(topic_service.list_topics(limit=100000)) # Increased limit for more accurate count
        print(f"  Total topics in database: {total}")
        
        # Show by category
        categories = topic_service.get_unique_categories()
        print(f"  Categories: {len(categories)}")
        for cat in categories:
            count = len(topic_service.list_topics(filter_query={'category': cat}, limit=100000))
            print(f"    - {cat}: {count} topics")
        
        print("\n✅ Migration complete!")
        
        return 0 if errors == 0 else 1
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())