
import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.topic_service import TopicService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_case_sensitivity():
    print("="*50)
    print("TESTING CASE SENSITIVITY")
    print("="*50)
    
    service = TopicService()
    
    # 1. Setup: Create a test topic with mixed case
    topic_data = {
        'title': 'TEST TOPIC Mixed Case',
        'department': 'Computer Engineering',
        'option': 'Software Engineering',
        'status': 'In_Progress'
    }
    
    # Clean up previous test
    print("Cleaning up previous test topics...")
    existing = service.search_topics(query='TEST TOPIC', limit=100)
    for t in existing:
        service.delete_topic(t['topic_id'])
        
    print(f"Adding topic: {topic_data['title']}")
    topic_id = service.add_topic(topic_data)
    print(f"Topic ID: {topic_id}")
    
    if not topic_id:
        print("❌ Failed to add topic. Aborting.")
        return

    # 2. Test Search (Title)
    print("\n--- Testing Title Search ---")
    
    # Lowercase search
    res_lower = service.search_topics(query="test topic")
    print(f"Search 'test topic': {len(res_lower)} found")
    
    # Uppercase search
    res_upper = service.search_topics(query="TEST TOPIC")
    print(f"Search 'TEST TOPIC': {len(res_upper)} found")
    
    # Mixed search
    res_mixed = service.search_topics(query="TeSt ToPiC")
    print(f"Search 'TeSt ToPiC': {len(res_mixed)} found")
    
    if len(res_lower) > 0 and len(res_upper) > 0:
        print("✅ Title search is case-insensitive")
    else:
        print("❌ Title search FAILED case-sensitivity check")

    # 3. Test Filters (Department)
    print("\n--- Testing Filters ---")
    
    # Exact match different case
    res_dept = service.list_topics(filter_query={'department': 'computer engineering'})
    print(f"Filter dept 'computer engineering': {len(res_dept)} found")
    
    if len(res_dept) > 0:
        print("✅ Department filter is case-insensitive")
    else:
        print("❌ Department filter FAILED case-sensitivity check")

    # 4. Test ID Lookup
    print("\n--- Testing ID Lookup ---")
    
    # Exact ID
    t1 = service.get_topic_by_id(topic_id)
    print(f"Get ID '{topic_id}': {'Found' if t1 else 'Not Found'}")
    
    # Modified Case ID (if topic_id has letters)
    # topic_id usually starts with 'topic_'. Let's try 'TOPIC_...'
    topic_id_upper = topic_id.replace('topic_', 'TOPIC_')
    t2 = service.get_topic_by_id(topic_id_upper)
    print(f"Get ID '{topic_id_upper}': {'Found' if t2 else 'Not Found'}")
    
    if t2:
         print("✅ ID lookup is case-insensitive")
    else:
         print("⚠️ ID lookup is case-SENSITIVE (This might be the issue)")

    # 5. Test "topic" vs "TOPIC" derived from user complaint
    # User said: "when i type 'topic' vs 'TOPIC'"
    print("\n--- Specific User Scenario ---")
    res_u1 = service.search_topics(query="topic")
    print(f"Search 'topic': {len(res_u1)} found")
    
    res_u2 = service.search_topics(query="TOPIC")
    print(f"Search 'TOPIC': {len(res_u2)} found")

    # Clean up
    service.delete_topic(topic_id)
    print("\nTest Complete")

if __name__ == "__main__":
    test_case_sensitivity()
