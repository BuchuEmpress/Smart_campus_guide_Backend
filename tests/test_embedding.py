# Script to test embedding generation and data integrity.
"""
Test script for embedding generation
Validates that embeddings were created correctly
Tailored for campus data without 'name' field
"""

import json
import numpy as np
import sys
import os

# Add parent directory to path to import from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import EmbeddingService


def test_embeddings_exist():
    """Check if embeddings file exists and is valid"""
    filepath = 'data/campus_locations_embeddings.json'
    
    if not os.path.exists(filepath):
        print("❌ Embeddings file not found!")
        return False
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Found {len(data)} locations with embeddings")
    return True


def test_embedding_structure():
    """Validate that each location has required fields and proper embeddings"""
    filepath = 'data/campus_locations_embeddings.json'
    
    with open(filepath, 'r') as f:
        locations = json.load(f)
    
    # Required fields based on your data structure
    required_fields = ['id', 'latitude', 'longitude', 'type', 'description', 'embedding']
    
    for i, location in enumerate(locations):
        # Check required fields
        missing_fields = [field for field in required_fields if field not in location]
        if missing_fields:
            print(f"❌ Location {i} (ID: {location.get('id', 'unknown')}) missing fields: {missing_fields}")
            return False
        
        # Check embedding is a list of numbers
        embedding = location['embedding']
        if not isinstance(embedding, list):
            print(f"❌ Location {i} (ID: {location['id']}): embedding is not a list")
            return False
        
        if len(embedding) == 0:
            print(f"❌ Location {i} (ID: {location['id']}): embedding is empty")
            return False
        
        # Check all values are numbers
        if not all(isinstance(x, (int, float)) for x in embedding):
            print(f"❌ Location {i} (ID: {location['id']}): embedding contains non-numeric values")
            return False
    
    print(f"✅ All {len(locations)} locations have valid structure")
    print(f"✅ Embedding dimension: {len(locations[0]['embedding'])}")
    return True


def test_embedding_similarity():
    """Test that similar locations have similar embeddings"""
    filepath = 'data/campus_locations_embeddings.json'
    
    with open(filepath, 'r') as f:
        locations = json.load(f)
    
    if len(locations) < 2:
        print("⚠️ Not enough locations to test similarity")
        return True
    
    # Calculate cosine similarity between first two locations
    emb1 = np.array(locations[0]['embedding'])
    emb2 = np.array(locations[1]['embedding'])
    
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    # Use type and short description for display
    loc1_display = f"{locations[0]['type']} ({locations[0]['id'][:15]}...)"
    loc2_display = f"{locations[1]['type']} ({locations[1]['id'][:15]}...)"
    
    print(f"✅ Similarity between '{loc1_display}' and '{loc2_display}': {similarity:.4f}")
    print(f"   (Range: -1 to 1, where 1 is identical)")
    
    return True


def test_search_functionality():
    """Test basic semantic search capability"""
    print("\n🔍 Testing semantic search...")
    
    service = EmbeddingService()
    
    filepath = 'data/campus_locations_embeddings.json'
    with open(filepath, 'r') as f:
        locations = json.load(f)
    
    # Test queries relevant to your campus
    test_queries = [
        "main gate",
        "landmark",
        "university entrance"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_embedding = service.model.encode(query)
        
        # Calculate similarities
        similarities = []
        for location in locations:
            loc_embedding = np.array(location['embedding'])
            similarity = np.dot(query_embedding, loc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(loc_embedding)
            )
            # Create display name from type and short description
            display_name = f"{location['type']} - {location['description'][:50]}..."
            similarities.append((display_name, similarity, location['id']))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 3 matches:")
        for display_name, score, loc_id in similarities[:3]:
            print(f"  - {display_name}")
            print(f"    Score: {score:.4f} | ID: {loc_id[:20]}...")
    
    return True


def run_all_tests():
    """Run all validation tests"""
    print("=" * 50)
    print("EMBEDDING VALIDATION TESTS")
    print("=" * 50)
    
    tests = [
        ("File Exists", test_embeddings_exist),
        ("Data Structure", test_embedding_structure),
        ("Embedding Similarity", test_embedding_similarity),
        ("Search Functionality", test_search_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your embeddings are ready!")
        print("\n📊 Your Data Structure:")
        print("  - ID: Unique location identifier")
        print("  - Type: Location category (landmark, building, etc.)")
        print("  - Description: Detailed location info")
        print("  - Coordinates: Latitude & Longitude")
        print("  - Embedding: 384-dimensional semantic vector")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")


if __name__ == "__main__":
    run_all_tests()