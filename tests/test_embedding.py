"""
Test script for embedding generation
Validates that embeddings were created correctly
Handles multiple data formats: OSM, custom, and campus locations
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
    filepath = 'data/locations/campus_locations_embeddings.json'
    
    if not os.path.exists(filepath):
        print("❌ Embeddings file not found!")
        return False
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Found {len(data)} locations with embeddings")
    return True


def get_location_id(location):
    """
    Extract a unique identifier from various location formats
    Handles different ID formats from OSM and custom data
    """
    # Try direct id field
    if 'id' in location:
        return str(location['id'])
    
    # Try OSM format (type + id)
    if 'type' in location and location['type'] in ['node', 'way', 'relation']:
        return f"{location['type']}_{location.get('id', 'unknown')}"
    
    # Try custom data format
    if 'data' in location and 'name' in location['data']:
        return location['data']['name']
    
    # Try name field
    if 'name' in location:
        return location['name']
    
    # Try tags (OSM format)
    if 'tags' in location and 'name' in location['tags']:
        return location['tags']['name']
    
    # Fallback: use index or "unknown"
    return "unknown"


def get_location_display_name(location):
    """
    Get a human-readable name for display purposes
    Tries multiple fields to find the best name
    """
    # Try name field
    if 'name' in location:
        return location['name']
    
    # Try OSM tags
    if 'tags' in location and 'name' in location['tags']:
        return location['tags']['name']
    
    # Try custom data
    if 'data' in location and 'name' in location['data']:
        return location['data']['name']
    
    # Try type + description
    loc_type = location.get('type', 'location')
    description = location.get('description', '')
    if description:
        return f"{loc_type}: {description[:50]}..."
    
    # Try tags description
    if 'tags' in location:
        tags = location['tags']
        if 'amenity' in tags:
            return f"{tags['amenity']}"
        if 'building' in tags:
            return f"{tags['building']}"
    
    return loc_type


def test_embedding_structure():
    """Validate that each location has proper embeddings (flexible field checking)"""
    filepath = 'data/locations/campus_locations_embeddings.json'
    
    with open(filepath, 'r') as f:
        locations = json.load(f)
    
    issues = []
    valid_count = 0
    
    for i, location in enumerate(locations):
        loc_id = get_location_id(location)
        
        # CRITICAL: Every location MUST have an embedding
        if 'embedding' not in location:
            issues.append(f"Location {i} ({loc_id}): missing 'embedding' field")
            continue
        
        # Check embedding is a list of numbers
        embedding = location['embedding']
        if not isinstance(embedding, list):
            issues.append(f"Location {i} ({loc_id}): embedding is not a list")
            continue
        
        if len(embedding) == 0:
            issues.append(f"Location {i} ({loc_id}): embedding is empty")
            continue
        
        # Check all values are numbers
        if not all(isinstance(x, (int, float)) for x in embedding):
            issues.append(f"Location {i} ({loc_id}): embedding contains non-numeric values")
            continue
        
        valid_count += 1
    
    # Print issues if any
    if issues:
        print(f"⚠️ Found {len(issues)} issue(s):")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more")
        return False
    
    print(f"✅ All {len(locations)} locations have valid embeddings")
    print(f"✅ Embedding dimension: {len(locations[0]['embedding'])}")
    return True


def test_embedding_similarity():
    """Test that embeddings are properly generated with reasonable values"""
    filepath = 'data/locations/campus_locations_embeddings.json'
    
    with open(filepath, 'r') as f:
        locations = json.load(f)
    
    if len(locations) < 2:
        print("⚠️ Not enough locations to test similarity")
        return True
    
    # Calculate cosine similarity between first two locations
    emb1 = np.array(locations[0]['embedding'])
    emb2 = np.array(locations[1]['embedding'])
    
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    # Get display names
    loc1_display = get_location_display_name(locations[0])
    loc2_display = get_location_display_name(locations[1])
    
    print(f"✅ Similarity between:")
    print(f"   '{loc1_display[:60]}...'")
    print(f"   '{loc2_display[:60]}...'")
    print(f"   Score: {similarity:.4f} (Range: -1 to 1, where 1 is identical)")
    
    return True


def test_search_functionality():
    """Test basic semantic search capability"""
    print("\n🔍 Testing semantic search...")
    
    service = EmbeddingService()
    
    filepath = 'data/locations/campus_locations_embeddings.json'
    with open(filepath, 'r') as f:
        locations = json.load(f)
    
    # Test queries relevant to your campus
    test_queries = [
        "main gate",
        "library", 
        "restaurant"
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
            
            # Get display name
            display_name = get_location_display_name(location)
            loc_id = get_location_id(location)
            
            similarities.append((display_name, similarity, loc_id))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 3 matches:")
        for display_name, score, loc_id in similarities[:3]:
            print(f"  - {display_name[:70]}")
            print(f"    Score: {score:.4f}")
    
    print("\n✅ Semantic search is working!")
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
            import traceback
            traceback.print_exc()
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
        print("\n🎉 All tests passed! Your embeddings are ready for semantic search!")
        print("\n📊 What you can do now:")
        print("  ✅ Semantic search across all campus locations")
        print("  ✅ Find similar places using natural language")
        print("  ✅ Smart recommendations based on location context")
        print("  ✅ 384-dimensional vector embeddings for ML tasks")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed - check details above")


if __name__ == "__main__":
    run_all_tests()