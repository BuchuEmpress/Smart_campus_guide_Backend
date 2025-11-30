"""
Test script for Qdrant integration
Tests connection, upload, search, and data retrieval
"""

import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.qdrant_service import QdrantService
from sentence_transformers import SentenceTransformer


def test_connection():
    """Test connection to Qdrant cloud"""
    print("--- Testing Qdrant Connection ---")
    
    try:
        service = QdrantService()
        print("✅ Connected to Qdrant successfully!")
        return True, service
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print("\nTroubleshooting:")
        print("- Check config/qdrant_config.json exists")
        print("- Verify URL and API key are correct")
        print("- Check internet connection")
        return False, None


def test_collection_exists(service):
    """Test if collection exists and has data"""
    print("\n--- Checking Collection ---")
    
    try:
        info = service.get_collection_info()
        if info and info.points_count > 0:
            print(f"✅ Collection has {info.points_count} points")
            print(f"✅ Vector size: {info.config.params.vectors.size}")
            print(f"✅ Distance metric: {info.config.params.vectors.distance}")
            return True
        elif info and info.points_count == 0:
            print("⚠️ Collection exists but is empty")
            print("Run: python services/qdrant_service.py upload")
            return False
        else:
            print("❌ Collection not found")
            print("Run: python services/qdrant_service.py upload")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Collection may not exist yet")
        print("Run: python services/qdrant_service.py upload")
        return False


def test_semantic_search(service):
    """Test semantic search functionality"""
    print("\n--- Testing Semantic Search ---")
    
    try:
        # Test queries relevant to campus
        test_queries = [
            "main gate",
            "university entrance",
            "landmark"
        ]
        
        all_passed = True
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Search
            results = service.search(query, limit=3)
            
            if not results:
                print("  ⚠️ No results found")
                all_passed = False
                continue
            
            print(f"  ✅ Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                score = result['score']
                location_type = result['type']
                description = result['description'][:60]
                
                print(f"  {i}. [{location_type}] Score: {score:.4f}")
                print(f"     {description}...")
            
        if all_passed:
            print("\n✅ Semantic search is working!")
        return all_passed
        
    except Exception as e:
        print(f"❌ Search test failed: {str(e)}")
        return False


def test_filtered_search(service):
    """Test search with type filtering"""
    print("\n--- Testing Filtered Search ---")
    
    try:
        # Get a sample query
        query = "location"
        
        print(f"Query: '{query}' with filter: type='landmark'")
        
        # Search only landmarks
        results = service.search(query, limit=3, filter_type="landmark")
        
        if results:
            print(f"✅ Found {len(results)} landmarks:")
            for result in results:
                print(f"  - Type: {result['type']}")
                print(f"    {result['description'][:60]}...")
            
            # Verify all results are landmarks
            all_landmarks = all(r['type'] == 'landmark' for r in results)
            if all_landmarks:
                print("✅ Filter working correctly - all results are landmarks")
                return True
            else:
                print("⚠️ Filter not working - mixed types in results")
                return False
        else:
            print("⚠️ No landmarks found (might not have any in data)")
            return True  # Not necessarily a failure
            
    except Exception as e:
        print(f"❌ Filtered search failed: {str(e)}")
        return False


def test_coordinates_retrieval(service):
    """Test that coordinates are properly stored and retrieved"""
    print("\n--- Testing Coordinate Retrieval ---")
    
    try:
        # Do a simple search
        results = service.search("location", limit=1)
        
        if results:
            result = results[0]
            lat = result.get('latitude')
            lon = result.get('longitude')
            
            if lat is not None and lon is not None:
                print(f"✅ Coordinates retrieved successfully")
                print(f"   Latitude: {lat}")
                print(f"   Longitude: {lon}")
                print(f"   Type: {result['type']}")
                return True
            else:
                print("❌ Coordinates missing from payload")
                return False
        else:
            print("❌ No results to test coordinates")
            return False
            
    except Exception as e:
        print(f"❌ Coordinate test failed: {str(e)}")
        return False


def test_payload_completeness(service):
    """Test that all metadata is stored correctly"""
    print("\n--- Testing Payload Completeness ---")
    
    try:
        results = service.search("location", limit=1)
        
        if not results:
            print("❌ No results to test payload")
            return False
        
        result = results[0]
        required_fields = ['id', 'type', 'latitude', 'longitude', 'description']
        
        missing = [field for field in required_fields if result.get(field) is None]
        
        if missing:
            print(f"❌ Missing fields in payload: {missing}")
            return False
        else:
            print("✅ All required fields present in payload:")
            for field in required_fields:
                value = result[field]
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"   {field}: {value}")
            return True
            
    except Exception as e:
        print(f"❌ Payload test failed: {str(e)}")
        return False


def test_score_threshold(service):
    """Test search with score threshold"""
    print("\n--- Testing Score Threshold ---")
    
    try:
        # Search with high threshold
        print("Searching with score threshold 0.7 (only high-quality matches)")
        results = service.search("main gate", limit=5, score_threshold=0.7)
        
        if results:
            print(f"✅ Found {len(results)} high-quality matches:")
            for result in results:
                print(f"   Score: {result['score']:.4f} - {result['description'][:50]}...")
            
            # Verify all scores are above threshold
            all_above = all(r['score'] >= 0.7 for r in results)
            if all_above:
                print("✅ Score threshold working correctly")
                return True
            else:
                print("⚠️ Some scores below threshold")
                return False
        else:
            print("⚠️ No results above threshold (query might not match well)")
            return True  # Not necessarily a failure
            
    except Exception as e:
        print(f"❌ Score threshold test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all validation tests"""
    print("=" * 80)
    print("QDRANT INTEGRATION TESTS")
    print("=" * 80)
    print()
    
    # Test connection first
    success, service = test_connection()
    if not success:
        print("\n❌ Cannot proceed without Qdrant connection")
        print("\nSetup checklist:")
        print("  1. Install: pip install qdrant-client")
        print("  2. Create: config/qdrant_config.json")
        print("  3. Upload: python services/qdrant_service.py upload")
        return
    
    # Run all tests
    tests = [
        ("Collection Exists", lambda: test_collection_exists(service)),
        ("Semantic Search", lambda: test_semantic_search(service)),
        ("Filtered Search", lambda: test_filtered_search(service)),
        ("Coordinate Retrieval", lambda: test_coordinates_retrieval(service)),
        ("Payload Completeness", lambda: test_payload_completeness(service)),
        ("Score Threshold", lambda: test_score_threshold(service))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Qdrant is working perfectly!")
        print("\n✅ Your vector database is ready for:")
        print("   - Semantic search queries")
        print("   - Location-based filtering")
        print("   - Integration with Google Maps API")
        print("   - Integration with Gemini API")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        print("\nNext steps:")
        print("  - Check failed tests above")
        print("  - Verify data was uploaded: python services/qdrant_service.py info")
        print("  - Re-upload if needed: python services/qdrant_service.py upload")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_all_tests()