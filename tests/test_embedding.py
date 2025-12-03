"""
Test script for embedding generation
Validates that embeddings were created correctly
Handles multiple data formats: OSM, custom, and campus locations
"""

import json
import pytest
import numpy as np # ✅ FIX: Added missing import for numpy
import sys
import os

# Add parent directory to path to import from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.generate_embeddings import EmbeddingService


def test_embeddings_exist():
    """Check if embeddings file exists and is valid"""
    filepath = 'data/locations/campus_locations_embeddings.json'
    assert os.path.exists(filepath), "Embeddings file not found!"
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    assert data, "Embeddings file is empty"
    assert isinstance(data, list), "Embeddings file is not a JSON list"


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
    """Validate that each location has proper embeddings."""
    filepath = 'data/locations/campus_locations_embeddings.json'
    
    with open(filepath, 'r') as f:
        locations = json.load(f)

    assert locations, "Embeddings file is empty"

    for location in locations:
        loc_id = get_location_id(location)
        
        assert 'embedding' in location, f"Location {loc_id} is missing 'embedding' field"
        
        embedding = location['embedding']
        assert isinstance(embedding, list), f"Location {loc_id} embedding is not a list"
        assert len(embedding) > 0, f"Location {loc_id} embedding is empty"
        assert all(isinstance(x, (int, float)) for x in embedding), f"Location {loc_id} embedding contains non-numeric values"


def test_embedding_similarity():
    """Test that embedding similarity scores are within a valid range."""
    filepath = 'data/locations/campus_locations_embeddings.json'
    
    with open(filepath, 'r') as f:
        locations = json.load(f)
    
    if len(locations) < 2:
        pytest.skip("Not enough locations to test similarity")
    
    # Calculate cosine similarity between first two locations
    emb1 = np.array(locations[0]['embedding'])
    emb2 = np.array(locations[1]['embedding'])
    
    # handle potential zero vectors
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0 or norm2 == 0:
        pytest.skip("Cannot calculate similarity with a zero vector")

    similarity = np.dot(emb1, emb2) / (norm1 * norm2)
    
    assert -1.0001 <= similarity <= 1.0001, f"Cosine similarity is outside the valid range of [-1, 1]: {similarity}"


def test_search_functionality():
    """Test basic semantic search capability."""
    service = EmbeddingService()
    
    filepath = 'data/locations/campus_locations_embeddings.json'
    with open(filepath, 'r') as f:
        locations = json.load(f)

    if len(locations) < 2:
        pytest.skip("Not enough locations to test search")

    test_query = "library"
    query_embedding = service.model.encode(test_query)

    similarities = []
    for location in locations:
        loc_embedding = np.array(location['embedding'])
        norm_query = np.linalg.norm(query_embedding)
        norm_loc = np.linalg.norm(loc_embedding)

        if norm_query == 0 or norm_loc == 0:
            continue

        similarity = np.dot(query_embedding, loc_embedding) / (norm_query * norm_loc)
        similarities.append(similarity)

    if len(similarities) < 2:
        pytest.skip("Not enough valid similarities to compare")

    similarities.sort(reverse=True)

    assert similarities[0] >= similarities[-1], "Top search result is not more similar than the last result"

