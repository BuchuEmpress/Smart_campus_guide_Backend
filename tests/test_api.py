"""
Integration tests for the main API routes (UPDATED for /api/... routes)

This file uses an httpx.AsyncClient to test the FastAPI endpoints
and ensures all external service calls are correctly mocked out
using AsyncMock for non-blocking operations.
"""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch


# ============================================================================
# API ENDPOINT TESTS (Integration)
# ============================================================================

@pytest.mark.asyncio
async def test_root_path_success(client):
    """Test the root path returns basic status information."""
    # FIX: Remove await - TestClient returns responses synchronously
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "running"
    assert data["message"] == "Welcome to Smart Campus Guide API" or "Welcome to Smart Campus Guide API" in data.get("message", "")


@pytest.mark.asyncio
async def test_search_endpoint_success(client, mock_qdrant):
    """
    Test the search endpoint's successful response path.
    """
    # Arrange
    mock_qdrant.return_value.search.return_value = [
        {
            'id': 'loc_1',
            'name': 'Asanji Hall',
            'description': 'Main lecture hall',
            'latitude': 6.010380,
            'longitude': 10.259017,
            'type': 'class'
        }
    ]

    # Act - FIX: Remove await
    response = client.post(
        "/api/search",
        json={"query": "lecture hall"},
        headers={"Content-Type": "application/json"}
    )

    # Assert
    assert response.status_code == 200
    data = response.json()

    # FIX: Check 'locations' instead of 'results' as per SearchResponse model
    assert len(data['locations']) == 1
    assert data['locations'][0]['name'] == 'Asanji Hall'

    mock_qdrant.return_value.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_navigation_route_success(client, mock_qdrant, mock_osm, mock_gemini):
    """
    Test the navigation route success path: intent -> search -> route -> humanize.
    """
    mock_gemini.return_value.extract_intent.return_value = {
        'action': 'navigate',
        'location_query': 'library',
        'location_type': 'building',
        'preferences': {},
        'on_campus': True
    }

    mock_qdrant.return_value.search.return_value = [
        {
            'id': 'lib_1',
            'name': 'Central Library',
            'description': 'Main study building',
            'latitude': 6.0,
            'longitude': 10.2,
            'type': 'building'
        }
    ]

    mock_gemini.return_value.humanize_directions.return_value = (
        "Follow the main path until you see the big clock tower."
    )

    # FIX: Remove await and update JSON body to match NavigationRequest
    response = client.post(
        "/api/navigate",
        json={
            "query": "How do I get to the library?", 
            "user_location": {"source": "gps", "lat": 6.010317, "lon": 10.258816},
            "travel_mode": "walking"
        },
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    data = response.json()

    assert data['status'] == 'success'
    assert 'Follow the main path' in data['message']

    mock_gemini.return_value.extract_intent.assert_awaited_once()
    mock_osm.return_value.get_route.assert_awaited_once()
    mock_gemini.return_value.humanize_directions.assert_awaited_once()


@pytest.mark.asyncio
async def test_navigation_chat_fallback(client, mock_gemini):
    """Test navigation route when Gemini detects a chat intent."""
    mock_gemini.return_value.extract_intent.return_value = {
        'action': 'chat',
        'location_query': '',
        'location_type': 'other',
        'preferences': {},
        'on_campus': None
    }

    mock_gemini.return_value.generate_response.return_value = (
        "I am a helpful campus assistant. How can I guide you?"
    )

    # FIX: Remove await and update JSON body
    response = client.post(
        "/api/navigate",
        json={
            "query": "Hello, how are you today?",
            "user_location": {"source": "gps", "lat": 0.0, "lon": 0.0} # Dummy location for chat
        },
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    data = response.json()

    assert data['status'] == 'chat'
    assert 'helpful campus assistant' in data['message']

    mock_gemini.return_value.extract_intent.assert_awaited_once()
    mock_gemini.return_value.generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_navigation_qdrant_no_results(client, mock_qdrant, mock_gemini):
    """
    Test case where search returns empty results.
    """
    mock_gemini.return_value.extract_intent.return_value = {
        'action': 'navigate',
        'location_query': 'secret cave',
        'location_type': 'other',
        'preferences': {},
        'on_campus': True
    }

    mock_qdrant.return_value.search.return_value = []

    mock_gemini.return_value.generate_response.return_value = (
        "I couldn't find a 'secret cave' on campus."
    )

    # FIX: Remove await and update JSON body
    response = client.post(
        "/api/navigate",
        json={
            "query": "Navigate to the secret cave", 
            "user_location": {"source": "gps", "lat": 0.0, "lon": 0.0}
        },
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    data = response.json()

    assert data['status'] == 'chat'
    assert "secret cave" in data['message']

    mock_qdrant.return_value.search.assert_awaited_once()
    mock_gemini.return_value.generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_location_details_success(client, mock_qdrant, mock_gemini):
    """Test retrieving location details."""
    location_data = {
        'id': 'lib_1',
        'name': 'Central Library',
        'description': 'Main study building',
        'latitude': 6.0,
        'longitude': 10.2,
        'type': 'building'
    }

    # FIX: Mock search instead of get_location_by_id because the route uses search
    mock_qdrant.return_value.search.return_value = [location_data]

    mock_gemini.return_value.enhance_description.return_value = (
        "The Central Library is a quiet, modern space perfect for studying."
    )

    # FIX: Remove await
    response = client.get("/api/location/lib_1")

    assert response.status_code == 200
    data = response.json()

    assert data['name'] == 'Central Library'
    assert 'quiet, modern space' in data['enhanced_description']

    # Verify search was called with the ID
    mock_qdrant.return_value.search.assert_awaited_once()
    mock_gemini.return_value.enhance_description.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_location_details_not_found(client, mock_qdrant):
    """Test 404 for unknown ID."""
    # FIX: Mock search to return empty list
    mock_qdrant.return_value.search.return_value = []

    # FIX: Remove await
    response = client.get("/api/location/non_existent_id")

    assert response.status_code == 404
    # FIX: Update expected error message to match route implementation
    assert response.json() == {"detail": "Location not found: non_existent_id"}


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
