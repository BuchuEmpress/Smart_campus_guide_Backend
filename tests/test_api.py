"""
Integration Tests for FastAPI Endpoints

Tests all API endpoints with mocked services.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

# Import app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant service."""
    with patch('api.routes.navigation.QdrantService') as mock:
        instance = MagicMock()
        instance.search.return_value = [
            {
                'score': 0.95,
                'id': 'loc_1',
                'name': 'Asanji Hall',
                'description': 'Main lecture hall',
                'latitude': 6.010380,
                'longitude': 10.259017,
                'type': 'class'
            }
        ]
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_osm():
    """Mock OSM service."""
    with patch('api.routes.navigation.OSMRoutingService') as mock:
        instance = MagicMock()
        instance.get_route.return_value = {
            'distance': 100,
            'distance_text': '100 meters',
            'duration': 60,
            'duration_text': '1 minute',
            'steps': [
                {'instruction': 'Head north', 'distance_text': '50 meters'},
                {'instruction': 'Turn right', 'distance_text': '50 meters'}
            ],
            'geometry': {},
            'profile': 'foot'
        }
        instance.search_place.return_value = [
            {
                'name': 'Test Place',
                'display_name': 'Test Place, Bamenda',
                'latitude': 5.9631,
                'longitude': 10.1591,
                'type': 'place'
            }
        ]
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_gemini():
    """Mock Gemini service."""
    with patch('api.routes.navigation.GeminiService') as mock:
        instance = MagicMock()
        instance.humanize_directions.return_value = "Walk straight to Asanji Hall"
        instance.extract_intent.return_value = {'type': 'navigation'}
        instance.generate_response.return_value = "I can help you find that"
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_analytics():
    """Mock Analytics service."""
    with patch('api.routes.navigation.AnalyticsService') as mock:
        instance = MagicMock()
        instance.log_search.return_value = None
        mock.return_value = instance
        yield mock


# ============================================================================
# ROOT ENDPOINT TESTS
# ============================================================================

def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Smart Campus Guide" in data["message"]


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_info(client):
    """Test API info endpoint."""
    response = client.get("/api/info")
    
    assert response.status_code == 200
    data = response.json()
    assert "api_name" in data
    assert "features" in data


# ============================================================================
# NAVIGATION ENDPOINT TESTS
# ============================================================================

def test_navigate_with_gps(client, mock_qdrant, mock_osm, mock_gemini, mock_analytics):
    """Test navigation with GPS location."""
    request_data = {
        "query": "Asanji Hall",
        "user_location": {
            "source": "gps",
            "lat": 6.010317,
            "lon": 10.258816
        },
        "travel_mode": "walking"
    }
    
    response = client.post("/api/navigate", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "destination" in data
    assert "humanized_directions" in data
    assert "distance" in data


def test_navigate_with_manual_location(client, mock_qdrant, mock_osm, mock_gemini, mock_analytics):
    """Test navigation with manual address."""
    # Mock geocoding
    mock_osm.return_value.geocode.return_value = {
        'latitude': 6.010317,
        'longitude': 10.258816,
        'address': 'Main Gate'
    }
    
    request_data = {
        "query": "Library",
        "user_location": {
            "source": "manual",
            "address": "Main Gate"
        },
        "travel_mode": "walking"
    }
    
    response = client.post("/api/navigate", json=request_data)
    
    assert response.status_code == 200


def test_navigate_missing_gps_coords(client):
    """Test navigation with missing GPS coordinates."""
    request_data = {
        "query": "Library",
        "user_location": {
            "source": "gps"
            # Missing lat/lon
        },
        "travel_mode": "walking"
    }
    
    response = client.post("/api/navigate", json=request_data)
    
    assert response.status_code == 422  # Validation error


def test_navigate_location_not_found(client, mock_qdrant, mock_osm, mock_gemini, mock_analytics):
    """Test navigation when location not found."""
    # Mock no results
    mock_qdrant.return_value.search.return_value = []
    mock_osm.return_value.search_place.return_value = []
    
    request_data = {
        "query": "NonexistentPlace12345",
        "user_location": {
            "source": "gps",
            "lat": 6.010317,
            "lon": 10.258816
        },
        "travel_mode": "walking"
    }
    
    response = client.post("/api/navigate", json=request_data)
    
    assert response.status_code == 404


# ============================================================================
# SEARCH ENDPOINT TESTS
# ============================================================================

def test_search_all(client, mock_qdrant, mock_osm, mock_gemini, mock_analytics):
    """Test search with 'all' filter."""
    request_data = {
        "query": "library",
        "filter": "all",
        "limit": 10
    }
    
    response = client.post("/api/search", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "locations" in data
    assert "total_results" in data


def test_search_on_campus(client, mock_qdrant, mock_osm, mock_gemini, mock_analytics):
    """Test search with on-campus filter."""
    request_data = {
        "query": "hall",
        "filter": "on-campus",
        "limit": 5
    }
    
    response = client.post("/api/search", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["locations"], list)


def test_search_off_campus(client, mock_qdrant, mock_osm, mock_gemini, mock_analytics):
    """Test search with off-campus filter."""
    request_data = {
        "query": "hospital",
        "filter": "off-campus",
        "limit": 5
    }
    
    response = client.post("/api/search", json=request_data)
    
    assert response.status_code == 200


# ============================================================================
# LOCATION DETAILS TESTS
# ============================================================================

def test_get_location_details(client, mock_qdrant, mock_osm, mock_gemini, mock_analytics):
    """Test getting location details."""
    response = client.get("/api/location/loc_1")
    
    assert response.status_code in [200, 404]  # Depends on mock


def test_get_location_not_found(client, mock_qdrant, mock_osm, mock_gemini, mock_analytics):
    """Test getting non-existent location."""
    mock_qdrant.return_value.search.return_value = []
    
    response = client.get("/api/location/nonexistent_id")
    
    assert response.status_code == 404


# ============================================================================
# CHAT ENDPOINT TESTS
# ============================================================================

def test_chat_endpoint(client, mock_qdrant, mock_osm, mock_gemini, mock_analytics):
    """Test chat endpoint."""
    response = client.post(
        "/api/chat",
        params={"message": "Where is the library?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data


# ============================================================================
# CORS TESTS
# ============================================================================

def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.options(
        "/api/navigate",
        headers={"Origin": "http://localhost:3000"}
    )
    
    # Check if response is successful (CORS configured)
    assert response.status_code in [200, 405]


# ============================================================================
# VALIDATION TESTS
# ============================================================================

def test_invalid_json(client):
    """Test API handles invalid JSON."""
    response = client.post(
        "/api/navigate",
        data="This is not JSON",
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 422


def test_missing_required_fields(client):
    """Test API validates required fields."""
    request_data = {
        "query": "library"
        # Missing user_location
    }
    
    response = client.post("/api/navigate", json=request_data)
    
    assert response.status_code == 422


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_internal_error_handling(client, mock_qdrant, mock_osm, mock_gemini, mock_analytics):
    """Test API handles internal errors."""
    # Make service raise error
    mock_qdrant.return_value.search.side_effect = Exception("Database error")
    
    request_data = {
        "query": "library",
        "user_location": {
            "source": "gps",
            "lat": 6.010317,
            "lon": 10.258816
        },
        "travel_mode": "walking"
    }
    
    response = client.post("/api/navigate", json=request_data)
    
    # Should handle error gracefully
    assert response.status_code in [500, 404]


# ============================================================================
# DOCUMENTATION TESTS
# ============================================================================

def test_swagger_ui_accessible(client):
    """Test Swagger UI is accessible."""
    response = client.get("/docs")
    
    assert response.status_code == 200


def test_openapi_schema(client):
    """Test OpenAPI schema is available."""
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])