import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from api.main import app

client = TestClient(app)

@pytest.fixture
def mock_mongodb():
    service = MagicMock()
    service.connect = AsyncMock(return_value=True)
    service.get_chat_history = AsyncMock(return_value=[])
    service.save_chat_message = AsyncMock(return_value=True)
    service.disconnect = AsyncMock()
    
    app.dependency_overrides[from_routes.get_mongodb_service] = lambda: service
    yield service
    app.dependency_overrides.pop(from_routes.get_mongodb_service)

@pytest.fixture
def mock_gemini():
    service = MagicMock()
    service.extract_intent = AsyncMock(return_value={"action": "chat", "location_query": None})
    service.generate_response = AsyncMock(return_value="Hello! I'm your guide.")
    
    app.dependency_overrides[from_routes.get_gemini_service] = lambda: service
    yield service
    app.dependency_overrides.pop(from_routes.get_gemini_service)

@pytest.fixture
def mock_qdrant():
    service = MagicMock()
    app.dependency_overrides[from_routes.get_qdrant_service] = lambda: service
    yield service
    app.dependency_overrides.pop(from_routes.get_qdrant_service)

from api.routes import navigation as from_routes

def test_navigation_chat_success(mock_mongodb, mock_gemini, mock_qdrant):
    """Test standard chat flow."""
    payload = {
        "message": "hey",
        "session_id": "test_session",
        "user_location": {
            "lat": 6.01,
            "lon": 10.25,
            "source": "gps"
        }
    }
    
    response = client.post("/api/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Hello! I'm your guide."
    
    mock_mongodb.connect.assert_called_once()
    mock_gemini.extract_intent.assert_called_once()
    mock_gemini.generate_response.assert_called_once()

def test_navigation_chat_navigate_action(mock_mongodb, mock_gemini, mock_qdrant):
    """Test chat flow when intent is navigation."""
    mock_gemini.extract_intent.return_value = {
        "action": "navigate",
        "location_query": "library"
    }
    
    # Mock Qdrant search
    mock_qdrant.search_by_text = AsyncMock(return_value=[{
        "id": "lib_1",
        "name": "Library",
        "latitude": 6.0,
        "longitude": 10.2
    }])
    
    # Mock OSM routing
    with patch('api.routes.navigation.OSMRoutingService') as mock_osm_class:
        mock_osm = MagicMock()
        mock_osm.get_route.return_value = {"steps": []}
        mock_osm_class.return_value = mock_osm
        
        mock_gemini.humanize_directions = AsyncMock(return_value="Go straight.")
        
        payload = {
            "message": "go to library",
            "session_id": "test_session",
            "user_location": {
                "lat": 6.01,
                "lon": 10.25,
                "source": "gps"
            }
        }
        
        response = client.post("/api/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        # Since it's Agentic chat, the final message comes from generate_response
        # but the logic should have successfully fetched the route data.
        mock_gemini.generate_response.assert_called_once()
        # Verify it used the correct lat/lon from user_location
        # start_coords = (request.user_location.lat, request.user_location.lon)
        # So it should NOT have crashed.
