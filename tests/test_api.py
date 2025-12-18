import pytest
import json
from unittest.mock import MagicMock, AsyncMock
from api.main import app
from api.routes import navigation as from_routes

@pytest.fixture
def mock_qdrant():
    service = MagicMock()
    service.search = AsyncMock(return_value=[])
    service.search_by_text = AsyncMock(return_value=[])
    service.get_by_id = AsyncMock(return_value=None)
    app.dependency_overrides[from_routes.get_qdrant_service] = lambda: service
    yield service
    app.dependency_overrides.pop(from_routes.get_qdrant_service)

@pytest.fixture
def mock_gemini():
    service = MagicMock()
    service.extract_intent = AsyncMock(return_value={"action": "chat", "location_query": ""})
    service.generate_response = AsyncMock(return_value="I am a chatbot.")
    service.humanize_directions = AsyncMock(return_value="Walk straight.")
    service.enhance_description = AsyncMock(return_value="Enhanced.")
    app.dependency_overrides[from_routes.get_gemini_service] = lambda: service
    yield service
    app.dependency_overrides.pop(from_routes.get_gemini_service)

@pytest.fixture
def mock_osm():
    service = MagicMock()
    service.get_route = MagicMock(return_value={"steps": []})
    service.search_place = MagicMock(return_value=[])
    app.dependency_overrides[from_routes.get_osm_routing_service] = lambda: service
    yield service
    app.dependency_overrides.pop(from_routes.get_osm_routing_service)

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

@pytest.mark.asyncio
async def test_root_path_success(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "Smart Campus Guide API" in data["message"]

@pytest.mark.asyncio
async def test_search_endpoint_success(client, mock_qdrant, mock_gemini):
    mock_gemini.extract_intent.return_value = {"action": "search", "location_query": "lecture hall"}
    mock_qdrant.search_by_text.return_value = [{
        'id': 'loc_1',
        'name': 'Asanji Hall',
        'description': 'Main lecture hall',
        'latitude': 6.01,
        'longitude': 10.25,
        'type': 'class'
    }]
    
    response = client.post("/api/search", json={"query": "lecture hall"})
    assert response.status_code == 200
    data = response.json()
    assert len(data['locations']) == 1
    assert data['locations'][0]['name'] == 'Asanji Hall'

@pytest.mark.asyncio
async def test_navigation_route_success(client, mock_qdrant, mock_osm, mock_gemini):
    mock_gemini.extract_intent.return_value = {'action': 'navigate', 'location_query': 'library'}
    mock_qdrant.search_by_text.return_value = [{
        'id': 'lib_1',
        'name': 'Library',
        'latitude': 6.0,
        'longitude': 10.2,
        'type': 'building'
    }]
    
    response = client.post("/api/navigate", json={
        "query": "go to library", 
        "user_location": {"source": "gps", "lat": 6.01, "lon": 10.25}
    })
    
    assert response.status_code == 200
    assert response.json()['status'] == 'success'

@pytest.mark.asyncio
async def test_get_location_details_success(client, mock_qdrant, mock_gemini):
    mock_qdrant.get_by_id.return_value = {
        'id': 'lib_1',
        'name': 'Library',
        'latitude': 6.0,
        'longitude': 10.2,
        'type': 'building'
    }
    
    response = client.get("/api/location/lib_1")
    assert response.status_code == 200
    data = response.json()
    assert data['name'] == 'Library'
    assert data['enhanced_description'] == 'Enhanced.'
