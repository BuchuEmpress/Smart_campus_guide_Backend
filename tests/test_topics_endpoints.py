
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from api.routes.topics import router
from services.topic_service import TopicService
from services.topic_intelligence_service import TopicIntelligenceService

# Create a TestClient with the router's main application path
from api.main import app
client = TestClient(app)

@pytest.fixture
def mock_topic_service():
    # Patch where it's used in the routes
    with patch('api.routes.topics.topic_service') as mock:
        yield mock

@pytest.fixture
def mock_topic_ai():
    # Patch where it's used in the routes
    with patch('api.routes.topics.topic_ai') as mock:
        yield mock

def test_create_topic_success(mock_topic_service):
    """Test successful topic creation."""
    # Setup mock
    mock_topic_service.find_duplicate.return_value = []
    mock_topic_service.add_topic.return_value = "new_id"
    mock_topic_service.get_topic_by_id.return_value = {
        "topic_id": "new_id",
        "title": "New Topic",
        "department": "CS",
        "option": "AI",
        "year": 2024,
        "status": "reserved"
    }
    
    payload = {
        "title": "New Topic",
        "department": "CS",
        "option": "AI",
        "year": 2024
    }
    
    response = client.post("/api/topics/", json=payload)
    
    assert response.status_code == 200
    mock_topic_service.find_duplicate.assert_called_once()
    mock_topic_service.add_topic.assert_called_once()
    
def test_list_topics(mock_topic_service):
    mock_topic_service.list_topics.return_value = []
    response = client.get("/api/topics/")
    assert response.status_code == 200
    mock_topic_service.list_topics.assert_called_once()

def test_get_statistics(mock_topic_service):
    mock_topic_service.get_statistics.return_value = {
        "total_topics": 10,
        "total_views": 100,
        "total_searches": 50,
        "by_department": {"CS": 10},
        "by_option": {"AI": 10},
        "by_year": {"2024": 10},
        "by_status": {"reserved": 10}
    }
    response = client.get("/api/topics/stats/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["total_topics"] == 10
    assert data["total_views"] == 100
    assert data["total_searches"] == 50
    assert data["by_category"] == {"AI": 10}  # Check legacy alias
    assert data["by_difficulty"] == {"reserved": 10}  # Check legacy alias
    mock_topic_service.get_statistics.assert_called_once()
