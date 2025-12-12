
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from api.routes.topics import router
from services.topic_service import TopicService
from services.topic_intelligence_service import TopicIntelligenceService

# Create a TestClient
client = TestClient(router)

@pytest.fixture
def mock_topic_service():
    with patch('api.routes.topics.topic_service') as mock:
        yield mock

@pytest.fixture
def mock_topic_ai():
    with patch('api.routes.topics.topic_ai') as mock:
        yield mock

def test_create_topic_find_duplicate_error(mock_topic_service):
    """
    Test that create_topic fails because find_duplicate does not exist on TopicService.
    This test expects the current implementation to fail or if we mock it, 
    we verify what it tries to call.
    """
    # We want to see what happens when we call the endpoint.
    # Since we are mocking the service instance in the router module,
    # we can check if it calls find_duplicate.
    
    # Setup the mock to raise AttributeError if find_duplicate is accessed
    # strictly speaking, MagicMock will create the method if accessed, 
    # so we might not see an AttributeError unless we spec it.
    # But the real TopicService doesn't have it.
    
    # Let's try to simulate the real service behavior or just check the call.
    
    payload = {
        "title": "New Topic",
        "department": "CS",
        "option": "AI",
        "year": 2024,
        "category": "AI"
    }
    
    # If we don't mock find_duplicate, calling it on the mock will succeed (MagicMock behavior).
    # But we want to verify that the code *tries* to call find_duplicate.
    
    response = client.post("/api/topics/", json=payload)
    
    # If the code calls find_duplicate, the mock should record it.
    mock_topic_service.find_duplicate.assert_called_once()
    
    # However, the user wants us to FIX it. 
    # So this test confirms that the code is indeed calling find_duplicate.
    
def test_list_topics(mock_topic_service):
    mock_topic_service.list_topics.return_value = []
    response = client.get("/api/topics/")
    assert response.status_code == 200
    mock_topic_service.list_topics.assert_called_once()

def test_get_statistics(mock_topic_service):
    mock_topic_service.get_statistics.return_value = {}
    response = client.get("/api/topics/stats/overview")
    assert response.status_code == 200
    mock_topic_service.get_statistics.assert_called_once()
