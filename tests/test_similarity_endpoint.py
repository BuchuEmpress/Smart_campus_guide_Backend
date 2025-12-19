import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from api.main import app # Assuming 'app' is your FastAPI instance
from services.topic_intelligence_service import TopicIntelligenceService

from api.routes.topics import topic_ai as real_topic_ai # Import the real topic_ai instance

client = TestClient(app)

@pytest.fixture
def mock_topic_ai_instance():
    """Mock the topic_ai instance directly."""
    # Create a mock object that will replace real_topic_ai
    mock_instance = AsyncMock(spec=real_topic_ai)
    
    # Patch the real_topic_ai instance in api.routes.topics
    with patch('api.routes.topics.topic_ai', new=mock_instance):
        yield mock_instance

@pytest.mark.asyncio
async def test_check_similarity_endpoint(mock_topic_ai_instance):
    """
    Test the /api/topics/ai/similarity endpoint.
    """
    # Define the mock return value for check_similarity
    mock_topic_ai_instance.check_similarity.return_value = [
        {
            'topic_id': 'topic_123',
            'title': 'Mock Similar Topic 1',
            'similarity_score': 0.95,
            'status': 'available'
        },
        {
            'topic_id': 'topic_456',
            'title': 'Mock Similar Topic 2',
            'similarity_score': 0.88,
            'status': 'reserved'
        }
    ]

    # Define the request payload
    request_payload = {
        "title": "My New Topic Idea",
        "option": "SEN",
        "subgroup": "AI",
        "threshold": 0.8
    }

    # Make the API call
    response = client.post("/api/topics/ai/similarity", json=request_payload)

    # Assertions
    assert response.status_code == 200
    response_data = response.json()

    # Verify that check_similarity was called with the correct arguments
    mock_topic_ai_instance.check_similarity.assert_awaited_once_with(
        title="My New Topic Idea",
        option="AI", # Should use subgroup if provided
        threshold=0.8
    )

    # Verify the response structure and content
    assert "similar_topics" in response_data
    assert len(response_data["similar_topics"]) == 2
    assert response_data["similar_topics"][0]["title"] == "Mock Similar Topic 1"
    assert response_data["similar_topics"][0]["similarity_score"] == 0.95
    assert response_data["similar_topics"][1]["topic_id"] == "topic_456"

@pytest.mark.asyncio
async def test_check_similarity_endpoint_no_subgroup(mock_topic_ai_instance):
    """
    Test the /api/topics/ai/similarity endpoint when no subgroup is provided.
    Should use the 'option' field for the similarity check.
    """
    mock_topic_ai_instance.check_similarity.return_value = []

    request_payload = {
        "title": "Another Topic Idea",
        "option": "DAS",
        "threshold": 0.7
    }

    response = client.post("/api/topics/ai/similarity", json=request_payload)

    assert response.status_code == 200
    mock_topic_ai_instance.check_similarity.assert_awaited_once_with(
        title="Another Topic Idea",
        option="DAS", # Should use option if no subgroup
        threshold=0.7
    )
    assert response.json()["similar_topics"] == []

@pytest.mark.asyncio
async def test_check_similarity_endpoint_no_results(mock_topic_ai_instance):
    """
    Test the /api/topics/ai/similarity endpoint when no similar topics are found.
    """
    mock_topic_ai_instance.check_similarity.return_value = []

    request_payload = {
        "title": "Unique Topic",
        "option": "CNSM",
        "threshold": 0.6
    }

    response = client.post("/api/topics/ai/similarity", json=request_payload)

    assert response.status_code == 200
    response_data = response.json()
    assert "similar_topics" in response_data
    assert len(response_data["similar_topics"]) == 0
    mock_topic_ai_instance.check_similarity.assert_awaited_once()

@pytest.mark.asyncio
async def test_check_similarity_endpoint_error_handling(mock_topic_ai_instance):
    """
    Test error handling for the /api/topics/ai/similarity endpoint.
    """
    mock_topic_ai_instance.check_similarity.side_effect = Exception("Service error")

    request_payload = {
        "title": "Error Prone Topic",
        "option": "SEN",
        "threshold": 0.7
    }

    response = client.post("/api/topics/ai/similarity", json=request_payload)

    # The endpoint catches exceptions and returns an empty list, so status code should still be 200
    # The error logging would happen internally.
    assert response.status_code == 200
    response_data = response.json()
    assert "similar_topics" in response_data
    assert len(response_data["similar_topics"]) == 0
    mock_topic_ai_instance.check_similarity.assert_awaited_once()

