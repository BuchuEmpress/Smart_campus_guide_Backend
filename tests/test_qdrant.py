"""
Tests for the QdrantService class.

These tests ensure the service interacts correctly with the Qdrant client
and handles asynchronous operations, errors, and data transformation.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch 
from qdrant_client.http.exceptions import UnexpectedResponse

from services.qdrant_service import QdrantService
from qdrant_client.models import PointStruct, Distance


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_gemini_service():
    """Mock GeminiService."""
    with patch('services.qdrant_service.GeminiService') as mock:
        instance = MagicMock()
        instance.embed_text = AsyncMock(return_value=[0.1] * 768)
        instance.embed_batch = AsyncMock(return_value=[[0.1] * 768])
        mock.return_value = instance
        yield instance

@pytest.fixture
def mock_qdrant_client():
    """Mock the synchronous Qdrant Client (which is used via to_thread)."""
    with patch('services.qdrant_service.QdrantClient') as MockClient:
        mock_client_instance = MockClient.return_value
        
        # Mock methods (they are sync in QdrantClient)
        mock_client_instance.collection_exists = MagicMock(return_value=True)
        mock_client_instance.delete_collection = MagicMock(return_value=None)
        mock_client_instance.create_collection = MagicMock(return_value=None)
        mock_client_instance.upsert = MagicMock(return_value=None)
        mock_client_instance.query_points = MagicMock() 
        mock_client_instance.scroll = MagicMock(return_value=([], None))
        
        # Mock get_collection
        mock_client_instance.get_collection = MagicMock(return_value=MagicMock(
            status='green',
            optimizer_status='ok',
            vectors_count=1,
            points_count=1,
            segments_count=1,
            config=MagicMock(
                params=MagicMock(
                    vectors=MagicMock(
                        size=768,
                        distance=Distance.COSINE
                    )
                )
            )
        ))
        
        yield mock_client_instance

@pytest.fixture
def qdrant_service(mock_qdrant_client, mock_gemini_service):
    """Create a QdrantService instance with a mocked client."""
    with patch.dict('os.environ', {
        'QDRANT_HOST': 'http://test:6333', 
        'QDRANT_API_KEY': 'test_key',
        'GEMINI_API_KEY': 'test_gemini_key'
    }):
        service = QdrantService()
        service.client = mock_qdrant_client 
        service.gemini = mock_gemini_service
        service.collection_name = "test_collection"
        return service


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

def test_qdrant_service_initialization(qdrant_service):
    """Test QdrantService initializes correctly."""
    assert isinstance(qdrant_service, QdrantService)
    assert qdrant_service.collection_name == "test_collection"
    assert qdrant_service.client is not None 
    # Gemini service should be initialized
    assert qdrant_service.gemini is not None


def test_qdrant_service_initialization_no_env():
    """Test QdrantService raises error when required environment variables are missing."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="Missing QDRANT_HOST or QDRANT_API_KEY"):
            QdrantService()


@pytest.fixture
def qdrant_test_data():
    """Sample data structure for Qdrant payload."""
    return [
        {
            "id": "loc_1",
            "name": "Library Hall",
            "description": "Main library entrance",
            "latitude": 6.01,
            "longitude": 10.26,
            "type": "building"
        }
    ]


# ============================================================================
# ASYNCHRONOUS METHOD TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_ensure_collection_exists_success(qdrant_service, mock_qdrant_client):
    """Test collection creation when it doesn't exist."""
    # Arrange: Mock collection_exists to return False
    mock_qdrant_client.collection_exists.return_value = False
    
    # Act
    result = await qdrant_service.create_collection(force_recreate=True)
    
    # Assert
    assert result == True
    mock_qdrant_client.create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_collection_exists_already_present(qdrant_service, mock_qdrant_client):
    """Test collection check when it already exists."""
    # Arrange: Mock collection_exists to return True
    mock_qdrant_client.collection_exists.return_value = True
    
    # Act
    result = await qdrant_service.create_collection(force_recreate=False)
    
    # Assert
    assert result == False
    mock_qdrant_client.collection_exists.assert_called_once()
    mock_qdrant_client.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_data_success(qdrant_service, mock_qdrant_client, qdrant_test_data):
    """Test upserting data successfully."""
    # Act
    await qdrant_service.upload_points(qdrant_test_data)
    
    # Assert
    mock_qdrant_client.upsert.assert_called_once()
    
    # Check the points structure
    call_kwargs = mock_qdrant_client.upsert.call_args[1]
    assert call_kwargs['collection_name'] == 'test_collection'
    
    points_arg = call_kwargs['points']
    assert isinstance(points_arg[0], PointStruct)
    assert points_arg[0].id == "loc_1"
    assert points_arg[0].payload['name'] == "Library Hall"


@pytest.mark.asyncio
async def test_search_success(qdrant_service, mock_qdrant_client):
    """Test a successful search operation."""
    # Arrange: Mock the Qdrant query_points result
    mock_response = MagicMock()
    mock_response.points = [
        MagicMock(
            score=0.9,
            id='loc_2',
            payload={
                'id': 'loc_2',
                'name': 'Cafeteria',
                'description': 'Main dining hall',
                'latitude': 6.02,
                'longitude': 10.27,
                'type': 'food'
            }
        )
    ]
    mock_qdrant_client.query_points.return_value = mock_response
    
    # Act
    results = await qdrant_service.search("Where can I eat?")
    
    # Assert
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]['name'] == 'Cafeteria'
    assert results[0]['score'] == 0.9
    mock_qdrant_client.query_points.assert_called_once()


@pytest.mark.asyncio
async def test_search_api_error(qdrant_service, mock_qdrant_client):
    """Test search gracefully handles Qdrant API errors."""
    # Arrange: Mock query_points to raise an error
    mock_qdrant_client.query_points.side_effect = Exception("Qdrant error")
    
    # Act
    results = await qdrant_service.search("test query")
    
    # Assert - service returns empty list on search error
    assert results == []
    mock_qdrant_client.query_points.assert_called_once()


@pytest.mark.asyncio
async def test_get_by_id_success(qdrant_service, mock_qdrant_client):
    """Test retrieving a point by ID successfully."""
    # Arrange: Mock Qdrant's scroll method
    mock_item = MagicMock()
    mock_item.payload = {
        'id': 'test_id_1',
        'name': 'Admin Building',
        'description': 'Office',
        'latitude': 6.01,
        'longitude': 10.26,
        'type': 'office'
    }
    mock_qdrant_client.scroll.return_value = ([mock_item], None)
    
    # Act
    location = await qdrant_service.get_by_id('test_id_1')
    
    # Assert
    assert location['name'] == 'Admin Building'
    mock_qdrant_client.scroll.assert_called_once()


@pytest.mark.asyncio
async def test_get_by_id_not_found(qdrant_service, mock_qdrant_client):
    """Test retrieving a point that does not exist."""
    # Arrange: Mock scroll to return empty list
    mock_qdrant_client.scroll.return_value = ([], None)
    
    # Act
    location = await qdrant_service.get_by_id('missing_id')
    
    # Assert
    assert location is None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])