
import pytest
from services.topic_service import TopicService
from unittest.mock import MagicMock

def test_search_topics_arguments():
    """
    Verify that search_topics accepts all the new arguments without crashing.
    """
    service = TopicService()
    # Mock the mongo service to avoid actual DB calls
    service.mongo = MagicMock()
    service.mongo.db = MagicMock()
    service.mongo.db['topics'] = MagicMock()
    
    # Mock find return
    mock_cursor = MagicMock()
    mock_cursor.limit.return_value.sort.return_value = []
    service.mongo.db['topics'].find.return_value = mock_cursor
    
    # This call should NOT raise TypeError
    results = service.search_topics(
        query="test",
        department="Computer Engineering",
        option="Software",
        year=2024,
        status="approved",
        limit=10
    )
    
    assert isinstance(results, list)
    
    # Verify arguments were used in filter
    call_args = service.mongo.db['topics'].find.call_args
    assert call_args is not None
    query_arg = call_args[0][0]
    
    assert query_arg['department'] == "Computer Engineering"
    assert query_arg['option'] == "Software"
    assert query_arg['year'] == 2024
    assert query_arg['status'] == "approved"
