"""
Tests for Gemini Service

Tests all Gemini AI functionality:
- Direction humanization (converting robotic GPS to natural language)
- Intent extraction
- Description enhancement
- Prompt quality verification
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock 
from services.gemini_service import GeminiService


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_gemini_client():
    """Mock the Gemini Client to avoid real API calls."""
    # Patch the Client class where it is imported in gemini_service.py
    # We try patching 'google.genai.Client' as that's what the service likely uses
    with patch('google.genai.Client') as MockClient:
        # Create a mock instance for the client
        mock_client_instance = MockClient.return_value
        
        # Mock the async method models.generate_content to return an awaitable mock
        # This will be used as the default mock response
        mock_response = MagicMock()
        mock_response.text = "Mocked Response Text"
        mock_generate_content = AsyncMock(return_value=mock_response)
        
        # Set up the mock client instance's structure: client.models.generate_content
        # Note: We use models.generate_content for text generation methods
        mock_client_instance.models.generate_content = mock_generate_content
        
        yield mock_client_instance # Yield the mock instance


@pytest.fixture
def gemini_service(mock_gemini_client):
    """Create GeminiService instance with mocked client."""
    # We also need to patch the synchronous Client used for initialization
    with patch('google.genai.Client'):
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key_123'}):
            service = GeminiService()
            # Set the mock client instance directly for easier test access
            service.client = mock_gemini_client 
            return service


@pytest.fixture
def sample_route_data():
    """Sample route data from Google Maps."""
    return {
        'steps': [
            {
                'instruction': 'Head north on Main St',
                'distance': {'meters': 150, 'text': '150 m'},
                'duration': {'seconds': 120, 'text': '2 mins'}
            },
            {
                'instruction': 'Turn right onto Campus Ave',
                'distance': {'meters': 350, 'text': '350 m'},
                'duration': {'seconds': 240, 'text': '4 mins'}
            }
        ],
        'distance': {'meters': 500, 'text': '500 m'},
        'duration': {'seconds': 360, 'text': '6 mins'}
    }


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

def test_gemini_service_initialization_with_key():
    """Test GeminiService initializes correctly with API key."""
    # Patch the Client used in __init__
    with patch('google.genai.Client') as MockClient:
        service = GeminiService(api_key='test_key')
        # Check that the client was instantiated
        MockClient.assert_called_once()


def test_gemini_service_initialization_from_env():
    """Test GeminiService reads API key from environment."""
    # Patch the Client used in __init__
    with patch('google.genai.Client') as MockClient:
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'env_key'}):
            service = GeminiService()
            # Check that the client was instantiated
            MockClient.assert_called_once()


def test_gemini_service_initialization_no_key():
    """Test GeminiService raises error when no API key provided."""
    with patch.dict('os.environ', {}, clear=True):
        # The expected assertion has been correctly updated to the actual service error message.
        with pytest.raises(ValueError, match="GEMINI_API_KEY must be set"):
            GeminiService()


# ============================================================================
# DIRECTION HUMANIZATION TESTS
# ============================================================================

@pytest.mark.asyncio 
async def test_humanize_directions_basic(gemini_service, sample_route_data):
    """Test basic direction humanization."""
    # Mock Gemini response
    mock_response = MagicMock()
    mock_response.text = (
        "Walk straight ahead from where you are. You'll pass the cafeteria on your right. "
        "Keep going for about 5 minutes, and the library will be on your left - "
        "it's the tall building with glass windows."
    )
    gemini_service.client.models.generate_content = AsyncMock(return_value=mock_response)
    
    result = await gemini_service.humanize_directions(sample_route_data)
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    # Check that result doesn't contain robotic phrases
    assert "northeast" not in result.lower()
    assert "150 meters" not in result.lower() # Case-insensitive check


@pytest.mark.asyncio 
async def test_humanize_directions_with_landmarks(gemini_service, sample_route_data):
    """Test direction humanization with campus landmarks."""
    mock_response = MagicMock()
    mock_response.text = (
        "From the main gate, walk towards the cafeteria. "
        "You'll see students gathering near the fountain. "
        "Pass by the sports field, and the library is right there."
    )
    gemini_service.client.models.generate_content = AsyncMock(return_value=mock_response)
    
    landmarks = ["Main Gate", "Cafeteria", "Fountain", "Sports Field"]
    result = await gemini_service.humanize_directions(
        sample_route_data,
        nearby_landmarks=landmarks
    )
    
    assert result is not None
    assert len(result) > 0


@pytest.mark.asyncio 
async def test_humanize_directions_campus_context(gemini_service, sample_route_data):
    """Test direction humanization with campus context."""
    mock_response = MagicMock()
    mock_response.text = "Walk across the campus quad towards the library."
    gemini_service.client.models.generate_content = AsyncMock(return_value=mock_response)
    
    campus_context = {
        'is_on_campus': True,
        'campus_name': 'University of Bamenda'
    }
    
    result = await gemini_service.humanize_directions(
        sample_route_data,
        campus_context=campus_context
    )
    
    assert result is not None
    assert len(result) > 0


@pytest.mark.asyncio 
async def test_humanize_directions_no_robotic_phrases(gemini_service, sample_route_data):
    """Test that humanized directions don't contain robotic GPS phrases."""
    mock_response = MagicMock()
    mock_response.text = (
        "Walk straight ahead. You'll pass the cafeteria on your right. "
        "The library is about a 5-minute walk from here."
    )
    gemini_service.client.models.generate_content = AsyncMock(return_value=mock_response)
    
    result = await gemini_service.humanize_directions(sample_route_data)
    
    # Robotic phrases that should NOT appear
    robotic_phrases = [
        "proceed north",
        "head northeast",
        "turn left in 150 meters",
        "walk 200 meters",
        "45 degrees",
        "bearing"
    ]
    
    result_lower = result.lower()
    for phrase in robotic_phrases:
        assert phrase not in result_lower, f"Found robotic phrase: {phrase}"


@pytest.mark.asyncio 
async def test_humanize_directions_api_error(gemini_service, sample_route_data):
    """Test direction humanization handles API errors gracefully."""
    gemini_service.client.models.generate_content = AsyncMock(
        side_effect=Exception("API Error")
    )
    
    result = await gemini_service.humanize_directions(sample_route_data)
    
    # The actual service returns a fallback string, not None
    assert isinstance(result, str)
    assert "help you get there" in result


# ============================================================================
# INTENT EXTRACTION TESTS
# ============================================================================

@pytest.mark.asyncio 
async def test_extract_intent_on_campus_location(gemini_service):
    """Test extracting intent for on-campus location query."""
    mock_response = MagicMock()
    # Ensure JSON matches the structure the service *expects* (from the prompt)
    mock_response.text = '{"action": "search", "location_query": "library", "location_type": "building", "preferences": {}, "on_campus": true}'
    gemini_service.client.models.generate_content = AsyncMock(return_value=mock_response)
    
    result = await gemini_service.extract_intent("Where is the library?")
    
    assert result is not None
    assert isinstance(result, dict)
    assert result.get('action') == 'search'


@pytest.mark.asyncio 
async def test_extract_intent_off_campus_location(gemini_service):
    """Test extracting intent for off-campus location query."""
    mock_response = MagicMock()
    mock_response.text = '{"action": "search", "location_query": "hospital", "location_type": "other", "preferences": {}, "on_campus": false}'
    gemini_service.client.models.generate_content = AsyncMock(return_value=mock_response)
    
    result = await gemini_service.extract_intent("Where is the nearest hospital?")
    
    assert result is not None
    assert isinstance(result, dict)
    assert result.get('on_campus') == False


@pytest.mark.asyncio 
async def test_extract_intent_navigation_request(gemini_service):
    """Test extracting intent for navigation request."""
    mock_response = MagicMock()
    mock_response.text = '{"action": "navigate", "location_query": "cafeteria", "location_type": "food", "preferences": {}, "urgency": "normal"}'
    gemini_service.client.models.generate_content = AsyncMock(return_value=mock_response)
    
    result = await gemini_service.extract_intent("How do I get to the cafeteria?")
    
    assert result is not None
    assert result.get('action') == 'navigate'


@pytest.mark.asyncio 
async def test_extract_intent_empty_query(gemini_service):
    """Test extracting intent from empty query returns fallback."""
    # Since the underlying AI would likely return a search intent for an empty query, 
    # we rely on the service's error handling for non-string prompts which is complex.
    # For a robust test, we mock the expected non-JSON fallback response.
    mock_response = MagicMock()
    mock_response.text = "" # Empty response simulating failure/no content
    gemini_service.client.models.generate_content = AsyncMock(return_value=mock_response)
    
    result = await gemini_service.extract_intent("")
    
    assert isinstance(result, dict)
    assert result.get('action') == 'chat' # Fallback intent for empty query is chat


@pytest.mark.asyncio 
async def test_extract_intent_api_error(gemini_service):
    """Test intent extraction handles API errors gracefully."""
    gemini_service.client.models.generate_content = AsyncMock(
        side_effect=Exception("API Error")
    )
    
    result = await gemini_service.extract_intent("Where is the library?")
    
    # Should return the fallback intent dictionary
    assert isinstance(result, dict)
    assert result.get('action') == 'search'


# ============================================================================
# DESCRIPTION ENHANCEMENT TESTS
# ============================================================================

@pytest.mark.asyncio 
async def test_enhance_description_basic(gemini_service):
    """Test basic description enhancement."""
    mock_response = MagicMock()
    mock_response.text = (
        "The University Library is a modern three-story building with extensive "
        "study spaces, computer labs, and a quiet reading room. It's located in "
        "the center of campus near the main quad."
    )
    gemini_service.client.models.generate_content = AsyncMock(return_value=mock_response)
    
    location = {
        'name': 'University Library',
        'type': 'building',
        'description': 'Main library building'
    }
    
    result = await gemini_service.enhance_description(location)
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > len(location['description'])


@pytest.mark.asyncio 
async def test_enhance_description_with_context(gemini_service):
    """Test description enhancement with additional context."""
    mock_response = MagicMock()
    mock_response.text = "A popular study spot with great Wi-Fi and air conditioning."
    gemini_service.client.models.generate_content = AsyncMock(return_value=mock_response)
    
    location = {'name': 'Library', 'type': 'building', 'description': 'Main library building'}
    context = {'time_of_day': 'evening', 'weather': 'hot'}
    
    result = await gemini_service.enhance_description(location, context=context)
    
    assert result is not None
    assert len(result) > 0


@pytest.mark.asyncio
async def test_enhance_description_empty_location(gemini_service):
    """Test enhancing description with empty location returns fallback."""
    location = {}
    # The service handles KeyError internally and returns a fallback string
    result = await gemini_service.enhance_description(location)
    assert isinstance(result, str)
    # Check for the expected fallback content (or part of it)
    assert "Unable to enhance" in result


# ============================================================================
# GENERAL RESPONSE GENERATION TESTS
# ============================================================================

@pytest.mark.asyncio 
async def test_generate_response_simple_prompt(gemini_service):
    """Test generating response from simple prompt."""
    mock_response = MagicMock()
    mock_response.text = "The library is open from 8 AM to 10 PM on weekdays."
    gemini_service.client.models.generate_content = AsyncMock(return_value=mock_response)
    
    result = await gemini_service.generate_response("What are the library hours?")
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio 
async def test_generate_response_with_context(gemini_service):
    """Test generating response with conversation context."""
    mock_response = MagicMock()
    mock_response.text = "Yes, the library has computers available for student use."
    gemini_service.client.models.generate_content = AsyncMock(return_value=mock_response)
    
    context = [
        {'role': 'user', 'content': 'Where is the library?'},
        {'role': 'assistant', 'content': 'The library is in the main quad.'}
    ]
    
    result = await gemini_service.generate_response(
        "Does it have computers?",
        context=context
    )
    
    assert result is not None


@pytest.mark.asyncio
async def test_generate_response_empty_prompt(gemini_service):
    """Test generating response with empty prompt returns fallback."""
    # The service returns a fallback message in the generate_response error handler
    result = await gemini_service.generate_response("")
    assert isinstance(result, str)
    # Check for the expected fallback content (or part of it)
    assert "rephrasing your question" in result


# ============================================================================
# PROMPT QUALITY TESTS
# ============================================================================

@pytest.mark.asyncio 
async def test_humanization_prompt_includes_rules(gemini_service, sample_route_data):
    """Test that humanization prompt includes quality rules."""
    mock_response = MagicMock()
    mock_response.text = "Natural directions here"
    
    mock_generate_content = AsyncMock(return_value=mock_response)
    gemini_service.client.models.generate_content = mock_generate_content
    
    await gemini_service.humanize_directions(sample_route_data)
    
    # Get the actual prompt sent to Gemini
    mock_generate_content.assert_called_once()
    call_args, call_kwargs = mock_generate_content.call_args
    prompt = call_args[0]
    
    # Verify prompt includes important instructions
    assert "NEVER use compass directions" in prompt
    assert "ALWAYS use visible landmarks" in prompt
    assert "Write in friendly, conversational English" in prompt


@pytest.mark.asyncio 
async def test_multiple_calls_same_service(gemini_service, sample_route_data):
    """Test that service can handle multiple calls."""
    mock_response_humanize = MagicMock()
    mock_response_humanize.text = "Natural directions"
    
    mock_response_intent = MagicMock()
    # Mock return valid JSON
    mock_response_intent.text = '{"action": "search", "location_query": "library", "location_type": "building", "preferences": {}, "on_campus": true}' 
    
    mock_response_general = MagicMock()
    mock_response_general.text = "Hello back"
    
    # Use side_effect to provide different return values for sequential calls
    gemini_service.client.models.generate_content = AsyncMock(side_effect=[
        mock_response_humanize,
        mock_response_intent,
        mock_response_general
    ])
    
    # Make multiple calls
    result1 = await gemini_service.humanize_directions(sample_route_data)
    result2 = await gemini_service.extract_intent("Where is the library?")
    result3 = await gemini_service.generate_response("Hello")
    
    assert result1 is not None
    # This assertion was the fix for the previous bug: ensuring intent returns a dict
    assert isinstance(result2, dict) 
    assert result3 is not None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.asyncio 
async def test_handles_network_error(gemini_service):
    """Test service handles network errors gracefully."""
    # The service catches Exception (which includes ConnectionError)
    gemini_service.client.models.generate_content = AsyncMock(
        side_effect=ConnectionError("Network error")
    )
    
    result = await gemini_service.generate_response("test prompt")
    # Should return fallback string
    assert isinstance(result, str)


@pytest.mark.asyncio 
async def test_handles_timeout_error(gemini_service, sample_route_data):
    """Test service handles timeout errors gracefully."""
    # The service catches Exception (which includes TimeoutError)
    gemini_service.client.models.generate_content = AsyncMock(
        side_effect=TimeoutError("Request timeout")
    )
    
    result = await gemini_service.humanize_directions(sample_route_data)
    # Should return fallback string
    assert isinstance(result, str)


@pytest.mark.asyncio 
async def test_handles_invalid_json_response(gemini_service):
    """Test service handles invalid JSON in API response."""
    mock_response = MagicMock()
    mock_response.text = "This is not valid JSON"
    gemini_service.client.models.generate_content = AsyncMock(return_value=mock_response)
    
    result = await gemini_service.extract_intent("test query")
    # Should handle gracefully by returning the fallback intent
    assert isinstance(result, dict)
    assert result.get('action') == 'search' # Fallback intent is returned


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])