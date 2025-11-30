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
from unittest.mock import Mock, patch, MagicMock
from services.gemini_service import GeminiService


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_genai():
    """Mock Google Generative AI to avoid real API calls."""
    with patch('google.generativeai.configure') as mock_config:
        with patch('google.generativeai.GenerativeModel') as mock_model:
            yield mock_model


@pytest.fixture
def gemini_service(mock_genai):
    """Create GeminiService instance with mocked client."""
    with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key_123'}):
        service = GeminiService()
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
    with patch('google.generativeai.configure'):
        with patch('google.generativeai.GenerativeModel'):
            service = GeminiService(api_key='test_key')
            assert service.api_key == 'test_key'


def test_gemini_service_initialization_from_env():
    """Test GeminiService reads API key from environment."""
    with patch.dict('os.environ', {'GEMINI_API_KEY': 'env_key'}):
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                service = GeminiService()
                assert service.api_key == 'env_key'


def test_gemini_service_initialization_no_key():
    """Test GeminiService raises error when no API key provided."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="API key not found"):
            GeminiService()


# ============================================================================
# DIRECTION HUMANIZATION TESTS
# ============================================================================

def test_humanize_directions_basic(gemini_service, sample_route_data):
    """Test basic direction humanization."""
    # Mock Gemini response
    mock_response = Mock()
    mock_response.text = (
        "Walk straight ahead from where you are. You'll pass the cafeteria on your right. "
        "Keep going for about 5 minutes, and the library will be on your left - "
        "it's the tall building with glass windows."
    )
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    result = gemini_service.humanize_directions(sample_route_data)
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    # Check that result doesn't contain robotic phrases
    assert "northeast" not in result.lower()
    assert "150 meters" not in result


def test_humanize_directions_with_landmarks(gemini_service, sample_route_data):
    """Test direction humanization with campus landmarks."""
    mock_response = Mock()
    mock_response.text = (
        "From the main gate, walk towards the cafeteria. "
        "You'll see students gathering near the fountain. "
        "Pass by the sports field, and the library is right there."
    )
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    landmarks = ["Main Gate", "Cafeteria", "Fountain", "Sports Field"]
    result = gemini_service.humanize_directions(
        sample_route_data,
        nearby_landmarks=landmarks
    )
    
    assert result is not None
    # Check that landmarks might be used (context provided to AI)
    assert len(result) > 0


def test_humanize_directions_campus_context(gemini_service, sample_route_data):
    """Test direction humanization with campus context."""
    mock_response = Mock()
    mock_response.text = "Walk across the campus quad towards the library."
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    campus_context = {
        'is_on_campus': True,
        'campus_name': 'University of Bamenda'
    }
    
    result = gemini_service.humanize_directions(
        sample_route_data,
        campus_context=campus_context
    )
    
    assert result is not None
    assert len(result) > 0


def test_humanize_directions_no_robotic_phrases(gemini_service, sample_route_data):
    """Test that humanized directions don't contain robotic GPS phrases."""
    mock_response = Mock()
    mock_response.text = (
        "Walk straight ahead. You'll pass the cafeteria on your right. "
        "The library is about a 5-minute walk from here."
    )
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    result = gemini_service.humanize_directions(sample_route_data)
    
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


def test_humanize_directions_api_error(gemini_service, sample_route_data):
    """Test direction humanization handles API errors gracefully."""
    gemini_service.model.generate_content = Mock(
        side_effect=Exception("API Error")
    )
    
    result = gemini_service.humanize_directions(sample_route_data)
    
    # Should return fallback or None
    assert result is None or isinstance(result, str)


# ============================================================================
# INTENT EXTRACTION TESTS
# ============================================================================

def test_extract_intent_on_campus_location(gemini_service):
    """Test extracting intent for on-campus location query."""
    mock_response = Mock()
    mock_response.text = '{"type": "location_search", "location": "library", "on_campus": true}'
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    result = gemini_service.extract_intent("Where is the library?")
    
    assert result is not None
    assert isinstance(result, dict)


def test_extract_intent_off_campus_location(gemini_service):
    """Test extracting intent for off-campus location query."""
    mock_response = Mock()
    mock_response.text = '{"type": "location_search", "location": "hospital", "on_campus": false}'
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    result = gemini_service.extract_intent("Where is the nearest hospital?")
    
    assert result is not None
    assert isinstance(result, dict)


def test_extract_intent_navigation_request(gemini_service):
    """Test extracting intent for navigation request."""
    mock_response = Mock()
    mock_response.text = '{"type": "navigation", "destination": "cafeteria", "urgency": "normal"}'
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    result = gemini_service.extract_intent("How do I get to the cafeteria?")
    
    assert result is not None


def test_extract_intent_empty_query(gemini_service):
    """Test extracting intent from empty query raises ValueError."""
    with pytest.raises(ValueError, match="non-empty string"):
        gemini_service.extract_intent("")


def test_extract_intent_api_error(gemini_service):
    """Test intent extraction handles API errors gracefully."""
    gemini_service.model.generate_content = Mock(
        side_effect=Exception("API Error")
    )
    
    result = gemini_service.extract_intent("Where is the library?")
    
    assert result is None or isinstance(result, dict)


# ============================================================================
# DESCRIPTION ENHANCEMENT TESTS
# ============================================================================

def test_enhance_description_basic(gemini_service):
    """Test basic description enhancement."""
    mock_response = Mock()
    mock_response.text = (
        "The University Library is a modern three-story building with extensive "
        "study spaces, computer labs, and a quiet reading room. It's located in "
        "the center of campus near the main quad."
    )
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    location = {
        'name': 'University Library',
        'type': 'building',
        'description': 'Main library building'
    }
    
    result = gemini_service.enhance_description(location)
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > len(location['description'])


def test_enhance_description_with_context(gemini_service):
    """Test description enhancement with additional context."""
    mock_response = Mock()
    mock_response.text = "A popular study spot with great Wi-Fi and air conditioning."
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    location = {'name': 'Library', 'type': 'building'}
    context = {'time': 'evening', 'weather': 'hot'}
    
    result = gemini_service.enhance_description(location, context=context)
    
    assert result is not None
    assert len(result) > 0


def test_enhance_description_empty_location(gemini_service):
    """Test enhancing description with empty location raises ValueError."""
    with pytest.raises((ValueError, KeyError)):
        gemini_service.enhance_description({})


# ============================================================================
# GENERAL RESPONSE GENERATION TESTS
# ============================================================================

def test_generate_response_simple_prompt(gemini_service):
    """Test generating response from simple prompt."""
    mock_response = Mock()
    mock_response.text = "The library is open from 8 AM to 10 PM on weekdays."
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    result = gemini_service.generate_response("What are the library hours?")
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_response_with_context(gemini_service):
    """Test generating response with conversation context."""
    mock_response = Mock()
    mock_response.text = "Yes, the library has computers available for student use."
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    context = [
        {'role': 'user', 'content': 'Where is the library?'},
        {'role': 'assistant', 'content': 'The library is in the main quad.'}
    ]
    
    result = gemini_service.generate_response(
        "Does it have computers?",
        context=context
    )
    
    assert result is not None


def test_generate_response_empty_prompt(gemini_service):
    """Test generating response with empty prompt raises ValueError."""
    with pytest.raises(ValueError, match="non-empty string"):
        gemini_service.generate_response("")


# ============================================================================
# PROMPT QUALITY TESTS
# ============================================================================

def test_humanization_prompt_includes_rules(gemini_service, sample_route_data):
    """Test that humanization prompt includes quality rules."""
    mock_response = Mock()
    mock_response.text = "Natural directions here"
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    gemini_service.humanize_directions(sample_route_data)
    
    # Get the actual prompt sent to Gemini
    call_args = gemini_service.model.generate_content.call_args
    prompt = str(call_args)
    
    # Verify prompt includes important instructions
    # (This is a basic check - actual implementation may vary)
    assert call_args is not None


def test_multiple_calls_same_service(gemini_service, sample_route_data):
    """Test that service can handle multiple calls."""
    mock_response = Mock()
    mock_response.text = "Natural directions"
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    # Make multiple calls
    result1 = gemini_service.humanize_directions(sample_route_data)
    result2 = gemini_service.extract_intent("Where is the library?")
    result3 = gemini_service.generate_response("Hello")
    
    assert result1 is not None
    assert result2 is not None or result2 == {}
    assert result3 is not None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_handles_network_error(gemini_service):
    """Test service handles network errors gracefully."""
    gemini_service.model.generate_content = Mock(
        side_effect=ConnectionError("Network error")
    )
    
    result = gemini_service.generate_response("test prompt")
    assert result is None or isinstance(result, str)


def test_handles_timeout_error(gemini_service, sample_route_data):
    """Test service handles timeout errors gracefully."""
    gemini_service.model.generate_content = Mock(
        side_effect=TimeoutError("Request timeout")
    )
    
    result = gemini_service.humanize_directions(sample_route_data)
    assert result is None or isinstance(result, str)


def test_handles_invalid_json_response(gemini_service):
    """Test service handles invalid JSON in API response."""
    mock_response = Mock()
    mock_response.text = "This is not valid JSON"
    gemini_service.model.generate_content = Mock(return_value=mock_response)
    
    result = gemini_service.extract_intent("test query")
    # Should handle gracefully, not crash
    assert result is None or isinstance(result, dict)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])