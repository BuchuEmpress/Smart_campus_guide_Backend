# tests/conftest.py

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

# =======================================================================
# 🚀 CRITICAL FIX: PATH CORRECTION MUST BE FIRST
# =======================================================================
# Temporarily add the project root to the path 
sys.path.insert(0, str(Path(__file__).parent.parent))

# =======================================================================
# 🚀 ENVIRONMENT SETUP (Mock Env Vars BEFORE App Import)
# =======================================================================
os.environ["GEMINI_API_KEY"] = "test_gemini_key"
os.environ["QDRANT_HOST"] = "http://localhost:6333"
os.environ["QDRANT_API_KEY"] = "test_qdrant_key"
os.environ["GOOGLE_MAPS_API_KEY"] = "test_maps_key"
os.environ["MONGODB_URI"] = "mongodb://localhost:27017"
os.environ["DEBUG_MODE"] = "True"

# =======================================================================
# 🛑 PATCH MEMORY-INTENSIVE SERVICES (SECOND STEP)
# =======================================================================

# 1. Patch the TopicIntelligenceService's memory-heavy dependency 
# (Assuming the SentenceTransformer model is the memory bottleneck)
with patch('services.topic_intelligence_service.SentenceTransformer'):
    
    # 2. Patch the GeminiService class, targeting its presumed definition module.
    # The previous error confirms it's not an attribute of qdrant_service.
    # We must patch the class itself (likely defined in services/gemini_service.py)
    with patch('services.gemini_service.GeminiService'):
        
        # 3. Import the main app
        from api.main import app
        
# Patches automatically end here.


# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """
    Create a synchronous TestClient for testing FastAPI endpoints.
    """
    return TestClient(app) # 🚀 Use TestClient for stable API testing


@pytest.fixture
def mock_qdrant():
    """Mock QdrantService using AsyncMock for awaitable methods."""
    with patch('api.routes.navigation.QdrantService') as mock:
        instance = MagicMock()
        
        instance.search = AsyncMock(return_value=[ 
            {
                'score': 0.95,
                'id': 'loc_1',
                'name': 'Asanji Hall',
                'description': 'Main lecture hall',
                'latitude': 6.010380,
                'longitude': 10.259017,
                'type': 'class'
            }
        ])
        instance.get_location_by_id = AsyncMock(return_value={
                'score': 0.95,
                'id': 'loc_1',
                'name': 'Asanji Hall',
                'description': 'Main lecture hall',
                'latitude': 6.010380,
                'longitude': 10.259017,
                'type': 'class'
            })
        
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_osm():
    """Mock OSMRoutingService using AsyncMock for awaitable methods."""
    with patch('api.routes.navigation.OSMRoutingService') as mock:
        instance = MagicMock()
        
        instance.get_route = AsyncMock(return_value={
            'distance': 100,
            'distance_text': '100 meters',
            'duration': 60,
            'duration_text': '1 minute',
            'steps': [],
            'geometry': {},
            'profile': 'foot',
            'path': 'some_path'  # Add a dummy path
        })
        instance.search_place = AsyncMock(return_value=[
            {
                'name': 'Test Place',
                'display_name': 'Test Place, Bamenda',
                'latitude': 5.9631,
                'longitude': 10.1591,
                'type': 'place'
            }
        ])
        instance.geocode = AsyncMock(return_value={
            'latitude': 6.010317,
            'longitude': 10.258816,
            'address': 'Main Gate'
        })
        
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_gemini():
    """Mock GeminiService using AsyncMock for awaitable methods."""
    # This patches where it is used in the routes, not where it is defined/loaded
    with patch('api.routes.navigation.GeminiService') as mock:
        instance = MagicMock()
        
        instance.humanize_directions = AsyncMock(return_value="Walk straight to Asanji Hall")
        instance.extract_intent = AsyncMock(return_value={'action': 'navigate'})
        instance.generate_response = AsyncMock(return_value="I can help you find that")
        instance.enhance_description = AsyncMock(return_value="Enhanced description")
        
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_analytics():
    """Mock AnalyticsService (assumed synchronous)."""
    with patch('api.routes.navigation.AnalyticsService') as mock:
        instance = MagicMock()
        instance.log_search.return_value = None
        mock.return_value = instance
        yield mock