"""
Tests for OSM Routing Service

Tests all OpenStreetMap functionality:
- Route calculation
- Place search
- Geocoding
- Reverse geocoding
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from services.osm_routing_service import OSMRoutingService


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def osm_service():
    """Create OSM service instance."""
    return OSMRoutingService()


@pytest.fixture
def sample_route_response():
    """Sample OSRM route response."""
    return {
        'code': 'Ok',
        'routes': [{
            'distance': 500,
            'duration': 360,
            'legs': [{
                'steps': [
                    {
                        'distance': 150,
                        'duration': 120,
                        'name': 'Main Street',
                        'maneuver': {
                            'instruction': 'Head north',
                            'type': 'depart',
                            'modifier': ''
                        }
                    },
                    {
                        'distance': 350,
                        'duration': 240,
                        'name': 'Campus Road',
                        'maneuver': {
                            'instruction': 'Turn right',
                            'type': 'turn',
                            'modifier': 'right'
                        }
                    }
                ]
            }],
            'geometry': {
                'coordinates': [[10.1591, 5.9631], [10.1601, 5.9641]],
                'type': 'LineString'
            }
        }]
    }


@pytest.fixture
def sample_search_response():
    """Sample Nominatim search response."""
    return [
        {
            'name': 'Bamenda Regional Hospital',
            'display_name': 'Bamenda Regional Hospital, Mankon, Bamenda, Cameroon',
            'lat': '5.9639',
            'lon': '10.1580',
            'type': 'hospital',
            'class': 'amenity',
            'importance': 0.75,
            'place_id': '12345',
            'osm_type': 'node',
            'osm_id': '67890'
        }
    ]


# ============================================================================
# ROUTING TESTS
# ============================================================================

def test_get_route_success(osm_service, sample_route_response):
    """Test successful route calculation."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = sample_route_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        route = osm_service.get_route(
            start=(5.9631, 10.1591),
            end=(5.9641, 10.1601),
            profile='foot'
        )
        
        assert route is not None
        assert route['distance'] == 500
        assert route['duration'] == 360
        assert len(route['steps']) == 2
        assert 'distance_text' in route
        assert 'duration_text' in route


def test_get_route_invalid_start(osm_service):
    """Test route with invalid start coordinates."""
    with pytest.raises(ValueError, match="Invalid start"):
        osm_service.get_route(
            start=(95, 200),  # Invalid coordinates
            end=(5.9641, 10.1601)
        )


def test_get_route_invalid_end(osm_service):
    """Test route with invalid end coordinates."""
    with pytest.raises(ValueError, match="Invalid end"):
        osm_service.get_route(
            start=(5.9631, 10.1591),
            end=(95, 200)  # Invalid coordinates
        )


def test_get_route_invalid_profile(osm_service):
    """Test route with invalid profile."""
    with pytest.raises(ValueError, match="Invalid profile"):
        osm_service.get_route(
            start=(5.9631, 10.1591),
            end=(5.9641, 10.1601),
            profile='flying'  # Invalid profile
        )


def test_get_route_not_found(osm_service):
    """Test route when no route found."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            'code': 'NoRoute',
            'message': 'No route found'
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        route = osm_service.get_route(
            start=(5.9631, 10.1591),
            end=(5.9641, 10.1601)
        )
        
        assert route is None


def test_get_route_network_error(osm_service):
    """Test route with network error."""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = ConnectionError("Network error")
        
        route = osm_service.get_route(
            start=(5.9631, 10.1591),
            end=(5.9641, 10.1601)
        )
        
        assert route is None


def test_get_route_timeout(osm_service):
    """Test route with timeout."""
    with patch('requests.get') as mock_get:
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Timeout")
        
        route = osm_service.get_route(
            start=(5.9631, 10.1591),
            end=(5.9641, 10.1601)
        )
        
        assert route is None


# ============================================================================
# SEARCH TESTS
# ============================================================================

def test_search_place_success(osm_service, sample_search_response):
    """Test successful place search."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = sample_search_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch('time.sleep'):  # Skip sleep in tests
            places = osm_service.search_place("hospital")
        
        assert len(places) == 1
        assert places[0]['name'] == 'Bamenda Regional Hospital'
        assert places[0]['latitude'] == 5.9639
        assert places[0]['longitude'] == 10.1580


def test_search_place_with_location(osm_service, sample_search_response):
    """Test place search near a location."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = sample_search_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch('time.sleep'):
            places = osm_service.search_place(
                "hospital",
                near=(5.9631, 10.1591)
            )
        
        assert len(places) == 1
        # Verify location parameter was included
        call_args = mock_get.call_args
        assert 'lat' in call_args[1]['params']
        assert 'lon' in call_args[1]['params']


def test_search_place_empty_query(osm_service):
    """Test search with empty query."""
    with pytest.raises(ValueError, match="non-empty string"):
        osm_service.search_place("")


def test_search_place_not_found(osm_service):
    """Test search when no places found."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch('time.sleep'):
            places = osm_service.search_place("NonexistentPlace12345")
        
        assert places == []


def test_search_place_network_error(osm_service):
    """Test search with network error."""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = ConnectionError("Network error")
        
        with patch('time.sleep'):
            places = osm_service.search_place("hospital")
        
        assert places == []


# ============================================================================
# GEOCODING TESTS
# ============================================================================

def test_geocode_success(osm_service, sample_search_response):
    """Test successful geocoding."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = sample_search_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch('time.sleep'):
            result = osm_service.geocode("Bamenda, Cameroon")
        
        assert result is not None
        assert 'latitude' in result
        assert 'longitude' in result
        assert 'address' in result


def test_geocode_not_found(osm_service):
    """Test geocoding with address not found."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch('time.sleep'):
            result = osm_service.geocode("NonexistentPlace12345")
        
        assert result is None


# ============================================================================
# REVERSE GEOCODING TESTS
# ============================================================================

def test_reverse_geocode_success(osm_service):
    """Test successful reverse geocoding."""
    reverse_response = {
        'display_name': 'Bambili, Bamenda, Northwest Region, Cameroon',
        'type': 'village',
        'place_id': '12345'
    }
    
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = reverse_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch('time.sleep'):
            result = osm_service.reverse_geocode(5.9631, 10.1591)
        
        assert result is not None
        assert result['address'] == 'Bambili, Bamenda, Northwest Region, Cameroon'
        assert result['latitude'] == 5.9631
        assert result['longitude'] == 10.1591


def test_reverse_geocode_invalid_coordinates(osm_service):
    """Test reverse geocoding with invalid coordinates."""
    with pytest.raises(ValueError, match="Invalid coordinates"):
        osm_service.reverse_geocode(95, 200)


def test_reverse_geocode_not_found(osm_service):
    """Test reverse geocoding when no address found."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {'error': 'Unable to geocode'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch('time.sleep'):
            result = osm_service.reverse_geocode(5.9631, 10.1591)
        
        assert result is None


# ============================================================================
# PROFILE TESTS
# ============================================================================

def test_all_profiles(osm_service, sample_route_response):
    """Test all travel profiles."""
    profiles = ['foot', 'car', 'bike']
    
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = sample_route_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        for profile in profiles:
            route = osm_service.get_route(
                start=(5.9631, 10.1591),
                end=(5.9641, 10.1601),
                profile=profile
            )
            
            assert route is not None
            assert route['profile'] == profile


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])