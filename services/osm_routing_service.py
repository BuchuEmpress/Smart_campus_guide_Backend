"""
OpenStreetMap Routing Service

FREE routing and geocoding using OpenStreetMap APIs.
NO API KEY REQUIRED! Completely FREE!

Uses:
- OSRM (Open Source Routing Machine) for directions
- Nominatim for geocoding and place search

Advantages:
- Free forever, no card needed
- Good coverage for Cameroon/Africa
- Walking, driving, and cycling support
- No rate limits on OSRM
- Privacy-friendly (no tracking)
"""

import requests
import logging
from typing import Optional, Dict, List, Tuple
from time import sleep

from services.locations_utils import validate_coordinates, format_distance, format_time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FREE Public APIs - NO API KEY NEEDED!
OSRM_API = "https://router.project-osrm.org"
NOMINATIM_API = "https://nominatim.openstreetmap.org"

# Request timeout (seconds)
TIMEOUT = 10

# Nominatim requires 1 request per second
NOMINATIM_DELAY = 1


class OSMRoutingService:
    """
    OpenStreetMap routing and geocoding service.
    
    Completely FREE - no API key required!
    
    Features:
    - Get turn-by-turn directions (walking, driving, cycling)
    - Search for places by name
    - Geocode addresses to coordinates
    - Reverse geocode coordinates to addresses
    
    Example:
        >>> osm = OSMRoutingService()
        >>> route = osm.get_route((5.9631, 10.1591), (5.9638, 10.1515))
        >>> print(route['distance_text'])
        '1.2 kilometers'
    """
    
    def __init__(self):
        """Initialize OSM service - no API key needed!"""
        logger.info("🗺️  OSM Routing Service initialized (FREE!)")
    
    def get_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile: str = "foot"
    ) -> Optional[Dict]:
        """
        Get route with turn-by-turn directions.
        
        Args:
            start: (latitude, longitude) starting point
            end: (latitude, longitude) destination
            profile: Travel mode - 'foot' (walking), 'car' (driving), 'bike' (cycling)
        
        Returns:
            Dictionary containing:
                - distance: Distance in meters
                - distance_text: Formatted distance (e.g., "1.2 kilometers")
                - duration: Duration in seconds
                - duration_text: Formatted duration (e.g., "15 minutes")
                - steps: List of turn-by-turn instructions
                - geometry: Route coordinates for map display
            Returns None if route not found or error occurs.
        
        Raises:
            ValueError: If coordinates are invalid or profile is unknown
        
        Example:
            >>> route = osm.get_route(
            ...     start=(5.9631, 10.1591),
            ...     end=(5.9638, 10.1515),
            ...     profile='foot'
            ... )
            >>> print(f"{route['distance_text']} in {route['duration_text']}")
            '1.2 kilometers in 15 minutes'
        """
        try:
            # Validate inputs
            if not validate_coordinates(start[0], start[1]):
                raise ValueError(f"Invalid start coordinates: {start}")
            
            if not validate_coordinates(end[0], end[1]):
                raise ValueError(f"Invalid end coordinates: {end}")
            
            valid_profiles = ['foot', 'car', 'bike']
            if profile not in valid_profiles:
                raise ValueError(f"Invalid profile: {profile}. Must be one of {valid_profiles}")
            
            # IMPORTANT: OSRM uses lon,lat format (NOT lat,lon!)
            url = f"{OSRM_API}/route/v1/{profile}/{start[1]},{start[0]};{end[1]},{end[0]}"
            
            params = {
                'overview': 'full',      # Get complete route
                'steps': 'true',         # Get turn-by-turn steps
                'geometries': 'geojson'  # Route format
            }
            
            logger.info(f"🚶 Getting {profile} route from {start} to {end}")
            
            # Make request
            response = requests.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if route found
            if data.get('code') != 'Ok':
                logger.warning(f"OSRM error: {data.get('message', 'Unknown error')}")
                return None
            
            if not data.get('routes'):
                logger.warning("No routes found")
                return None
            
            # Parse route data
            route = data['routes'][0]
            leg = route['legs'][0]
            
            # Extract turn-by-turn steps
            steps = []
            for step in leg.get('steps', []):
                maneuver = step.get('maneuver', {})
                
                step_info = {
                    'instruction': maneuver.get('instruction', 'Continue'),
                    'type': maneuver.get('type', 'turn'),
                    'modifier': maneuver.get('modifier', ''),
                    'distance': step.get('distance', 0),
                    'distance_text': format_distance(step.get('distance', 0)),
                    'duration': step.get('duration', 0),
                    'duration_text': format_time(int(step.get('duration', 0))),
                    'name': step.get('name', ''),
                    'ref': step.get('ref', '')  # Road reference/number
                }
                steps.append(step_info)
            
            # Build result
            result = {
                'distance': route['distance'],
                'distance_text': format_distance(route['distance']),
                'duration': route['duration'],
                'duration_text': format_time(int(route['duration'])),
                'steps': steps,
                'geometry': route.get('geometry', {}),  # For map display
                'profile': profile
            }
            
            logger.info(f"✅ Route found: {result['distance_text']}, {result['duration_text']}")
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"❌ Request timeout after {TIMEOUT}s")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error: {str(e)}")
            return None
        except ValueError as e:
            logger.error(f"❌ Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error getting route: {str(e)}")
            return None
    
    def search_place(
        self,
        query: str,
        near: Optional[Tuple[float, float]] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for places by name.
        
        Args:
            query: Place name to search (e.g., "Bamenda Hospital")
            near: Optional (lat, lon) to prioritize nearby results
            limit: Maximum number of results (1-10)
        
        Returns:
            List of place dictionaries containing:
                - name: Place name
                - display_name: Full address
                - latitude: Latitude coordinate
                - longitude: Longitude coordinate
                - type: Place type (e.g., 'hospital', 'hotel')
                - importance: Relevance score (0-1)
        
        Example:
            >>> places = osm.search_place("hospital", near=(5.9631, 10.1591))
            >>> print(places[0]['name'])
            'Bamenda Regional Hospital'
        """
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            
            url = f"{NOMINATIM_API}/search"
            
            # Bamenda Bounding Box (Approximate)
            # North: 6.05, South: 5.90, West: 10.10, East: 10.30
            params = {
                'q': query.strip(),
                'format': 'json',
                'limit': min(max(limit, 1), 10),
                'addressdetails': 1,
                'viewbox': '10.10,6.05,10.30,5.90',  # West,North,East,South
                'bounded': 1  # Strict bounding
            }
            
            # Add location bias if provided (still useful within the box)
            if near:
                if validate_coordinates(near[0], near[1]):
                    params['lat'] = near[0]
                    params['lon'] = near[1]
            
            # Nominatim requires User-Agent header
            headers = {
                'User-Agent': 'SmartCampusGuide/1.0 (Educational Project)'
            }
            
            logger.info(f"🔍 Searching OSM for: '{query}'")
            
            # Respect Nominatim rate limit (1 request/second)
            sleep(NOMINATIM_DELAY)
            
            # Make request
            response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            
            results = response.json()
            
            if not results:
                logger.warning(f"No results found for: '{query}'")
                return []
            
            # Format results
            places = []
            for result in results:
                place = {
                    'name': result.get('name', result.get('display_name', 'Unknown')),
                    'display_name': result.get('display_name', ''),
                    'latitude': float(result['lat']),
                    'longitude': float(result['lon']),
                    'type': result.get('type', 'place'),
                    'class': result.get('class', ''),
                    'importance': result.get('importance', 0),
                    'place_id': result.get('place_id', ''),
                    'osm_type': result.get('osm_type', ''),
                    'osm_id': result.get('osm_id', '')
                }
                places.append(place)
            
            logger.info(f"✅ Found {len(places)} places")
            return places
            
        except requests.exceptions.Timeout:
            logger.error(f"❌ Request timeout after {TIMEOUT}s")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error: {str(e)}")
            return []
        except ValueError as e:
            logger.error(f"❌ Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error searching places: {str(e)}")
            return []
    
    def geocode(self, address: str) -> Optional[Dict]:
        """
        Convert address to coordinates.
        
        Args:
            address: Address string (e.g., "University of Bamenda, Cameroon")
        
        Returns:
            Dictionary with latitude, longitude, and formatted address.
            Returns None if address not found.
        
        Example:
            >>> result = osm.geocode("Bamenda, Cameroon")
            >>> print(f"({result['latitude']}, {result['longitude']})")
            '(5.9631, 10.1591)'
        """
        try:
            places = self.search_place(address, limit=1)
            
            if not places:
                return None
            
            place = places[0]
            
            return {
                'address': place['display_name'],
                'latitude': place['latitude'],
                'longitude': place['longitude'],
                'type': place['type']
            }
            
        except Exception as e:
            logger.error(f"❌ Error geocoding address: {str(e)}")
            return None
    
    def reverse_geocode(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Convert coordinates to address.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dictionary with formatted address and place details.
            Returns None if no address found.
        
        Example:
            >>> result = osm.reverse_geocode(5.9631, 10.1591)
            >>> print(result['address'])
            'Bambili, Bamenda, Northwest Region, Cameroon'
        """
        try:
            if not validate_coordinates(lat, lon):
                raise ValueError(f"Invalid coordinates: ({lat}, {lon})")
            
            url = f"{NOMINATIM_API}/reverse"
            
            params = {
                'lat': lat,
                'lon': lon,
                'format': 'json',
                'addressdetails': 1
            }
            
            headers = {
                'User-Agent': 'SmartCampusGuide/1.0 (Educational Project)'
            }
            
            logger.info(f"🔍 Reverse geocoding ({lat}, {lon})")
            
            # Respect rate limit
            sleep(NOMINATIM_DELAY)
            
            response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            if 'error' in data:
                logger.warning(f"No address found for ({lat}, {lon})")
                return None
            
            result = {
                'address': data.get('display_name', 'Unknown location'),
                'latitude': lat,
                'longitude': lon,
                'type': data.get('type', 'location'),
                'place_id': data.get('place_id', '')
            }
            
            logger.info(f"✅ Found address: {result['address']}")
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"❌ Request timeout after {TIMEOUT}s")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error: {str(e)}")
            return None
        except ValueError as e:
            logger.error(f"❌ Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            return None


# Usage examples
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🗺️  OSM ROUTING SERVICE TEST")
    print("="*80 + "\n")
    
    # Initialize service
    osm = OSMRoutingService()
    
    # Test 1: Get route
    print("📍 TEST 1: Getting walking route")
    print("-" * 80)
    start = (6.010317, 10.258816)  # Main gate
    end = (6.010380, 10.259017)    # Asanji Hall
    
    route = osm.get_route(start, end, profile='foot')
    
    if route:
        print(f"✅ Route found!")
        print(f"   Distance: {route['distance_text']}")
        print(f"   Duration: {route['duration_text']}")
        print(f"   Steps: {len(route['steps'])}")
        
        print(f"\n   First 3 steps:")
        for i, step in enumerate(route['steps'][:3], 1):
            print(f"   {i}. {step['instruction']} - {step['distance_text']}")
    else:
        print("❌ Route not found")
    
    # Test 2: Search place
    print("\n📍 TEST 2: Searching for hospital")
    print("-" * 80)
    places = osm.search_place("Bamenda Hospital", near=start, limit=3)
    
    if places:
        print(f"✅ Found {len(places)} places:")
        for i, place in enumerate(places, 1):
            print(f"   {i}. {place['name']}")
            print(f"      {place['display_name']}")
            print(f"      ({place['latitude']:.6f}, {place['longitude']:.6f})")
    else:
        print("❌ No places found")
    
    # Test 3: Geocode
    print("\n📍 TEST 3: Geocoding address")
    print("-" * 80)
    result = osm.geocode("Bamenda, Cameroon")
    
    if result:
        print(f"✅ Address found:")
        print(f"   {result['address']}")
        print(f"   ({result['latitude']:.6f}, {result['longitude']:.6f})")
    else:
        print("❌ Address not found")
    
    # Test 4: Reverse geocode
    print("\n📍 TEST 4: Reverse geocoding")
    print("-" * 80)
    result = osm.reverse_geocode(5.9631, 10.1591)
    
    if result:
        print(f"✅ Location identified:")
        print(f"   {result['address']}")
    else:
        print("❌ Location not found")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE!")
    print("="*80 + "\n")