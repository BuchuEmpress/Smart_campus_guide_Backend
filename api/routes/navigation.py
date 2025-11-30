"""
Navigation Routes Module

Handles all navigation-related endpoints using OSM routing.

Features:
- Search locations (Qdrant + OSM)
- Get directions (OSM routing + Gemini humanization)
- Location details
- Chat interface
"""

import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Path
from fastapi.responses import JSONResponse

from api.models.navigation_models import (
    NavigationRequest,
    NavigationResponse,
    SearchRequest,
    SearchResponse,
    LocationDetailResponse
)
from services.qdrant_service import QdrantService
from services.osm_routing_service import OSMRoutingService  # Changed from MapsService
from services.gemini_service import GeminiService
from services.analytics_service import AnalyticsService
from services.locations_utils import calculate_distance, format_distance

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api", tags=["Navigation"])

# Service instances (initialized on first use)
qdrant_service = None
osm_service = None  # Changed from maps_service
gemini_service = None
analytics_service = None


def get_services():
    """Initialize services if not already initialized."""
    global qdrant_service, osm_service, gemini_service, analytics_service
    
    if not qdrant_service:
        qdrant_service = QdrantService(load_model=True)
    if not osm_service:
        osm_service = OSMRoutingService()  # Changed from MapsService
    if not gemini_service:
        gemini_service = GeminiService()
    if not analytics_service:
        analytics_service = AnalyticsService()
    
    return {
        'qdrant': qdrant_service,
        'osm': osm_service,  # Changed from 'maps'
        'gemini': gemini_service,
        'analytics': analytics_service
    }


@router.post("/navigate", response_model=NavigationResponse)
async def navigate(request: NavigationRequest):
    """
    Main navigation endpoint - Get directions to a location.
    
    Flow:
    1. Get user's current location (GPS or manual)
    2. Search location in Qdrant database first
    3. If not found, search using OSM
    4. Calculate route using OSM
    5. Humanize directions using Gemini AI
    6. Return natural, friendly directions
    """
    try:
        services = get_services()
        logger.info(f"🧭 Navigation request: '{request.query}' from {request.user_location.source}")
        
        # STEP 1: Get user's current location
        user_coords = None
        
        if request.user_location.source == "gps":
            # GPS location provided
            if not request.user_location.lat or not request.user_location.lon:
                raise HTTPException(
                    status_code=400,
                    detail="GPS location requires latitude and longitude"
                )
            user_coords = (request.user_location.lat, request.user_location.lon)
            logger.info(f"📍 User location (GPS): {user_coords}")
        
        elif request.user_location.source == "manual":
            # Manual address provided - geocode it using OSM
            if not request.user_location.address:
                raise HTTPException(
                    status_code=400,
                    detail="Manual location requires an address"
                )
            
            logger.info(f"📍 Geocoding manual address: {request.user_location.address}")
            geocoded = services['osm'].geocode(request.user_location.address)
            
            if not geocoded:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not find location: {request.user_location.address}"
                )
            
            user_coords = (geocoded['latitude'], geocoded['longitude'])
            logger.info(f"📍 User location (geocoded): {user_coords}")
        
        # STEP 2: Search for destination in Qdrant FIRST (your data is best!)
        destination = None
        is_on_campus = False
        dest_coords = None
        
        logger.info(f"🔍 Searching Qdrant for: '{request.query}'")
        
        try:
            campus_results = services['qdrant'].search(request.query, limit=1)
            
            # Check if we have a good match (score > 0.7)
            if campus_results and campus_results[0].get('score', 0) > 0.7:
                destination = campus_results[0]
                is_on_campus = True
                dest_coords = (destination['latitude'], destination['longitude'])
                
                logger.info(f"✅ Found in Qdrant: {destination.get('name', request.query)}")
                logger.info(f"   Score: {destination['score']:.4f}")
        
        except Exception as e:
            logger.warning(f"⚠️  Qdrant search failed: {str(e)}")
        
        # STEP 3: If not found in Qdrant, search using OSM
        if not destination:
            logger.info(f"🔍 Searching OSM for: '{request.query}'")
            
            places = services['osm'].search_place(
                request.query,
                near=user_coords,
                limit=5
            )
            
            if not places:
                raise HTTPException(
                    status_code=404,
                    detail=f"Location not found: {request.query}"
                )
            
            # Use first result
            destination = places[0]
            is_on_campus = False
            dest_coords = (destination['latitude'], destination['longitude'])
            
            logger.info(f"✅ Found in OSM: {destination.get('name', request.query)}")
        
        # STEP 4: Calculate route using OSM
        logger.info(f"🗺️  Calculating {request.travel_mode} route")
        
        # Map travel_mode to OSM profile
        profile_map = {
            'walking': 'foot',
            'driving': 'car',
            'cycling': 'bike',
            'transit': 'foot'  # OSM doesn't have transit, use walking
        }
        osm_profile = profile_map.get(request.travel_mode, 'foot')
        
        route = services['osm'].get_route(
            start=user_coords,
            end=dest_coords,
            profile=osm_profile
        )
        
        if not route:
            raise HTTPException(
                status_code=500,
                detail="Could not calculate route"
            )
        
        logger.info(f"✅ Route calculated: {route['distance_text']}, {route['duration_text']}")
        
        # STEP 5: Get nearby landmarks from Qdrant (for context)
        nearby_landmarks = []
        
        if is_on_campus:
            try:
                # Get all campus locations
                all_locations = services['qdrant'].search("", limit=50)
                
                # Filter for nearby locations (within 500m)
                for loc in all_locations:
                    if 'latitude' in loc and 'longitude' in loc:
                        loc_coords = (loc['latitude'], loc['longitude'])
                        dist = calculate_distance(
                            user_coords[0], user_coords[1],
                            loc_coords[0], loc_coords[1]
                        )
                        if dist < 500:  # Within 500 meters
                            nearby_landmarks.append(loc.get('name', 'Unknown location'))
                
                logger.info(f"📍 Found {len(nearby_landmarks)} nearby landmarks")
            
            except Exception as e:
                logger.warning(f"⚠️  Could not get nearby landmarks: {str(e)}")
        
        # STEP 6: Humanize directions with Gemini AI
        logger.info(f"🤖 Humanizing directions with Gemini")
        
        try:
            destination_name = destination.get('name') or destination.get('display_name', request.query)
            destination_description = destination.get('description', '')
            
            humanized_directions = services['gemini'].humanize_directions(
                route_data=route,
                campus_context={'is_on_campus': is_on_campus},
                nearby_landmarks=nearby_landmarks[:10],  # Top 10 landmarks
                destination_name=destination_name,
                destination_description=destination_description
            )
            
            logger.info(f"✅ Directions humanized")
        
        except Exception as e:
            logger.error(f"❌ Gemini humanization failed: {str(e)}")
            # Fallback to basic directions
            humanized_directions = f"Walk to {destination_name}. Distance: {route['distance_text']}, estimated time: {route['duration_text']}."
        
        # STEP 7: Log search for analytics
        try:
            services['analytics'].log_search(
                query=request.query,
                location_found=destination.get('name') or destination.get('display_name', request.query),
                user_location=user_coords,
                is_on_campus=is_on_campus
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to log analytics: {str(e)}")
        
        # STEP 8: Build and return response
        response = NavigationResponse(
            destination=destination.get('name') or destination.get('display_name', request.query),
            destination_address=destination.get('display_name', destination.get('description', '')),
            destination_coords={
                'lat': dest_coords[0],
                'lng': dest_coords[1]
            },
            is_on_campus=is_on_campus,
            humanized_directions=humanized_directions,
            distance=route['distance_text'],
            estimated_time=route['duration_text'],
            travel_mode=request.travel_mode,
            raw_steps=route['steps'],
            polyline=str(route.get('geometry', ''))
        )
        
        logger.info(f"✅ Navigation complete: {response.destination}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Navigation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse)
async def search_locations(request: SearchRequest):
    """
    Search for locations with filtering options.
    
    Searches both:
    1. Your Qdrant database (campus + city data)
    2. OSM (for places not in your database)
    """
    try:
        services = get_services()
        logger.info(f"🔍 Search request: '{request.query}' with filter: {request.filter}")
        
        results = []
        
        # Get user coordinates if provided
        user_coords = None
        if request.user_location:
            if request.user_location.source == "gps":
                user_coords = (request.user_location.lat, request.user_location.lon)
            elif request.user_location.source == "manual" and request.user_location.address:
                geocoded = services['osm'].geocode(request.user_location.address)
                if geocoded:
                    user_coords = (geocoded['latitude'], geocoded['longitude'])
        
        # Search Qdrant (on-campus and city data)
        if request.filter in ["all", "on-campus"]:
            try:
                logger.info(f"🔍 Searching Qdrant")
                campus_results = services['qdrant'].search(
                    request.query,
                    limit=request.limit
                )
                
                for result in campus_results:
                    location_coords = (result['latitude'], result['longitude'])
                    
                    # Calculate distance if user location provided
                    distance_text = None
                    if user_coords:
                        dist = calculate_distance(
                            user_coords[0], user_coords[1],
                            location_coords[0], location_coords[1]
                        )
                        distance_text = format_distance(dist)
                    
                    results.append({
                        'id': result.get('id', ''),
                        'name': result.get('name', ''),
                        'description': result.get('description', ''),
                        'coordinates': {
                            'lat': result['latitude'],
                            'lng': result['longitude']
                        },
                        'type': result.get('type', 'location'),
                        'is_on_campus': True,
                        'distance_from_user': distance_text,
                        'relevance_score': result.get('score', 0)
                    })
                
                logger.info(f"✅ Found {len(campus_results)} results in Qdrant")
            
            except Exception as e:
                logger.warning(f"⚠️  Qdrant search failed: {str(e)}")
        
        # Search OSM (off-campus)
        if request.filter in ["all", "off-campus"]:
            try:
                logger.info(f"🔍 Searching OSM")
                places = services['osm'].search_place(
                    request.query,
                    near=user_coords,
                    limit=request.limit
                )
                
                for place in places:
                    # Calculate distance if user location provided
                    distance_text = None
                    if user_coords:
                        dist = calculate_distance(
                            user_coords[0], user_coords[1],
                            place['latitude'], place['longitude']
                        )
                        distance_text = format_distance(dist)
                    
                    results.append({
                        'id': place.get('place_id', ''),
                        'name': place.get('name', ''),
                        'description': place.get('display_name', ''),
                        'coordinates': {
                            'lat': place['latitude'],
                            'lng': place['longitude']
                        },
                        'type': place.get('type', 'place'),
                        'is_on_campus': False,
                        'distance_from_user': distance_text,
                        'relevance_score': place.get('importance', 0)
                    })
                
                logger.info(f"✅ Found {len(places)} results in OSM")
            
            except Exception as e:
                logger.warning(f"⚠️  OSM search failed: {str(e)}")
        
        # Sort by distance if user location provided
        if user_coords and results:
            results.sort(key=lambda x: (
                float('inf') if x['distance_from_user'] is None 
                else float(x['distance_from_user'].split()[0])
            ))
        
        logger.info(f"✅ Total search results: {len(results)}")
        
        return SearchResponse(
            query=request.query,
            total_results=len(results),
            locations=results[:request.limit]
        )
        
    except Exception as e:
        logger.error(f"❌ Search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/location/{location_id}", response_model=LocationDetailResponse)
async def get_location_details(
    location_id: str = Path(..., description="Location ID")
):
    """
    Get detailed information about a specific location.
    """
    try:
        services = get_services()
        logger.info(f"📍 Getting details for: {location_id}")
        
        # Try Qdrant first
        try:
            # Search by ID in Qdrant
            results = services['qdrant'].search(location_id, limit=1)
            
            if results and results[0].get('id') == location_id:
                location = results[0]
                
                return LocationDetailResponse(
                    id=location_id,
                    name=location.get('name', ''),
                    description=location.get('description', ''),
                    coordinates={
                        'lat': location['latitude'],
                        'lng': location['longitude']
                    },
                    type=location.get('type', 'location'),
                    is_on_campus=True
                )
        
        except Exception as e:
            logger.warning(f"⚠️  Qdrant lookup failed: {str(e)}")
        
        # If not found, location doesn't exist
        raise HTTPException(
            status_code=404,
            detail=f"Location not found: {location_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting location details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve location details: {str(e)}"
        )


@router.post("/chat")
async def chat(
    message: str = Query(..., description="User message"),
    context: Optional[str] = Query(None, description="Conversation context")
):
    """
    Conversational interface for navigation queries.
    """
    try:
        services = get_services()
        logger.info(f"💬 Chat message: '{message}'")
        
        # Extract intent using Gemini
        intent = services['gemini'].extract_intent(message)
        
        # Generate response
        response_text = services['gemini'].generate_response(
            prompt=message,
            context=context
        )
        
        return JSONResponse(content={
            'response': response_text,
            'intent': intent,
            'timestamp': str(datetime.now())
        })
        
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )