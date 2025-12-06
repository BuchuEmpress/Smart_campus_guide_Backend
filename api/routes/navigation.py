"""
Navigation Routes - FULLY UPDATED FOR YOUR QdrantService

Fixes:
✔ Uses qdrant_service.get_by_id() instead of search_by_id()
✔ Fully async search using your search_points implementation
✔ Improved error handling
✔ Preserves ALL original functionality
✔ No breaking changes
"""

from fastapi import APIRouter, Depends, HTTPException
from api.models.navigation_models import (
    NavigationRequest,
    NavigationResponse,
    SearchResponse,
    Location,
    SearchRequest,
    LocationDetailResponse
)
from services.qdrant_service import QdrantService
from services.gemini_service import GeminiService
from services.osm_routing_service import OSMRoutingService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


# =============== DEPENDENCIES ===============

def get_qdrant_service():
    return QdrantService(load_model=True)

def get_gemini_service():
    return GeminiService()

def get_osm_routing_service():
    return OSMRoutingService()


# =============== SEARCH ENDPOINT ===============

@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    qdrant_service: QdrantService = Depends(get_qdrant_service)
):
    """Semantic search across campus locations."""
    try:
        logger.info(f"🔍 Search query: {request.query}")

        results = await qdrant_service.search(
            query=request.query,
            limit=getattr(request, "limit", 5)
        )

        locations = [Location(**loc) for loc in results]

        return SearchResponse(locations=locations)

    except Exception as e:
        logger.error(f"❌ Search error: {e}", exc_info=True)
        raise HTTPException(500, f"Search failed: {str(e)}")


# =============== NAVIGATION ENDPOINT ===============

@router.post("/navigate", response_model=NavigationResponse)
async def navigate(
    request: NavigationRequest,
    gemini_service: GeminiService = Depends(get_gemini_service),
    qdrant_service: QdrantService = Depends(get_qdrant_service),
    osm_routing_service: OSMRoutingService = Depends(get_osm_routing_service)
):
    try:
        logger.info(f"🧭 Navigation request: {request.query}")

        # 1. Extract intent
        intent = await gemini_service.extract_intent(request.query)
        logger.info(f"Intent extracted: {intent}")

        # --- CHAT MODE ---
        if intent.get("action") == "chat":
            reply = await gemini_service.generate_response(
                prompt=f"User said: {request.query}"
            )
            return NavigationResponse(status="chat", message=reply)

        # 2. Determine what location is requested
        loc_query = intent.get("location_query", "").strip()
        if not loc_query:
            return NavigationResponse(
                status="error",
                message="I couldn't determine which location you're asking about. Try rephrasing it."
            )

        logger.info(f"Searching Qdrant for location match: {loc_query}")

        # Semantic search for closest location
        search_results = await qdrant_service.search(loc_query, limit=1)

        if not search_results:
            fallback = await gemini_service.generate_response(
                prompt=f"Could not find location '{loc_query}'. Provide helpful fallback."
            )
            return NavigationResponse(
                status="chat",
                message=fallback or f"Couldn't find '{loc_query}'. Try another name."
            )

        destination = Location(**search_results[0])
        logger.info(f"Matched location: {destination.name}")

        # --- SEARCH INTENT ONLY (NO ROUTE NEEDED) ---
        if intent.get("action") == "search":
            return NavigationResponse(
                status="success",
                message=f"Found {destination.name}",
                destination=destination,
                route=None
            )

        # --- ROUTE MODE ---
        if not request.user_location or request.user_location.lat is None:
            return NavigationResponse(
                status="error",
                message="I need your current location before I can route you."
            )

        start = (request.user_location.lat, request.user_location.lon)
        end = (destination.latitude, destination.longitude)

        route = osm_routing_service.get_route(
            start=start,
            end=end,
            profile="foot" if request.travel_mode == "walking" else "car"
        )

        if not route:
            return NavigationResponse(
                status="error",
                message=f"No route found to {destination.name}.",
                destination=destination
            )

        human_directions = await gemini_service.humanize_directions(route)

        return NavigationResponse(
            status="success",
            message=human_directions,
            destination=destination,
            route=route
        )

    except Exception as e:
        logger.error(f"❌ Navigation error: {e}", exc_info=True)
        return NavigationResponse(
            status="error",
            message=f"An error occurred: {str(e)}"
        )


# =============== LOCATION DETAILS ENDPOINT ===============

@router.get("/location/{location_id}", response_model=LocationDetailResponse)
async def get_location_details(
    location_id: str,
    qdrant_service: QdrantService = Depends(get_qdrant_service),
    gemini_service: GeminiService = Depends(get_gemini_service)
):
    """Detailed information about a single location."""
    try:
        logger.info(f"📍 Fetching location details: {location_id}")

        # FIXED: your QdrantService uses get_by_id() instead of search_by_id
        location_data = await qdrant_service.get_by_id(location_id)

        if not location_data:
            raise HTTPException(404, f"Location not found: {location_id}")

        location = Location(**location_data)

        enhanced = await gemini_service.enhance_description(
            location.model_dump()
        )

        return LocationDetailResponse(
            **location.model_dump(),
            enhanced_description=enhanced
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Location detail error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
