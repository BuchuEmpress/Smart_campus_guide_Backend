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
    LocationDetailResponse,
    ChatRequest,
    ChatResponse
)
from services.qdrant_service import QdrantService
from services.gemini_service import GeminiService
from services.osm_routing_service import OSMRoutingService
from services.mongodb_service import MongoDBService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


def _sanitize_location(loc_data: dict, fallback_id: str = "Unknown") -> dict:
    """Ensures location has a displayable name and required fields."""
    if not loc_data.get("name"):
        desc = loc_data.get("description", "")
        if desc:
            # First sentence of description
            loc_data["name"] = desc.split('.')[0][:50].strip() or desc[:50].strip()
        else:
            loc_data["name"] = f"Location {loc_data.get('id', fallback_id)}"
    
    # Ensure type exists for Pydantic
    if not loc_data.get("type"):
        loc_data["type"] = "landmark"
        
    return loc_data

# =============== DEPENDENCIES ===============

def get_qdrant_service():
    return QdrantService()

def get_gemini_service():
    return GeminiService()

def get_osm_routing_service():
    return OSMRoutingService()

def get_mongodb_service():
    service = MongoDBService()
    return service


# =============== SEARCH ENDPOINT ===============

@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    qdrant_service: QdrantService = Depends(get_qdrant_service),
    gemini_service: GeminiService = Depends(get_gemini_service),
    osm_routing_service: OSMRoutingService = Depends(get_osm_routing_service)
):
    """Semantic search across campus locations (Smart & Bamenda-Aware)."""
    try:
        logger.info(f"🔍 Smart Search query: {request.query}")

        intent = await gemini_service.extract_intent(request.query)
        clean_query = intent.get("location_query")
        
        if not clean_query or clean_query.lower() == request.query.lower():
            raw = request.query.lower()
            for prefix in ["please", "tell me", "show me", "can you", "where is", "where's"]:
                raw = raw.replace(prefix, "")
            clean_query = raw.strip()

        if not clean_query: clean_query = request.query

        valid_locations = []
        seen_ids = set()

        # 1. Aggressive Text Search (ID, Name, Description)
        logger.info(f"🔎 Running global keyword search for '{clean_query}'...")
        text_results = await qdrant_service.search_by_text(clean_query, limit=10)
        
        for loc in text_results:
            try:
                loc = _sanitize_location(loc)
                loc_id = loc.get("id")
                if loc_id not in seen_ids:
                    loc["score"] = 1.0 # Priority for direct matches
                    valid_locations.append(Location(**loc))
                    seen_ids.add(loc_id)
            except Exception as e:
                logger.warning(f"Skipping invalid text result: {e}")

        # 2. Vector Search (Semantic)
        logger.info(f"🧠 Running vector search for '{clean_query}'...")
        vector_results = await qdrant_service.search(clean_query, limit=5)

        for loc in vector_results:
            try:
                loc = _sanitize_location(loc)
                loc_id = loc.get("id")
                if loc_id not in seen_ids and loc.get("score", 0) >= 0.40:
                    valid_locations.append(Location(**loc))
                    seen_ids.add(loc_id)
            except Exception as e:
                logger.warning(f"Skipping invalid vector result: {e}")

        # 3. OSM Fallback
        if not valid_locations:
            osm_results = osm_routing_service.search_place(clean_query, limit=3)
            for osm_loc in osm_results:
                valid_locations.append(Location(
                    id=f"osm_{osm_loc.get('place_id')}",
                    name=osm_loc['name'],
                    description=osm_loc.get('display_name', 'External Location'),
                    type="external",
                    latitude=osm_loc['latitude'],
                    longitude=osm_loc['longitude']
                ))

        return SearchResponse(locations=valid_locations)

    except Exception as e:
        logger.error(f"❌ Search error: {e}", exc_info=True)
        return SearchResponse(locations=[])


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

        intent = await gemini_service.extract_intent(request.query)
        if intent.get("action") == "chat":
            reply = await gemini_service.generate_response(f"User said: {request.query}")
            return NavigationResponse(status="chat", message=reply)

        loc_query = intent.get("location_query", "").strip() or request.query
        
        # Priority 1: Direct ID or Text match
        results = await qdrant_service.search_by_text(loc_query, limit=1)
        
        # Priority 2: Vector Search
        if not results:
            results = await qdrant_service.search(loc_query, limit=1)
            if results and results[0].get("score", 0) < 0.40:
                results = []

        # Priority 3: OSM
        if not results:
            osm_res = osm_routing_service.search_place(loc_query, limit=1)
            if osm_res:
                osm_loc = osm_res[0]
                results = [{
                    "id": f"osm_{osm_loc.get('place_id')}",
                    "name": osm_loc['name'],
                    "description": osm_loc.get('display_name', 'External'),
                    "type": "external",
                    "latitude": osm_loc['latitude'],
                    "longitude": osm_loc['longitude'],
                    "score": 1.0
                }]

        if not results:
            return NavigationResponse(status="chat", message=f"I couldn't find '{loc_query}' on campus or in Bamenda.")

        loc_data = _sanitize_location(results[0])
        destination = Location(**loc_data)

        if intent.get("action") == "search":
            return NavigationResponse(status="success", message=f"Found {destination.name}", destination=destination)

        if not request.user_location or request.user_location.lat is None:
            return NavigationResponse(status="error", message="I need your location to route you.")

        route = osm_routing_service.get_route(
            start=(request.user_location.lat, request.user_location.lon),
            end=(destination.latitude, destination.longitude),
            profile="foot" if request.travel_mode == "walking" else "car"
        )

        if not route:
            return NavigationResponse(status="error", message=f"No route found to {destination.name}.", destination=destination)

        human_directions = await gemini_service.humanize_directions(route)
        return NavigationResponse(status="success", message=human_directions, destination=destination, route=route)

    except Exception as e:
        logger.error(f"❌ Navigation error: {e}", exc_info=True)
        return NavigationResponse(status="error", message=f"Error: {str(e)}")


# =============== LOCATION DETAILS ENDPOINT ===============

@router.get("/location/{location_id}", response_model=LocationDetailResponse)
async def get_location_details(
    location_id: str,
    qdrant_service: QdrantService = Depends(get_qdrant_service),
    gemini_service: GeminiService = Depends(get_gemini_service)
):
    """Detailed location view with humanized description."""
    try:
        # Try finding by ID
        loc = await qdrant_service.get_by_id(location_id)
        
        # Fallback to text search if not found
        if not loc:
            matches = await qdrant_service.search_by_text(location_id, limit=1)
            if matches:
                loc = matches[0]
        
        if not loc:
            raise HTTPException(status_code=404, detail="Location not found")
            
        loc = _sanitize_location(loc)
        enhanced = await gemini_service.enhance_description(loc)
        
        return LocationDetailResponse(
            **loc,
            enhanced_description=enhanced
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching location details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============== CHATBOT ENDPOINT ===============

@router.post("/chat", response_model=ChatResponse)
async def navigation_chat(
    request: ChatRequest,
    qdrant_service: QdrantService = Depends(get_qdrant_service),
    gemini_service: GeminiService = Depends(get_gemini_service),
    osm_routing_service: OSMRoutingService = Depends(get_osm_routing_service),
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """Conversational AI Agent for Navigation - Integrated with Search & Routing."""
    try:
        await mongodb_service.connect()
        history = await mongodb_service.get_chat_history(request.session_id, "location")
        
        # 1. Extract Intent and Data
        intent_response = await gemini_service.extract_intent(request.message)
        action = intent_response.get("action", "chat")
        location_query = intent_response.get("location_query")
        
        grounding_data = ""
        metadata = {"action_taken": action}

        # 2. Perform Action if needed
        if action == "search" and location_query:
            # Run our aggressive universal search
            results = await qdrant_service.search_by_text(location_query, limit=3)
            if results:
                found_names = [r.get("name") or r.get("id") for r in results]
                grounding_data = f"\n[INTERNAL DATA: I found these locations: {', '.join(found_names)}. One description: {results[0].get('description')}]"
                metadata["results_found"] = len(results)
            else:
                grounding_data = "\n[INTERNAL DATA: No locations found in our database for this query.]"

        elif action == "navigate" and location_query and request.user_location:
            # Find destination first
            dest_results = await qdrant_service.search_by_text(location_query, limit=1)
            if dest_results:
                dest = dest_results[0]
                # Get route
                start_coords = (request.user_location.lat, request.user_location.lon)
                end_coords = (dest.get("latitude"), dest.get("longitude"))
                
                route = osm_routing_service.get_route(start_coords, end_coords)
                if route:
                    human_dirs = await gemini_service.humanize_directions(route)
                    grounding_data = f"\n[INTERNAL DATA: Found route to {dest.get('name')}. Real Humanized Directions: {human_dirs}]"
                    metadata["destination"] = dest.get("name")
                else:
                    grounding_data = f"\n[INTERNAL DATA: Found {dest.get('name')} but couldn't calculate a route.]"
            else:
                grounding_data = "\n[INTERNAL DATA: Destination not found for navigation.]"

        # 3. Save user message
        await mongodb_service.save_chat_message(
            request.session_id, "location", "user", request.message, 
            metadata={"user_location": request.user_location.model_dump() if request.user_location else None}
        )
        
        # 4. Generate Agentic Response
        system_prompt = "You are the University of Bamenda Campus Guide. Be friendly and helpful. Use the [INTERNAL DATA] provided (if any) to answer accurately. If no data is found, admit it and suggest what the user can do."
        agent_prompt = f"{system_prompt}\n\nUser: {request.message}{grounding_data}"
        
        ai_response = await gemini_service.generate_response(agent_prompt, context=history)
        
        # 5. Save and Return
        await mongodb_service.save_chat_message(request.session_id, "location", "assistant", ai_response, metadata=metadata)
        await mongodb_service.disconnect()
        
        return ChatResponse(message=ai_response, session_id=request.session_id, metadata=metadata)
        
    except Exception as e:
        logger.error(f"Navigation Agent error: {e}")
        try: await mongodb_service.disconnect()
        except: pass
        return ChatResponse(
            status="error",
            message="I'm here to help, but having a quick technical hitch. Ask me about a campus location!",
            session_id=request.session_id
        )
