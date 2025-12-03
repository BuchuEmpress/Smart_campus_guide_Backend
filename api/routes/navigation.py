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

router = APIRouter()

def get_qdrant_service():
    return QdrantService()

def get_gemini_service():
    return GeminiService()

def get_osm_routing_service():
    return OSMRoutingService()

@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, qdrant_service: QdrantService = Depends(get_qdrant_service)):
    results = await qdrant_service.search(query=request.query, collection_name="locations")
    return SearchResponse(locations=[Location(**loc) for loc in results])

@router.post("/navigate", response_model=NavigationResponse)
async def navigate(request: NavigationRequest, 
                   gemini_service: GeminiService = Depends(get_gemini_service),
                   qdrant_service: QdrantService = Depends(get_qdrant_service),
                   osm_routing_service: OSMRoutingService = Depends(get_osm_routing_service)):
    
    intent = await gemini_service.extract_intent(request.query)

    if intent.get('action') == 'chat':
        response_text = await gemini_service.generate_response(prompt=f"User asked: {request.query}")
        return NavigationResponse(status="chat", message=response_text)

    if intent.get('action') == 'navigate':
        search_query = intent.get('location_query')
        if not search_query:
            return NavigationResponse(status="error", message="Could not determine destination from your query.")

        locations = await qdrant_service.search(query=search_query, collection_name="locations")
        if not locations:
            response_text = await gemini_service.generate_response(prompt=f"User wants to navigate to '{search_query}', but no matching locations were found.")
            return NavigationResponse(status="chat", message=response_text)
        
        destination = Location(**locations[0])
        
        start_coords = (request.user_location.lat, request.user_location.lon) if request.user_location and request.user_location.lat is not None else None
        if not start_coords:
             return NavigationResponse(status="error", message="User location is required for navigation.")

        end_coords = (destination.latitude, destination.longitude)
        
        route = await osm_routing_service.get_route(start_coords, end_coords, mode=request.travel_mode)
        if not route or not route.get('path'):
            return NavigationResponse(status="error", message="Could not find a route.")

        human_directions = await gemini_service.humanize_directions(route['path'])

        return NavigationResponse(
            status="success",
            message=human_directions,
            destination=destination,
            route=route
        )

    return NavigationResponse(status="error", message="Could not understand your request.")

@router.get("/location/{location_id}", response_model=LocationDetailResponse)
async def get_location_details(location_id: str, qdrant_service: QdrantService = Depends(get_qdrant_service), gemini_service: GeminiService = Depends(get_gemini_service)):
    # Using a direct ID lookup would be better, but search is used for now.
    results = await qdrant_service.search(query=location_id, collection_name="locations", limit=1)
    
    # Find the exact match from the search results
    for loc in results:
        if loc.get('id') == location_id:
            location = Location(**loc)
            enhanced_description = await gemini_service.enhance_description(location.description)
            return LocationDetailResponse(**location.dict(), enhanced_description=enhanced_description)
            
    raise HTTPException(status_code=404, detail=f"Location not found: {location_id}")

