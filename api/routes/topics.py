"""
Topics Routes Module

Handles all topic-related endpoints:
- CRUD operations
- Search
- Statistics
- AI-powered suggestions, improvements, and similarity checks
"""

import logging
import asyncio # Import asyncio for running blocking calls in a thread pool
from typing import List, Dict
from fastapi import APIRouter, HTTPException, Query, Path, Body

from api.models import topics_models as models
from services.topic_service import TopicService
from services.mongodb_service import MongoDBService
from services.topic_intelligence_service import TopicIntelligenceService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/topics", tags=["Topics"])

# Initialize services (assumed to be synchronous)
mongodb_service = MongoDBService()
topic_service = TopicService(mongodb_service)
# Note: TopicIntelligenceService relies on TopicService, which is blocking.
topic_ai = TopicIntelligenceService(topic_service=topic_service)

# ==========================
# CRUD ENDPOINTS
# ==========================
@router.post("/", response_model=models.TopicResponse)
async def create_topic(request: models.TopicCreateRequest):
    """
    Create a new topic.
    Checks for duplicates and saves to MongoDB.
    """
    try:
        # ✅ FIX: Wrap blocking find_duplicate call
        find_duplicate = topic_service.find_duplicate
        duplicates = await asyncio.to_thread(
            find_duplicate,
            title=request.title,
            department=request.department,
            option=request.option,
            year=request.year
        )
        if duplicates:
            raise HTTPException(status_code=400, detail="Topic already exists.")
        
        # ✅ FIX: Wrap blocking add_topic call
        add_topic = topic_service.add_topic
        topic_id = await asyncio.to_thread(add_topic, request.dict())
        
        # ✅ FIX: Wrap blocking get_topic_by_id call
        get_topic = topic_service.get_topic_by_id
        topic = await asyncio.to_thread(get_topic, topic_id)
        
        return models.TopicResponse(**topic, topic_id=str(topic_id))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating topic: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create topic")

@router.get("/", response_model=List[models.TopicResponse])
async def list_topics(
    department: str = Query(None),
    option: str = Query(None),
    year: int = Query(None)
):
    """List topics with optional filters."""
    filters = {'department': department, 'option': option, 'year': year}
    
    # Clean filters (remove None values)
    clean_filters = {k: v for k, v in filters.items() if v is not None}
    
    # ✅ FIX: Wrap blocking list_topics call
    list_func = topic_service.list_topics
    topics = await asyncio.to_thread(list_func, filters=clean_filters)
    
    return [models.TopicResponse(**t, topic_id=str(t['_id'])) for t in topics]

@router.get("/{topic_id}", response_model=models.TopicResponse)
async def get_topic(topic_id: str = Path(...)):
    """Get a single topic by ID."""
    # ✅ FIX: Wrap blocking get_topic_by_id call
    get_func = topic_service.get_topic_by_id
    topic = await asyncio.to_thread(get_func, topic_id)
    
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    return models.TopicResponse(**topic, topic_id=str(topic_id))

@router.put("/{topic_id}", response_model=models.TopicResponse)
async def update_topic(topic_id: str, request: models.TopicUpdateRequest = Body(...)):
    """Update topic fields."""
    # ✅ FIX: Wrap blocking update_topic call
    update_func = topic_service.update_topic
    await asyncio.to_thread(update_func, topic_id, request.dict(exclude_none=True))
    
    # ✅ FIX: Wrap blocking get_topic_by_id call
    get_func = topic_service.get_topic_by_id
    topic = await asyncio.to_thread(get_func, topic_id)
    
    if not topic:
        # Should ideally not happen if update succeeded, but good practice
        raise HTTPException(status_code=404, detail="Topic not found after update") 
        
    return models.TopicResponse(**topic, topic_id=str(topic_id))

@router.delete("/{topic_id}", status_code=204)
async def delete_topic(topic_id: str):
    """Delete a topic."""
    # ✅ FIX: Wrap blocking delete_topic call
    delete_func = topic_service.delete_topic
    await asyncio.to_thread(delete_func, topic_id)
    # Return 204 No Content

# ==========================
# SEARCH ENDPOINT
# ==========================
@router.post("/search", response_model=models.TopicSearchResponse)
async def search_topics(request: models.TopicSearchRequest):
    """Search topics with filters."""
    # ✅ FIX: Wrap blocking search_topics call
    search_func = topic_service.search_topics
    results = await asyncio.to_thread(
        search_func,
        query=request.query,
        department=request.department,
        option=request.option,
        year=request.year,
        limit=request.limit
    )
    return models.TopicSearchResponse(
        query=request.query,
        total_results=len(results),
        topics=[models.TopicResponse(**t, topic_id=str(t['_id'])) for t in results],
        filters_applied=request.dict(exclude_none=True)
    )

# ==========================
# STATISTICS ENDPOINT
# ==========================
@router.get("/stats/overview", response_model=models.TopicStatsResponse)
async def get_topic_statistics():
    """Get topic statistics."""
    # ✅ FIX: Wrap blocking list_topics call
    list_func = topic_service.list_topics
    all_topics = await asyncio.to_thread(list_func)
    
    total_topics = len(all_topics)
    by_department: Dict[str, int] = {}
    by_option: Dict[str, int] = {}
    by_year: Dict[int, int] = {}
    
    for t in all_topics:
        # Use .get() with default values to handle missing keys gracefully
        dept = t.get('department')
        opt = t.get('option')
        year = t.get('year')
        
        if dept:
            by_department[dept] = by_department.get(dept, 0) + 1
        if opt:
            by_option[opt] = by_option.get(opt, 0) + 1
        if year:
            # Ensure year is treated as int key if possible
            try:
                by_year[int(year)] = by_year.get(int(year), 0) + 1
            except (ValueError, TypeError):
                pass # Ignore if year is not a valid integer
                
    return models.TopicStatsResponse(
        total_topics=total_topics,
        by_department=by_department,
        by_option=by_option,
        by_year=by_year
    )

# ==========================
# AI-POWERED ENDPOINTS
# ==========================
@router.post("/ai/suggest", response_model=models.TopicSuggestionResponse)
async def suggest_topics(request: models.TopicSuggestionRequest):
    """Generate AI-powered topic suggestions."""
    # NOTE: Assuming TopicIntelligenceService methods are synchronous and need wrapping.
    # ✅ FIX: Wrap blocking suggest_topics call
    suggest_func = topic_ai.suggest_topics
    suggestions = await asyncio.to_thread(
        suggest_func,
        category=request.option,
        department=request.department,
        count=request.count,
        keywords=request.keywords
    )
    return models.TopicSuggestionResponse(suggestions=suggestions)

@router.post("/ai/improve", response_model=models.TopicImproveResponse)
async def improve_topic(request: models.TopicImproveRequest):
    """Improve an existing topic using AI."""
    # ✅ FIX: Wrap blocking improve_topic call
    improve_func = topic_ai.improve_topic
    result = await asyncio.to_thread(
        improve_func,
        title=request.title,
        description=request.description,
        category=request.option
    )
    return models.TopicImproveResponse(
        improved_title=result.get('improved_title', request.title),
        improved_description=result.get('improved_description', request.description),
        suggested_keywords=result.get('suggested_tags', []),
        suggested_status=result.get('suggested_difficulty', 'reserved')
    )

@router.post("/ai/similarity", response_model=models.TopicSimilarityResponse)
async def check_similarity(request: models.TopicSimilarityRequest):
    """Check semantic similarity of a topic."""
    # ✅ FIX: Wrap blocking check_similarity call
    similarity_func = topic_ai.check_similarity
    similar = await asyncio.to_thread(
        similarity_func,
        title=request.title,
        category=request.option,
        threshold=request.threshold
    )
    
    # The result 'similar' is assumed to be the list of similar topics as required by the model.
    return models.TopicSimilarityResponse(similar_topics=similar)

