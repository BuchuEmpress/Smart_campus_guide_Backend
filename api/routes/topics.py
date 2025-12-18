"""
Topics Routes Module (COMPLETE FIX)

MAJOR CHANGES:
1. All asyncio.to_thread() calls properly wrapped for sync TopicService methods
2. Fixed response model mismatches (tags→keywords, difficulty→status)
3. All functionality preserved
4. Better error handling

Handles all topic-related endpoints:
- CRUD operations
- Search
- Statistics
- AI-powered suggestions, improvements, and similarity checks
"""

import logging
import asyncio
from typing import List, Dict
from api.models import topics_models as models
from services.topic_service import TopicService
from services.topic_intelligence_service import TopicIntelligenceService
from services.gemini_service import GeminiService
from services.mongodb_service import MongoDBService
from fastapi import APIRouter, HTTPException, Query, Path, Body, Depends

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/topics", tags=["Topics"])

# Initialize services
topic_service = TopicService()
topic_ai = TopicIntelligenceService(topic_service=topic_service)

# Dependencies
def get_gemini_service():
    return GeminiService()

def get_mongodb_service():
    return MongoDBService()

# ==========================
# HELPER FUNCTIONS
# ==========================
async def sync_topic_to_qdrant(topic: Dict):
    """Helper to sync a single topic to Qdrant."""
    try:
        from services.qdrant_service import QdrantService
        import uuid
        
        # Initialize service (lazy load model)
        qdrant_service = QdrantService(collection_name="topics")
        
        # Prepare payload
        t_id = topic.get('topic_id')
        if not t_id:
            return
            
        # Generate deterministic UUID from topic_id
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(t_id)))
        
        text = f"{topic.get('title', '')}. {topic.get('description', '')}. Tags: {', '.join(topic.get('tags', []))}"
        
        payload = {
            "id": point_id,
            "name": text,
            "topic_id": t_id,
            "title": topic.get('title'),
            "department": topic.get('department'),
            "option": topic.get('option'),
            "year": topic.get('year'),
            "status": topic.get('status')
        }
        
        # Upload (wrap in list)
        await qdrant_service.upload_points([payload])
        logger.info(f"Synced topic {t_id} to Qdrant")
        
    except Exception as e:
        logger.error(f"Failed to sync topic to Qdrant: {e}")

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
        
        # Sync to Qdrant (Async)
        await sync_topic_to_qdrant(topic)
        
        return models.TopicResponse(**topic)
    
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
    """List topics with optional filters (ALL CASE-INSENSITIVE)."""
    filters = {'department': department, 'option': option, 'year': year}
    
    # Clean filters (remove None values)
    clean_filters = {k: v for k, v in filters.items() if v is not None}
    
    # ✅ FIX: Wrap blocking list_topics call
    list_func = topic_service.list_topics
    topics = await asyncio.to_thread(list_func, filter_query=clean_filters)
    
    return [models.TopicResponse(**t) for t in topics]

@router.get("/{topic_id}", response_model=models.TopicResponse)
async def get_topic(topic_id: str = Path(...)):
    """Get a single topic by ID."""
    # ✅ FIX: Wrap blocking get_topic_by_id call
    get_func = topic_service.get_topic_by_id
    topic = await asyncio.to_thread(get_func, topic_id)
    
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    return models.TopicResponse(**topic)

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
        
    # Sync to Qdrant (Async)
    await sync_topic_to_qdrant(topic)
        
    return models.TopicResponse(**topic)

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
    """Search topics with filters (ALL CASE-INSENSITIVE)."""
    # 1. Search MongoDB
    search_func = topic_service.search_topics
    mongo_results = await asyncio.to_thread(
        search_func,
        query=request.query,
        department=request.department,
        option=request.option,
        year=request.year,
        status=request.status,
        limit=request.limit
    )
    logger.info(f"MongoDB search found {len(mongo_results)} results")
    
    # 2. Search Vector DB (if query provided)
    qdrant_results = []
    if request.query:
        try:
            # Initialize Qdrant for topics
            from services.qdrant_service import QdrantService
            qdrant_service = QdrantService(collection_name="topics")
            
            # Search
            qdrant_hits = await qdrant_service.search(
                query=request.query,
                limit=request.limit
            )
            
            # Extract IDs from Qdrant hits
            qdrant_ids = []
            for hit in qdrant_hits:
                # Qdrant payload might have 'id' or '_id' or 'topic_id'
                q_id = hit.get('topic_id') or hit.get('id') or hit.get('_id')
                if q_id:
                    qdrant_ids.append(str(q_id))
            
            # Fetch full topics from MongoDB for these IDs
            if qdrant_ids:
                # ✅ FIX: Wrap blocking list_topics call
                list_func = topic_service.list_topics
                qdrant_results = await asyncio.to_thread(
                    list_func,
                    filter_query={'topic_id': {'$in': qdrant_ids}},
                    limit=len(qdrant_ids)
                )
            
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            # Continue with just MongoDB results
    
    # 3. Merge Results (Deduplicate by topic_id)
    merged_map = {}
    
    # Add MongoDB results first
    for t in mongo_results:
        t_id = str(t.get('topic_id') or t.get('_id'))
        merged_map[t_id] = t
        
    # Add Qdrant results (if not already present)
    for t in qdrant_results:
        t_id = str(t.get('topic_id') or t.get('_id'))
        if t_id not in merged_map:
            merged_map[t_id] = t
            
    merged_results = list(merged_map.values())
            
    logger.info(f"Final merged results count: {len(merged_results)}")
    
    return models.TopicSearchResponse(
        query=request.query,
        total_results=len(merged_results),
        topics=[models.TopicResponse(**t) for t in merged_results],
        filters_applied=request.dict(exclude_none=True)
    )

# ==========================
# STATISTICS ENDPOINT
# ==========================
@router.get("/stats/overview", response_model=models.TopicStatsResponse)
async def get_topic_statistics():
    """Get topic statistics."""
    stats_func = topic_service.get_statistics
    stats = await asyncio.to_thread(stats_func)
    
    return models.TopicStatsResponse(
        total_topics=stats.get('total_topics', 0),
        total_views=stats.get('total_views', 0),
        total_searches=stats.get('total_searches', 0),
        by_department=stats.get('by_department', {}),
        by_option=stats.get('by_option', {}),
        by_category=stats.get('by_option', {}),  # Map to legacy name
        by_year=stats.get('by_year', {}),
        by_status=stats.get('by_status', {}),
        by_difficulty=stats.get('by_status', {})  # Map to legacy name
    )

# ==========================
# AI-POWERED ENDPOINTS (FIXED)
# ==========================
@router.post("/ai/suggest", response_model=models.TopicSuggestionResponse)
async def suggest_topics(request: models.TopicSuggestionRequest):
    """
    Generate AI-powered topic suggestions.
    NOW USES 'option' field correctly!
    """
    # ✅ FIX: Use 'option' parameter name (matches service signature)
    suggestions = await topic_ai.suggest_topics(
        option=request.option,  # Changed from category=request.option
        department=request.department,
        count=request.count,
        keywords=request.keywords,
        user_request=request.user_request
    )
    
    # ✅ FIX: Ensure each suggestion has BOTH modern and legacy keys
    # Maps internal keys (difficulty, tags) -> modern API keys (status, keywords)
    for s in suggestions:
        if 'difficulty' in s and 'status' not in s:
            s['status'] = s['difficulty']
        if 'tags' in s and 'keywords' not in s:
            s['keywords'] = s['tags']
        # Also ensure legacy keys exist if service returned modern ones (just in case)
        if 'status' in s and 'difficulty' not in s:
            s['difficulty'] = s['status']
        if 'keywords' in s and 'tags' not in s:
            s['tags'] = s['keywords']
            
    return models.TopicSuggestionResponse(suggestions=suggestions)

@router.post("/ai/improve", response_model=models.TopicImproveResponse)
async def improve_topic(request: models.TopicImproveRequest):
    """
    Improve an existing topic using AI.
    NOW USES 'option' field correctly!
    """
    # ✅ FIX: Use 'option' parameter name
    result = await topic_ai.improve_topic(
        title=request.title,
        description=request.description,
        option=request.option,  # Changed from category=request.option
        user_instruction=request.user_instruction
    )
    
    # ✅ FIX: Map response keys to match TopicImproveResponse model
    # Provides BOTH modern and legacy names for "no-stress" compatibility
    tags = result.get('suggested_tags', [])
    difficulty = result.get('suggested_difficulty', 'reserved')
    
    return models.TopicImproveResponse(
        improved_title=result.get('improved_title', request.title),
        improved_description=result.get('improved_description', request.description),
        suggested_keywords=tags, 
        suggested_tags=tags,
        suggested_status=difficulty,
        suggested_difficulty=difficulty
    )

@router.post("/ai/similarity", response_model=models.TopicSimilarityResponse)
async def check_similarity(request: models.TopicSimilarityRequest):
    """
    Check semantic similarity of a topic.
    NOW USES 'option' field correctly and is CASE-INSENSITIVE!
    """
    # Directly await the now-async method
    similar = await topic_ai.check_similarity(
        title=request.title,
        option=request.option,
        threshold=request.threshold
    )
    
    return models.TopicSimilarityResponse(similar_topics=similar)


# =============== CHATBOT ENDPOINT ===============

@router.post("/chat", response_model=models.TopicChatResponse)
async def topics_chat(
    request: models.TopicChatRequest,
    gemini_service: GeminiService = Depends(get_gemini_service),
    mongodb_service: MongoDBService = Depends(get_mongodb_service)
):
    """Conversational AI Agent for Final Year Project Guidance - Fully Integrated."""
    try:
        await mongodb_service.connect()
        history = await mongodb_service.get_chat_history(request.session_id, "topics")
        
        # 1. Extract Intent
        intent = await gemini_service.extract_topic_intent(request.message)
        action = intent.get("action", "chat")
        grounding_data = ""
        metadata = {"action_taken": action}

        # 2. Retrieve Data based on Action
        if action == "search":
            q = intent.get("query") or request.message
            func = topic_service.search_topics
            results = await asyncio.to_thread(func, query=q, limit=3)
            if results:
                found = [t.get("title") for t in results]
                grounding_data = f"\n[DATABASE INFO: Found these existing topics in our library: {', '.join(found)}]"
                metadata["results_count"] = len(results)

        elif action == "suggest":
            dept = intent.get("department") or request.department or "Computer Engineering"
            opt = intent.get("option") or request.option or "Software Engineering"
            # Get real AI suggestions
            suggestions = await topic_ai.suggest_topics(department=dept, option=opt, count=3)
            if suggestions:
                titles = [s.get("title") for s in suggestions]
                grounding_data = f"\n[AI SUGGESTIONS: I generated these potential ideas: {', '.join(titles)}]"
                metadata["suggestions_count"] = len(suggestions)

        elif action == "improve" and intent.get("title"):
            improvement = await topic_ai.improve_topic(
                title=intent.get("title"),
                description="", # Optional
                option=request.option or "Software Engineering"
            )
            grounding_data = f"\n[AI IMPROVEMENT: Suggested Title: {improvement.get('improved_title')}. Why: {improvement.get('improved_description')}]"
            metadata["improved"] = True

        # 3. Save user message
        await mongodb_service.save_chat_message(
            request.session_id, "topics", "user", request.message,
            metadata={"department": request.department, "option": request.option}
        )
        
        # 4. Generate Rich AI Response
        system_context = f"You are a helpful academic advisor at the University of Bamenda. You are guiding a student in {request.department or 'Engineering'}. Use the [DATA] provided to give real concrete examples. Be academic yet encouraging."
        agent_prompt = f"{system_context}\n\nStudent: {request.message}{grounding_data}"
        
        ai_response = await gemini_service.generate_response(agent_prompt, context=history)
        
        # 5. Save assistant response
        await mongodb_service.save_chat_message(request.session_id, "topics", "assistant", ai_response, metadata=metadata)
        await mongodb_service.disconnect()
        
        return models.TopicChatResponse(message=ai_response, session_id=request.session_id, metadata=metadata)
        
    except Exception as e:
        logger.error(f"Topics Agent error: {e}")
        try: await mongodb_service.disconnect()
        except: pass
        return models.TopicChatResponse(
            status="error",
            message="I'm here to help with your project, but hit a small snag. What research areas are you interested in?",
            session_id=request.session_id
        )