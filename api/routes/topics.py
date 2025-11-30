"""
Topics Routes Module

Handles all topic-related endpoints:
- Create and manage academic topics
- Search topics by keyword or category
- Check for duplicate topics
- Get topic details and statistics
- Future: AI-powered topic suggestions
"""

import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse

from api.models.topic_models import (
    TopicCreateRequest,
    TopicResponse,
    TopicSearchRequest,
    TopicSearchResponse,
    TopicUpdateRequest,
    TopicStatsResponse
)
from services.topic_service import TopicService
from services.mongodb_service import MongoDBService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/topics", tags=["Topics"])

# Initialize services
topic_service = None
mongodb_service = None


def get_services():
    """Initialize services if not already initialized."""
    global topic_service, mongodb_service
    
    if not mongodb_service:
        mongodb_service = MongoDBService()
    if not topic_service:
        topic_service = TopicService(mongodb_service)
    
    return {
        'topics': topic_service,
        'mongodb': mongodb_service
    }


@router.post("/", response_model=TopicResponse, status_code=201)
async def create_topic(request: TopicCreateRequest):
    """
    Create a new academic topic.
    
    Validates that topic doesn't already exist before creation.
    Automatically adds metadata like creation timestamp and initial stats.
    
    Args:
        request: TopicCreateRequest with topic details
    
    Returns:
        TopicResponse with created topic information
    
    Raises:
        HTTPException: 400 if topic already exists, 500 for server errors
    """
    try:
        services = get_services()
        logger.info(f"Creating topic: {request.title}")
        
        # Check for duplicates
        existing = services['topics'].check_duplicate(
            title=request.title,
            category=request.category
        )
        
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Topic already exists: {request.title}"
            )
        
        # Create topic
        topic_data = {
            'title': request.title,
            'category': request.category,
            'description': request.description,
            'difficulty': request.difficulty,
            'tags': request.tags,
            'prerequisites': request.prerequisites,
            'resources': request.resources,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'views': 0,
            'searches': 0
        }
        
        topic_id = services['topics'].add_topic(topic_data)
        
        if not topic_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to create topic"
            )
        
        # Get created topic
        created_topic = services['topics'].get_topic_by_id(topic_id)
        
        logger.info(f"Topic created successfully: {topic_id}")
        
        return TopicResponse(
            id=str(topic_id),
            title=created_topic['title'],
            category=created_topic['category'],
            description=created_topic['description'],
            difficulty=created_topic['difficulty'],
            tags=created_topic.get('tags', []),
            prerequisites=created_topic.get('prerequisites', []),
            resources=created_topic.get('resources', []),
            created_at=created_topic['created_at'],
            updated_at=created_topic['updated_at'],
            views=created_topic.get('views', 0),
            searches=created_topic.get('searches', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating topic: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create topic: {str(e)}"
        )


@router.get("/", response_model=List[TopicResponse])
async def list_topics(
    category: Optional[str] = Query(None, description="Filter by category"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty level"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of topics to return"),
    skip: int = Query(0, ge=0, description="Number of topics to skip")
):
    """
    List all topics with optional filtering.
    
    Supports filtering by category and difficulty level.
    Includes pagination with limit and skip parameters.
    
    Args:
        category: Optional category filter
        difficulty: Optional difficulty filter (beginner, intermediate, advanced)
        limit: Maximum results to return (1-100)
        skip: Number of results to skip for pagination
    
    Returns:
        List of TopicResponse objects
    """
    try:
        services = get_services()
        logger.info(f"Listing topics: category={category}, difficulty={difficulty}")
        
        # Build filter
        filter_query = {}
        if category:
            filter_query['category'] = category
        if difficulty:
            filter_query['difficulty'] = difficulty
        
        # Get topics
        topics = services['topics'].list_topics(
            filter_query=filter_query,
            limit=limit,
            skip=skip
        )
        
        # Convert to response format
        response = []
        for topic in topics:
            response.append(TopicResponse(
                id=str(topic['_id']),
                title=topic['title'],
                category=topic['category'],
                description=topic.get('description', ''),
                difficulty=topic.get('difficulty', 'intermediate'),
                tags=topic.get('tags', []),
                prerequisites=topic.get('prerequisites', []),
                resources=topic.get('resources', []),
                created_at=topic.get('created_at'),
                updated_at=topic.get('updated_at'),
                views=topic.get('views', 0),
                searches=topic.get('searches', 0)
            ))
        
        logger.info(f"Returned {len(response)} topics")
        return response
        
    except Exception as e:
        logger.error(f"Error listing topics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list topics: {str(e)}"
        )


@router.get("/{topic_id}", response_model=TopicResponse)
async def get_topic(
    topic_id: str = Path(..., description="Topic ID")
):
    """
    Get detailed information about a specific topic.
    
    Increments view count when topic is accessed.
    
    Args:
        topic_id: Topic identifier (MongoDB ObjectId)
    
    Returns:
        TopicResponse with complete topic information
    
    Raises:
        HTTPException: 404 if topic not found, 500 for server errors
    """
    try:
        services = get_services()
        logger.info(f"Getting topic: {topic_id}")
        
        # Get topic
        topic = services['topics'].get_topic_by_id(topic_id)
        
        if not topic:
            raise HTTPException(
                status_code=404,
                detail=f"Topic not found: {topic_id}"
            )
        
        # Increment view count
        services['topics'].increment_views(topic_id)
        
        return TopicResponse(
            id=str(topic['_id']),
            title=topic['title'],
            category=topic['category'],
            description=topic.get('description', ''),
            difficulty=topic.get('difficulty', 'intermediate'),
            tags=topic.get('tags', []),
            prerequisites=topic.get('prerequisites', []),
            resources=topic.get('resources', []),
            created_at=topic.get('created_at'),
            updated_at=topic.get('updated_at'),
            views=topic.get('views', 0) + 1,  # Include the current view
            searches=topic.get('searches', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting topic: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get topic: {str(e)}"
        )


@router.post("/search", response_model=TopicSearchResponse)
async def search_topics(request: TopicSearchRequest):
    """
    Search topics by keyword with optional filters.
    
    Searches across title, description, tags, and category fields.
    Increments search count for found topics.
    
    Args:
        request: TopicSearchRequest with query and filters
    
    Returns:
        TopicSearchResponse with matching topics and metadata
    """
    try:
        services = get_services()
        logger.info(f"Searching topics: '{request.query}'")
        
        # Search topics
        results = services['topics'].search_topics(
            query=request.query,
            category=request.category,
            difficulty=request.difficulty,
            limit=request.limit
        )
        
        # Increment search counts for found topics
        for topic in results:
            try:
                services['topics'].increment_searches(str(topic['_id']))
            except Exception as e:
                logger.warning(f"Failed to increment search count: {str(e)}")
        
        # Convert to response format
        topics = []
        for topic in results:
            topics.append(TopicResponse(
                id=str(topic['_id']),
                title=topic['title'],
                category=topic['category'],
                description=topic.get('description', ''),
                difficulty=topic.get('difficulty', 'intermediate'),
                tags=topic.get('tags', []),
                prerequisites=topic.get('prerequisites', []),
                resources=topic.get('resources', []),
                created_at=topic.get('created_at'),
                updated_at=topic.get('updated_at'),
                views=topic.get('views', 0),
                searches=topic.get('searches', 0)
            ))
        
        logger.info(f"Search found {len(topics)} results")
        
        return TopicSearchResponse(
            query=request.query,
            total_results=len(topics),
            topics=topics,
            filters_applied={
                'category': request.category,
                'difficulty': request.difficulty
            }
        )
        
    except Exception as e:
        logger.error(f"Error searching topics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.put("/{topic_id}", response_model=TopicResponse)
async def update_topic(
    topic_id: str = Path(..., description="Topic ID"),
    request: TopicUpdateRequest = Body(...)
):
    """
    Update an existing topic.
    
    Only provided fields will be updated. Automatically updates the updated_at timestamp.
    
    Args:
        topic_id: Topic identifier
        request: TopicUpdateRequest with fields to update
    
    Returns:
        TopicResponse with updated topic information
    
    Raises:
        HTTPException: 404 if topic not found, 500 for server errors
    """
    try:
        services = get_services()
        logger.info(f"Updating topic: {topic_id}")
        
        # Check if topic exists
        existing = services['topics'].get_topic_by_id(topic_id)
        if not existing:
            raise HTTPException(
                status_code=404,
                detail=f"Topic not found: {topic_id}"
            )
        
        # Build update data (only include provided fields)
        update_data = {'updated_at': datetime.utcnow()}
        
        if request.title is not None:
            update_data['title'] = request.title
        if request.description is not None:
            update_data['description'] = request.description
        if request.difficulty is not None:
            update_data['difficulty'] = request.difficulty
        if request.tags is not None:
            update_data['tags'] = request.tags
        if request.prerequisites is not None:
            update_data['prerequisites'] = request.prerequisites
        if request.resources is not None:
            update_data['resources'] = request.resources
        
        # Update topic
        success = services['topics'].update_topic(topic_id, update_data)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update topic"
            )
        
        # Get updated topic
        updated = services['topics'].get_topic_by_id(topic_id)
        
        logger.info(f"Topic updated successfully: {topic_id}")
        
        return TopicResponse(
            id=str(updated['_id']),
            title=updated['title'],
            category=updated['category'],
            description=updated.get('description', ''),
            difficulty=updated.get('difficulty', 'intermediate'),
            tags=updated.get('tags', []),
            prerequisites=updated.get('prerequisites', []),
            resources=updated.get('resources', []),
            created_at=updated.get('created_at'),
            updated_at=updated.get('updated_at'),
            views=updated.get('views', 0),
            searches=updated.get('searches', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating topic: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update topic: {str(e)}"
        )


@router.delete("/{topic_id}", status_code=204)
async def delete_topic(
    topic_id: str = Path(..., description="Topic ID")
):
    """
    Delete a topic.
    
    Args:
        topic_id: Topic identifier
    
    Returns:
        204 No Content on success
    
    Raises:
        HTTPException: 404 if topic not found, 500 for server errors
    """
    try:
        services = get_services()
        logger.info(f"Deleting topic: {topic_id}")
        
        # Check if topic exists
        existing = services['topics'].get_topic_by_id(topic_id)
        if not existing:
            raise HTTPException(
                status_code=404,
                detail=f"Topic not found: {topic_id}"
            )
        
        # Delete topic
        success = services['topics'].delete_topic(topic_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete topic"
            )
        
        logger.info(f"Topic deleted successfully: {topic_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting topic: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete topic: {str(e)}"
        )


@router.get("/stats/overview", response_model=TopicStatsResponse)
async def get_topic_statistics():
    """
    Get overall topic statistics.
    
    Returns aggregated statistics including:
    - Total topics count
    - Topics by category
    - Topics by difficulty
    - Most viewed topics
    - Most searched topics
    
    Returns:
        TopicStatsResponse with comprehensive statistics
    """
    try:
        services = get_services()
        logger.info("Getting topic statistics")
        
        # Get all topics for statistics
        all_topics = services['topics'].list_topics(limit=1000)
        
        # Calculate statistics
        total_topics = len(all_topics)
        
        # Count by category
        by_category = {}
        for topic in all_topics:
            category = topic.get('category', 'Uncategorized')
            by_category[category] = by_category.get(category, 0) + 1
        
        # Count by difficulty
        by_difficulty = {}
        for topic in all_topics:
            difficulty = topic.get('difficulty', 'intermediate')
            by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
        
        # Most viewed
        most_viewed = sorted(
            all_topics,
            key=lambda x: x.get('views', 0),
            reverse=True
        )[:10]
        
        # Most searched
        most_searched = sorted(
            all_topics,
            key=lambda x: x.get('searches', 0),
            reverse=True
        )[:10]
        
        logger.info(f"Statistics calculated for {total_topics} topics")
        
        return TopicStatsResponse(
            total_topics=total_topics,
            by_category=by_category,
            by_difficulty=by_difficulty,
            most_viewed=[
                {
                    'id': str(t['_id']),
                    'title': t['title'],
                    'views': t.get('views', 0)
                }
                for t in most_viewed
            ],
            most_searched=[
                {
                    'id': str(t['_id']),
                    'title': t['title'],
                    'searches': t.get('searches', 0)
                }
                for t in most_searched
            ]
        )
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/categories/list")
async def list_categories():
    """
    Get list of all unique categories.
    
    Returns:
        JSON response with list of category names
    """
    try:
        services = get_services()
        logger.info("Listing categories")
        
        categories = services['topics'].get_unique_categories()
        
        return JSONResponse(content={
            'categories': categories,
            'total': len(categories)
        })
        
    except Exception as e:
        logger.error(f"Error listing categories: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list categories: {str(e)}"
        )