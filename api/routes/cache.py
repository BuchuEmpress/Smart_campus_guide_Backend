"""
Cache Routes Module

Handles offline cache management endpoints:
- Generate cache for offline mode
- Get cached data
- Cache status and metadata
- Manual cache refresh
"""

import logging
import os
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse

from api.models.cache_models import (
    CacheStatusResponse,
    CacheGenerateRequest,
    CacheDataResponse
)
# Assuming these services are synchronous (which is typical for I/O operations like file caching)
from services.osm_cache_service import CacheService
from services.analytics_service import AnalyticsService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/cache", tags=["Cache"])

# Initialize services
cache_service = None
analytics_service = None


def get_services():
    """Initialize services if not already initialized."""
    global cache_service, analytics_service
    
    if not cache_service:
        cache_service = CacheService()
    if not analytics_service:
        analytics_service = AnalyticsService()
    
    return {
        'cache': cache_service,
        'analytics': analytics_service
    }


@router.get("/status", response_model=CacheStatusResponse)
async def get_cache_status():
    """
    Get current cache status and metadata.
    
    Returns information about:
    - Cache existence and file path
    - Last generation timestamp
    - Cache file size
    - Number of locations cached
    - Cache validity status
    
    Returns:
        CacheStatusResponse with cache metadata
    """
    try:
        services = get_services()
        logger.info("Getting cache status")
        
        cache_path = "data/cache/popular_places_latest.json"
        cache_exists = os.path.exists(cache_path)
        
        if not cache_exists:
            return CacheStatusResponse(
                cache_exists=False,
                last_generated=None,
                file_size_kb=0,
                total_locations=0,
                is_stale=True,
                message="Cache has not been generated yet"
            )
        
        # Get file metadata
        file_stats = os.stat(cache_path)
        file_size_kb = file_stats.st_size / 1024
        last_modified = datetime.fromtimestamp(file_stats.st_mtime)
        
        # Load cache to get location count
        cache_data = services['cache'].load_cache()
        total_locations = len(cache_data.get('locations', []))
        
        # Check if cache is stale (older than 24 hours)
        time_since_update = datetime.now() - last_modified
        is_stale = time_since_update.total_seconds() > 86400  # 24 hours
        
        message = "Cache is up to date"
        if is_stale:
            message = f"Cache is stale (last updated {time_since_update.days} days ago)"
        
        logger.info(f"Cache status: {total_locations} locations, {file_size_kb:.2f} KB")
        
        return CacheStatusResponse(
            cache_exists=True,
            last_generated=last_modified,
            file_size_kb=round(file_size_kb, 2),
            total_locations=total_locations,
            is_stale=is_stale,
            message=message,
            cache_path=cache_path
        )
        
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache status: {str(e)}"
        )


@router.post("/generate", status_code=202)
async def generate_cache(
    background_tasks: BackgroundTasks,
    request: Optional[CacheGenerateRequest] = None
):
    """
    Generate or refresh offline cache.
    
    Cache generation runs in the background to avoid blocking the request.
    Includes popular locations based on search analytics.
    
    Args:
        background_tasks: FastAPI background tasks
        request: Optional CacheGenerateRequest with configuration
    
    Returns:
        202 Accepted with task status message
    """
    try:
        services = get_services()
        logger.info("Starting cache generation in background")
        
        # Default parameters
        min_searches = request.min_searches if request else 5
        limit = request.limit if request else 100
        
        # Add cache generation to background tasks
        background_tasks.add_task(
            _generate_cache_task,
            services['cache'],
            services['analytics'],
            min_searches,
            limit
        )
        
        return JSONResponse(
            status_code=202,
            content={
                'status': 'accepted',
                'message': 'Cache generation started in background',
                'parameters': {
                    'min_searches': min_searches,
                    'limit': limit
                },
                'timestamp': str(datetime.now())
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting cache generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start cache generation: {str(e)}"
        )


# ✅ FIX: Removed 'async' keyword from background task function
def _generate_cache_task( 
    cache_service: CacheService,
    analytics_service: AnalyticsService,
    min_searches: int,
    limit: int
):
    """
    Background task for cache generation.
    
    Args:
        cache_service: Cache service instance
        analytics_service: Analytics service instance
        min_searches: Minimum searches for a location to be cached
        limit: Maximum locations to cache
    """
    try:
        logger.info("Executing cache generation task")
        
        # Get popular locations from analytics
        popular_locations = analytics_service.get_popular_locations(
            min_searches=min_searches,
            limit=limit
        )
        
        # Generate cache
        success = cache_service.generate_cache(
            popular_locations=popular_locations
        )
        
        if success:
            logger.info(f"Cache generated successfully with {len(popular_locations)} locations")
        else:
            logger.error("Cache generation failed")
            
    except Exception as e:
        logger.error(f"Error in cache generation task: {str(e)}")


@router.get("/data", response_model=CacheDataResponse)
async def get_cache_data(
    include_details: bool = Query(True, description="Include full location details")
):
    """
    Get cached data for offline mode.
    
    Returns the current cache data with popular locations.
    Clients can use this data when offline.
    
    Args:
        include_details: If True, includes full location details; if False, only basic info
    
    Returns:
        CacheDataResponse with cached locations and metadata
    """
    try:
        services = get_services()
        logger.info("Getting cache data")
        
        cache_data = services['cache'].load_cache()
        
        if not cache_data:
            raise HTTPException(
                status_code=404,
                detail="Cache not found. Generate cache first using POST /cache/generate"
            )
        
        # Filter data based on include_details parameter
        locations = cache_data.get('locations', [])
        
        if not include_details:
            # Return only basic info
            locations = [
                {
                    'id': loc.get('id'),
                    'name': loc.get('name'),
                    'coordinates': loc.get('coordinates'),
                    'type': loc.get('type'),
                    'is_on_campus': loc.get('is_on_campus')
                }
                for loc in locations
            ]
        
        metadata = cache_data.get('metadata', {})
        
        logger.info(f"Returning cache data with {len(locations)} locations")
        
        return CacheDataResponse(
            locations=locations,
            metadata={
                'generated_at': metadata.get('generated_at'),
                'total_locations': len(locations),
                'version': metadata.get('version', '1.0'),
                'expires_at': metadata.get('expires_at')
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache data: {str(e)}"
        )


@router.get("/download")
async def download_cache():
    """
    Download cache file directly.
    
    Provides the cache JSON file for download.
    Useful for mobile apps that want to store cache locally.
    
    Returns:
        FileResponse with the cache JSON file
    
    Raises:
        HTTPException: 404 if cache doesn't exist
    """
    try:
        cache_path = "data/cache/popular_places_latest.json"
        
        if not os.path.exists(cache_path):
            raise HTTPException(
                status_code=404,
                detail="Cache file not found. Generate cache first."
            )
        
        logger.info("Serving cache file for download")
        
        return FileResponse(
            path=cache_path,
            media_type="application/json",
            filename=f"campus_cache_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download cache: {str(e)}"
        )


@router.delete("/clear", status_code=204)
async def clear_cache():
    """
    Clear/delete the current cache.
    
    Removes the cache file from the system.
    Use with caution - this will affect offline functionality.
    
    Returns:
        204 No Content on success
    """
    try:
        logger.info("Clearing cache")
        
        cache_path = "data/cache/popular_places_latest.json"
        
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info("Cache cleared successfully")
        else:
            logger.warning("No cache file to clear")
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get("/popular-locations")
async def get_popular_locations(
    limit: int = Query(20, ge=1, le=100, description="Number of locations to return"),
    min_searches: int = Query(5, ge=1, description="Minimum searches required")
):
    """
    Get popular locations based on search analytics.
    
    Returns most frequently searched locations without full cache generation.
    
    Args:
        limit: Maximum number of locations to return
        min_searches: Minimum number of searches for a location to be included
    
    Returns:
        JSON response with popular locations list
    """
    try:
        services = get_services()
        logger.info(f"Getting popular locations (limit: {limit}, min_searches: {min_searches})")
        
        popular = services['analytics'].get_popular_locations(
            min_searches=min_searches,
            limit=limit
        )
        
        return JSONResponse(content={
            'popular_locations': popular,
            'total': len(popular),
            'parameters': {
                'min_searches': min_searches,
                'limit': limit
            },
            'timestamp': str(datetime.now())
        })
        
    except Exception as e:
        logger.error(f"Error getting popular locations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get popular locations: {str(e)}"
        )


@router.get("/health")
async def cache_health_check():
    """
    Health check for cache system.
    
    Verifies that cache directory exists and is writable.
    
    Returns:
        JSON response with health status
    """
    try:
        cache_dir = "data/cache"
        
        # Check if directory exists
        dir_exists = os.path.exists(cache_dir)
        
        # Check if directory is writable
        is_writable = os.access(cache_dir, os.W_OK) if dir_exists else False
        
        # Check if cache file exists
        cache_file = os.path.join(cache_dir, "popular_places_latest.json")
        cache_exists = os.path.exists(cache_file)
        
        status = "healthy" if (dir_exists and is_writable) else "unhealthy"
        
        return JSONResponse(content={
            'status': status,
            'cache_directory_exists': dir_exists,
            'cache_directory_writable': is_writable,
            'cache_file_exists': cache_exists,
            'timestamp': str(datetime.now())
        })
        
    except Exception as e:
        logger.error(f"Error in cache health check: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': str(datetime.now())
            }
        )

# EOF