"""
Cache API Models (Pydantic)

Models for cache management endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class CacheStatusResponse(BaseModel):
    """Response with cache status information."""
    cache_exists: bool
    last_generated: Optional[datetime]
    file_size_kb: float
    total_locations: int
    is_stale: bool
    message: str
    cache_path: Optional[str] = None


class CacheGenerateRequest(BaseModel):
    """Request to generate cache."""
    min_searches: int = Field(default=5, ge=1, description="Minimum searches to include")
    limit: int = Field(default=100, ge=1, le=500, description="Maximum locations")


class CacheDataResponse(BaseModel):
    """Response with cached data."""
    locations: List[Dict]
    metadata: Dict