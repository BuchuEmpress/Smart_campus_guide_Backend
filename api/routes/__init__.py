"""
Routes Package

Central module for importing and registering all API routes.
"""

from fastapi import APIRouter
from api.routes import navigation, topics, cache

# Create main router
api_router = APIRouter()

# Include all route modules
api_router.include_router(navigation.router)
api_router.include_router(topics.router)
api_router.include_router(cache.router)

__all__ = ['api_router', 'navigation', 'topics', 'cache']