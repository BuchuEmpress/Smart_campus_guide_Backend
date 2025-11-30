"""
Smart Campus Guide - FastAPI Main Application

Main FastAPI application with:
- CORS configuration for React frontend
- Rate limiting to prevent abuse
- Organized route modules
- Comprehensive error handling
- Request/response logging
- OpenAPI documentation
"""

import os
import logging
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

# Import route modules
from api.routes import api_router

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI application
app = FastAPI(
    title="Smart Campus Guide API",
    description="""
    ## Smart Campus Guide Backend API
    
    A comprehensive navigation and information system for university campuses.
    
    ### Features:
    - 🗺️ **Navigation**: Get natural, human-like directions around campus
    - 🔍 **Location Search**: Find buildings, facilities, and off-campus locations
    - 📚 **Academic Topics**: Manage and search educational topics
    - 💾 **Offline Cache**: Support for offline functionality
    - 📊 **Analytics**: Track popular locations and search patterns
    
    ### Key Capabilities:
    - On-campus navigation using vector search (Qdrant)
    - Off-campus navigation using Google Maps
    - AI-powered natural language directions (Gemini)
    - Multiple travel modes (walking, driving, transit)
    - Real-time traffic information
    - Topic management with MongoDB
    
    ### Getting Started:
    1. Try the `/api/navigate` endpoint to get directions
    2. Use `/api/search` to find locations
    3. Explore `/api/topics` for academic content
    
    ### Authentication:
    Currently no authentication required (development mode).
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Smart Campus Guide Team",
        "email": "support@campusguide.com"
    },
    license_info={
        "name": "MIT License"
    }
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # React default port
        "http://localhost:5173",      # Vite default port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        # Add production domains here when deploying
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and responses."""
    start_time = datetime.now()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = (datetime.now() - start_time).total_seconds()
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"- Status: {response.status_code} "
            f"- Time: {process_time:.3f}s"
        )
        
        # Add custom header with processing time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        logger.error(f"Request failed: {request.method} {request.url.path} - Error: {str(e)}")
        raise


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG_MODE", "False") == "True" else "An unexpected error occurred",
            "path": request.url.path,
            "timestamp": str(datetime.now())
        }
    )


# Include all API routes
app.include_router(api_router)


# Root endpoint
@app.get("/", tags=["Root"])
@limiter.limit("100/minute")
async def root(request: Request):
    """
    Welcome endpoint with API information.
    
    Returns basic API info and links to documentation.
    """
    return {
        "message": "Welcome to Smart Campus Guide API",
        "version": "1.0.0",
        "status": "running",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "navigation": "/api/navigate",
            "search": "/api/search",
            "topics": "/api/topics",
            "cache": "/api/cache"
        },
        "timestamp": str(datetime.now())
    }


# Health check endpoint
@app.get("/health", tags=["Health"])
@limiter.limit("200/minute")
async def health_check(request: Request):
    """
    Health check endpoint.
    
    Returns service health status and system information.
    """
    try:
        # Check environment variables
        required_vars = [
            "GOOGLE_MAPS_API_KEY",
            "GEMINI_API_KEY",
            "QDRANT_HOST",
            "MONGODB_URI"
        ]
        
        env_status = {}
        for var in required_vars:
            env_status[var] = "configured" if os.getenv(var) else "missing"
        
        # Basic health check
        return {
            "status": "healthy",
            "timestamp": str(datetime.now()),
            "environment": env_status,
            "services": {
                "api": "operational",
                "rate_limiting": "active",
                "cors": "configured"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": str(datetime.now())
            }
        )


# API information endpoint
@app.get("/api/info", tags=["Info"])
@limiter.limit("100/minute")
async def api_info(request: Request):
    """
    Get detailed API information.
    
    Returns comprehensive information about available endpoints,
    rate limits, and usage guidelines.
    """
    return {
        "api_name": "Smart Campus Guide",
        "version": "1.0.0",
        "description": "Navigation and information system for university campuses",
        "rate_limits": {
            "default": "60 requests per minute per IP",
            "health_check": "200 requests per minute per IP"
        },
        "features": [
            "Natural language navigation",
            "On-campus and off-campus location search",
            "Multiple travel modes (walking, driving, transit)",
            "Academic topic management",
            "Offline cache support",
            "Search analytics"
        ],
        "endpoints": {
            "navigation": {
                "navigate": "POST /api/navigate - Get directions to a location",
                "search": "POST /api/search - Search for locations",
                "location_details": "GET /api/location/{id} - Get location details",
                "chat": "POST /api/chat - Conversational navigation"
            },
            "topics": {
                "create": "POST /api/topics - Create new topic",
                "list": "GET /api/topics - List all topics",
                "get": "GET /api/topics/{id} - Get topic details",
                "search": "POST /api/topics/search - Search topics",
                "update": "PUT /api/topics/{id} - Update topic",
                "delete": "DELETE /api/topics/{id} - Delete topic",
                "stats": "GET /api/topics/stats/overview - Get statistics"
            },
            "cache": {
                "status": "GET /api/cache/status - Get cache status",
                "generate": "POST /api/cache/generate - Generate cache",
                "data": "GET /api/cache/data - Get cached data",
                "download": "GET /api/cache/download - Download cache file",
                "popular": "GET /api/cache/popular-locations - Get popular locations"
            }
        },
        "support": {
            "documentation": "/docs",
            "email": "support@campusguide.com"
        },
        "timestamp": str(datetime.now())
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Execute on application startup."""
    logger.info("=" * 50)
    logger.info("Smart Campus Guide API Starting...")
    logger.info("=" * 50)
    
    # Check environment configuration
    required_vars = [
        "GOOGLE_MAPS_API_KEY",
        "GEMINI_API_KEY",
        "QDRANT_HOST",
        "MONGODB_URI"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("Some features may not work correctly!")
    else:
        logger.info("All environment variables configured ✓")
    
    logger.info("API Documentation available at: /docs")
    logger.info("API is ready to accept requests")
    logger.info("=" * 50)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown."""
    logger.info("Smart Campus Guide API shutting down...")


# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG_MODE", "True") == "True"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )