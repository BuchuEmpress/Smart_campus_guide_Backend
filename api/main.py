"""
Smart Campus Guide - FastAPI Main Application (Improved Version)

✔ Clean dependency injection (DI)
✔ Centralized config management
✔ Organized structure for production
✔ Unified logging & rate limiting
✔ Better CORS security
✔ Global error handler + request logger
"""

import os
import logging
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from dotenv import load_dotenv
from api.routes import api_router

# ============================================================
# Load Environment Variables
# ============================================================
load_dotenv()

# ============================================================
# Logging Configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SmartCampusAPI")

# ============================================================
# Rate Limiting
# ============================================================
limiter = Limiter(key_func=get_remote_address)

# ============================================================
# LIFESPAN EVENTS
# ============================================================
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Smart Campus Guide API starting up...")

    missing = [
        var for var in ["GOOGLE_MAPS_API_KEY", "GEMINI_API_KEY", "MONGODB_URI"]
        if not os.getenv(var)
    ]

    if missing:
        logger.warning(f"⚠ Missing environment variables: {missing}")
    else:
        logger.info("✅ All environment variables loaded.")
    
    yield
    
    # Shutdown
    logger.info("🛑 Smart Campus Guide API shutting down...")

# ============================================================
# Initialize FastAPI App
# ============================================================
app = FastAPI(
    title="Smart Campus Guide API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    description="""
## Smart Campus Guide API

### Features
- Human-like navigation
- Location search
- Topic management
- AI-powered enhancements
- Offline caching
""",
    lifespan=lifespan,
)

# ============================================================
# CORS Configuration
# ============================================================
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ============================================================
# Middleware: Request Logging
# ============================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = datetime.now()

    logger.info(f"➡️ {request.method} {request.url.path}")

    try:
        response = await call_next(request)

        duration = (datetime.now() - start).total_seconds()
        logger.info(
            f"⬅️ {request.method} {request.url.path} | {response.status_code} | {duration:.3f}s"
        )

        response.headers["X-Process-Time"] = str(duration)
        return response

    except Exception as e:
        logger.error(f"❌ Error during request: {e}")
        raise


# ============================================================
# Global Exception Handler
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG_MODE") == "True" else "Unexpected error",
            "path": request.url.path,
            "timestamp": str(datetime.now())
        }
    )


# ============================================================
# Include Versioned API Router
# ============================================================
app.include_router(api_router, prefix="/api")


# ============================================================
# ROOT ENDPOINT
# ============================================================
@app.get("/", tags=["Root"])
@limiter.limit("100/minute")
async def root(request: Request):
    return {
        "message": "Welcome to Smart Campus Guide API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "timestamp": str(datetime.now())
    }


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================
@app.get("/health", tags=["Health"])
@limiter.limit("200/minute")
async def health_check(request: Request):

    required_env = [
        "GOOGLE_MAPS_API_KEY",
        "GEMINI_API_KEY",
        "MONGODB_URI",
        "QDRANT_HOST"
    ]

    env_status = {var: bool(os.getenv(var)) for var in required_env}

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





# ============================================================
# Run App (Dev Mode)
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG_MODE", "True") == "True"
    )
