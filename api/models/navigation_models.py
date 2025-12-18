"""
Navigation API Models (Pydantic)

This module defines all Pydantic models for navigation endpoints.
These models handle:
- Request validation (ensure data is correct before processing)
- Response formatting (consistent API responses)
- API documentation (auto-generates Swagger docs)
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Optional, Dict, List, Literal, Any


# ============================================================================
# LOCATION & COORDINATE MODELS
# ============================================================================

class Coordinates(BaseModel):
    """Geographic coordinates."""
    lat: float = Field(description="Latitude")
    lng: float = Field(description="Longitude")


class Location(BaseModel):
    """Represents a physical location."""
    id: str = Field(description="Unique identifier for the location")
    name: Optional[str] = Field("Unnamed Location", description="Name of the location")
    description: Optional[str] = Field(None, description="A brief description of the location")
    latitude: float = Field(description="Latitude of the location")
    longitude: float = Field(description="Longitude of the location")
    type: str = Field("unknown", description="Type of location (e.g., building, landmark)")

# ============================================================================
# USER LOCATION MODELS
# ============================================================================

class UserLocation(BaseModel):
    """
    User's current location.
    
    Can be from GPS (automatic) or manually entered address.
    """
    model_config = ConfigDict(
        json_schema_extra = {
            "examples": [
                {
                    "source": "gps",
                    "lat": 6.010317,
                    "lon": 10.258816
                },
                {
                    "source": "manual",
                    "address": "University of Bamenda Main Gate"
                }
            ]
        }
    )

    source: Literal["gps", "manual"] = Field(
        description="How the location was obtained",
        examples=["gps", "manual"]
    )
    
    lat: Optional[float] = Field(
        None,
        ge=-90,
        le=90,
        description="Latitude in decimal degrees",
        examples=[6.010317]
    )
    
    lon: Optional[float] = Field(
        None,
        ge=-180,
        le=180,
        description="Longitude in decimal degrees",
        examples=[10.258816]
    )
    
    address: Optional[str] = Field(
        None,
        min_length=1,
        max_length=500,
        description="Address for manual location input",
        examples=["University of Bamenda Main Gate"]
    )
    
    @model_validator(mode='after')
    def validate_source_and_fields(self) -> 'UserLocation':
        if self.source == "gps":
            if self.lat is None or self.lon is None:
                raise ValueError("For source 'gps', 'lat' and 'lon' must be provided.")
            if self.address is not None:
                self.address = None
        elif self.source == "manual":
            if not self.address:
                raise ValueError("For source 'manual', 'address' must be provided.")
            if self.lat is not None or self.lon is not None:
                self.lat = None
                self.lon = None
        return self


# ============================================================================
# NAVIGATION & SEARCH REQUEST MODELS
# ============================================================================

class NavigationRequest(BaseModel):
    """Request for navigation/directions."""
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "query": "Where is the library?",
                "user_location": {
                    "source": "gps",
                    "lat": 6.010317,
                    "lon": 10.258816
                },
                "travel_mode": "walking"
            }
        }
    )
    
    query: str = Field(min_length=1, max_length=500)
    user_location: UserLocation
    travel_mode: Literal["walking", "driving", "cycling", "transit"] = "walking"
    preferences: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

class SearchRequest(BaseModel):
    """Request to search for locations."""
    query: str = Field(min_length=1, max_length=200)

# ============================================================================
# ROUTE & STEP MODELS
# ============================================================================

class RouteStep(BaseModel):
    """Single step in a route."""
    instruction: str
    distance: str
    duration: str

# ============================================================================
# API RESPONSE MODELS
# ============================================================================

class NavigationResponse(BaseModel):
    """Response with navigation directions or chat message."""
    status: Literal["success", "chat", "error"]
    message: str
    destination: Optional[Location] = None
    route: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    """Response for location search."""
    locations: List[Location]

class LocationDetailResponse(Location):
    """Detailed information about a specific location, including an enhanced description."""
    enhanced_description: str

class ChatRequest(BaseModel):
    """Request for chatbot conversation."""
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(..., description="Unique persistent session ID for memory")
    user_location: Optional[UserLocation] = None

class ChatResponse(BaseModel):
    """Response from the chatbot."""
    status: str = "success"
    message: str
    session_id: str
    metadata: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
