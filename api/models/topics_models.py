"""
Topic API Models (Pydantic)

Pydantic models for all topic-related API endpoints.
Handles:
- CRUD operations
- Search & filtering
- AI-powered suggestions
- Semantic similarity checks
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

# ==========================
# STUDENT INFORMATION MODEL
# ==========================

class StudentInfo(BaseModel):
    """Student information."""
    name: Optional[str] = Field(None, description="Student's full name", min_length=1, max_length=200)
    matric: Optional[str] = Field(None, description="Matriculation number", min_length=1, max_length=50)


# ==========================
# TOPIC REQUEST MODELS
# ==========================

class TopicCreateRequest(BaseModel):
    """Request to create a new topic."""
    title: str
    department: str
    option: str
    year: int
    student: Optional[StudentInfo] = None
    description: Optional[str] = None
    status: Optional[str] = "reserved"
    keywords: Optional[List[str]] = []
    supervisor: Optional[str] = None


class TopicSearchRequest(BaseModel):
    """Search topics with filters."""
    query: Optional[str] = Field(None, description="Keyword search")
    department: Optional[str] = None
    option: Optional[str] = None
    year: Optional[int] = None
    status: Optional[str] = None
    limit: int = Field(50, ge=1, le=100)


class TopicUpdateRequest(BaseModel):
    """Update topic fields."""
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    keywords: Optional[List[str]] = None
    supervisor: Optional[str] = None


# ==========================
# AI-RELATED REQUEST MODELS
# ==========================

class TopicSuggestionRequest(BaseModel):
    """Request AI to suggest new topics."""
    department: str
    option: str
    count: int = 5
    # Allow int or str in keywords and convert to str
    keywords: Optional[List[Union[str, int]]] = []
    user_request: Optional[str] = None

    @validator('keywords', each_item=True)
    def convert_keywords_to_string(cls, v):
        return str(v)


class TopicImproveRequest(BaseModel):
    """Request AI to improve a topic."""
    title: str
    description: str
    department: str
    option: str
    user_instruction: Optional[str] = None


class TopicSimilarityRequest(BaseModel):
    """Check semantic similarity with existing topics."""
    title: str
    department: str
    option: str
    threshold: float = 0.8


# ==========================
# TOPIC RESPONSE MODELS
# ==========================

class TopicResponse(BaseModel):
    """Full topic details."""
    topic_id: str
    title: str
    department: str
    option: Union[str, List[str]]  # Accept both string and list for flexibility
    year: int
    student: Optional[StudentInfo] = None
    description: Optional[str] = None
    status: Optional[str] = None
    keywords: Optional[List[str]] = []
    supervisor: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TopicSearchResponse(BaseModel):
    """Response for search."""
    query: Optional[str]
    total_results: int
    topics: List[TopicResponse]
    filters_applied: Dict[str, Any] # Used Dict[str, Any] for flexibility


class TopicStatsResponse(BaseModel):
    """Aggregated statistics."""
    total_topics: int
    total_views: Optional[int] = 0
    total_searches: Optional[int] = 0
    by_department: Dict[str, int]
    by_option: Dict[str, int]
    by_category: Optional[Dict[str, int]] = None  # Alias for by_option
    by_year: Dict[str, int] 
    by_status: Optional[Dict[str, int]] = None
    by_difficulty: Optional[Dict[str, int]] = None  # Alias for by_status


# ==========================
# AI-RELATED RESPONSE MODELS
# ==========================

class TopicSuggestionResponse(BaseModel):
    """AI-generated topics."""
    # Assuming suggestions is a list of dictionaries where each dict is a topic
    suggestions: List[Dict[str, Any]] 


class TopicImproveResponse(BaseModel):
    """Improved topic data."""
    improved_title: str
    improved_description: str
    suggested_keywords: List[str]
    suggested_tags: Optional[List[str]] = None  # Alias for suggested_keywords
    suggested_status: Optional[str] = "reserved"
    suggested_difficulty: Optional[str] = None  # Alias for suggested_status


class TopicSimilarityResponse(BaseModel):
    """Semantic similarity results."""
    # topic_id, title, similarity_score, status
    similar_topics: List[Dict[str, Any]] 

class TopicChatRequest(BaseModel):
    """Request for topic chatbot conversation."""
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(..., description="Unique persistent session ID for memory")
    department: Optional[str] = None
    option: Optional[str] = None

class TopicChatResponse(BaseModel):
    """Response from topic chatbot."""
    status: str = "success"
    message: str
    session_id: str
    metadata: Optional[Dict[str, Any]] = None
