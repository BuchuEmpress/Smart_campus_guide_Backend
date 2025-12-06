"""
Analytics Service Module

Tracks and analyzes user search behavior:
- Identify frequently searched locations
- Generate insights for cache optimization
- Provide usage statistics
- Track search performance

Every search query is logged here for analytics and offline cache generation.
Uses the asynchronous MongoDBService for non-blocking database operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import uuid

# Import our MongoDB service
from services.mongodb_service import MongoDBService  # Assuming this uses Motor/AsyncIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Service for tracking and analyzing search behavior.

    Logs every search query and provides analytics to identify
    popular locations and optimize offline cache.

    Attributes:
        mongo: MongoDB service instance for database operations
    """

    def __init__(self, mongo_service: Optional[MongoDBService] = None):
        """
        Initialize Analytics Service.
        """
        self.mongo = mongo_service or MongoDBService()
        logger.info("Analytics service initialized")

    async def connect(self) -> bool:
        """Connect to MongoDB database."""
        try:
            connected = await self.mongo.connect()
            if connected:
                logger.info("✅ Analytics service connected to database")
                return True
            logger.error("Failed to connect analytics service to database")
            return False
        except Exception as e:
            logger.error(f"Error connecting analytics service: {str(e)}")
            return False

    async def disconnect(self):
        """Disconnect from MongoDB database."""
        await self.mongo.disconnect()
        logger.info("Analytics service disconnected")

    async def log_search(
        self,
        query: str,
        location_id: Optional[str] = None,
        location_name: Optional[str] = None,
        location_type: Optional[str] = None,
        is_on_campus: Optional[bool] = None,
        user_location: Optional[Dict] = None,
        response_time_ms: Optional[int] = None,
        search_source: str = "unknown",
        session_id: Optional[str] = None,
        success: bool = True
    ) -> bool:
        """Log a search query for analytics."""
        try:
            search_data = {
                "query": query.lower().strip(),
                "location_id": location_id,
                "location_name": location_name,
                "location_type": location_type,
                "is_on_campus": is_on_campus,
                "user_location": user_location,
                "response_time_ms": response_time_ms,
                "search_source": search_source,
                "session_id": session_id,
                "success": success,
                "timestamp": datetime.utcnow(),
            }

            await self.mongo.analytics_collection.insert_one(search_data)
            logger.debug(f"Search logged: '{query}' → {location_name or 'not found'}")
            return True
        except Exception as e:
            logger.error(f"Error logging search: {str(e)}")
            return False

    async def get_popular_locations(
        self,
        limit: int = 20,
        days: Optional[int] = None,
        min_searches: int = 0
    ) -> List[Dict]:
        """Get the most frequently searched locations."""
        try:
            logger.info(f"Getting top {limit} popular locations (last {days or 'all'} days)")

            pipeline_match = {}
            if days:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                pipeline_match["timestamp"] = {"$gte": cutoff_date}
                logger.info(f"Filtering data after: {cutoff_date}")
            else:
                logger.info("Using all-time data for popular locations.")

            pipeline_match["location_id"] = {"$exists": True, "$ne": None}
            collection = self.mongo.analytics_collection

            pipeline = [
                {"$match": pipeline_match},
                {"$group": {
                    "_id": "$location_id",
                    "location_name": {"$first": "$location_name"},
                    "location_type": {"$first": "$location_type"},
                    "is_on_campus": {"$first": "$is_on_campus"},
                    "search_count": {"$sum": 1}
                }},
                {"$sort": {"search_count": -1}},
                {"$project": {
                    "_id": 0,
                    "location_id": "$_id",
                    "location_name": 1,
                    "location_type": 1,
                    "is_on_campus": 1,
                    "search_count": 1
                }}
            ]

            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=limit)

            if min_searches > 0:
                results = [loc for loc in results if loc.get("search_count", 0) >= min_searches]

            return results[:limit]

        except Exception as e:
            logger.error(f"Error getting popular locations: {str(e)}")
            return []

    async def get_search_statistics(self) -> Dict:
        """Get overall search statistics."""
        try:
            logger.info("Calculating search statistics")
            collection = self.mongo.analytics_collection

            total_searches = await collection.count_documents({})
            successful = await collection.count_documents({"success": True})
            failed = await collection.count_documents({"success": False})
            success_rate = (successful / total_searches * 100) if total_searches > 0 else 0

            pipeline = [
                {"$match": {"response_time_ms": {"$exists": True, "$ne": None}}},
                {"$group": {"_id": None, "avg_response_time": {"$avg": "$response_time_ms"}}}
            ]
            cursor = collection.aggregate(pipeline)
            result = await cursor.to_list(length=1)
            avg_response_time = result[0]["avg_response_time"] if result else 0

            pipeline_sources = [
                {"$group": {"_id": "$search_source", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            cursor_sources = collection.aggregate(pipeline_sources)
            sources = await cursor_sources.to_list(length=None)
            top_sources = {item["_id"]: item["count"] for item in sources}

            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            searches_today = await collection.count_documents({"timestamp": {"$gte": today_start}})
            week_start = datetime.utcnow() - timedelta(days=7)
            searches_week = await collection.count_documents({"timestamp": {"$gte": week_start}})

            stats = {
                "total_searches": total_searches,
                "successful_searches": successful,
                "failed_searches": failed,
                "success_rate": round(success_rate, 2),
                "avg_response_time_ms": round(avg_response_time, 2),
                "top_search_sources": top_sources,
                "searches_today": searches_today,
                "searches_this_week": searches_week
            }

            logger.info(f"Statistics calculated: {total_searches} total searches, {success_rate:.1f}% success rate")
            return stats

        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {
                "total_searches": 0,
                "successful_searches": 0,
                "failed_searches": 0,
                "success_rate": 0,
                "avg_response_time_ms": 0,
                "top_search_sources": {},
                "searches_today": 0,
                "searches_this_week": 0
            }

    async def get_trending_queries(self, limit: int = 10, days: int = 7) -> List[Dict]:
        """Get trending search queries."""
        try:
            logger.info(f"Getting top {limit} trending queries from last {days} days")
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            pipeline = [
                {"$match": {"timestamp": {"$gte": cutoff_date}, "success": True}},
                {"$group": {"_id": "$query", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": limit}
            ]

            collection = self.mongo.analytics_collection
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=limit)

            trending = [{"query": item["_id"], "count": item["count"], "growth": 0} for item in results]
            logger.info(f"Retrieved {len(trending)} trending queries")
            return trending

        except Exception as e:
            logger.error(f"Error getting trending queries: {str(e)}")
            return []

    def create_session_id(self) -> str:
        """Create a unique session ID."""
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        return f"session_{timestamp}_{unique_id}"
