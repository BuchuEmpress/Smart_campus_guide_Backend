"""
Analytics Service Module

This module tracks and analyzes user search behavior to:
- Identify frequently searched locations
- Generate insights for cache optimization
- Provide usage statistics
- Track search performance

Every search query is logged here for analytics and offline cache generation.

"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# Import our MongoDB service
from services.mongodb_service import MongoDBService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Service for tracking and analyzing search behavior.
    
    This service logs every search query and provides analytics
    to identify popular locations and optimize the offline cache.
    
    Attributes:
        mongo: MongoDB service instance for database operations
    
    Example:
        >>> analytics = AnalyticsService()
        >>> await analytics.connect()
        >>> await analytics.log_search(query="library", location_id="loc_123")
    """
    
    def __init__(self, mongo_service: Optional[MongoDBService] = None):
        """
        Initialize Analytics Service.
        
        Args:
            mongo_service: Existing MongoDB service instance. If None, creates new one.
        """
        # Use provided MongoDB service or create new one
        self.mongo = mongo_service or MongoDBService()
        
        logger.info("Analytics service initialized")
    
    async def connect(self) -> bool:
        """
        Connect to MongoDB database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Connect to MongoDB
            connected = await self.mongo.connect()
            
            if connected:
                logger.info("✅ Analytics service connected to database")
                return True
            else:
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
        """
        Log a search query for analytics.
        
        This is called EVERY TIME a user searches for a location.
        Critical for identifying popular locations for offline cache.
        
        Args:
            query: The search query string user entered
            location_id: ID of the location found (if any)
            location_name: Name of the location found (if any)
            location_type: Type of location (building, landmark, etc.)
            is_on_campus: Whether location is on campus
            user_location: User's GPS location as {'lat': float, 'lon': float}
            response_time_ms: How long the search took in milliseconds
            search_source: Where the result came from ('qdrant', 'google_maps', 'cache')
            session_id: User's session ID for tracking user journeys
            success: Whether the search was successful
        
        Returns:
            True if logged successfully, False otherwise
        
        Example:
            >>> await analytics.log_search(
            ...     query="library",
            ...     location_id="loc_123",
            ...     location_name="University Library",
            ...     location_type="building",
            ...     is_on_campus=True,
            ...     search_source="qdrant",
            ...     success=True
            ... )
        """
        try:
            # Build search data dictionary
            search_data = {
                'query': query.lower().strip(),  # Normalize query
                'location_id': location_id,
                'location_name': location_name,
                'location_type': location_type,
                'is_on_campus': is_on_campus,
                'user_location': user_location,
                'response_time_ms': response_time_ms,
                'search_source': search_source,
                'session_id': session_id,
                'success': success,
                'timestamp': datetime.utcnow()  # UTC timestamp
            }
            
            # Log to MongoDB
            logged = await self.mongo.log_search(search_data)
            
            if logged:
                logger.debug(f"Search logged: '{query}' → {location_name or 'not found'}")
            
            return logged
            
        except Exception as e:
            logger.error(f"Error logging search: {str(e)}")
            return False
    
    async def get_popular_locations(
        self,
        limit: int = 20,
        days: Optional[int] = None
    ) -> List[Dict]:
        """
        Get the most frequently searched locations.
        
        Args:
            limit: Number of top locations to return
            days: Only consider searches from last N days (None = all time)
        
        Returns:
            List of popular locations with search counts
            [
                {
                    'location_id': 'loc_123',
                    'location_name': 'University Library',
                    'location_type': 'building',
                    'is_on_campus': True,
                    'search_count': 156,
                    'last_searched': datetime
                },
                ...
            ]
        
        Example:
            >>> # Get top 10 locations from last 7 days
            >>> popular = await analytics.get_popular_locations(limit=10, days=7)
            >>> for loc in popular:
            ...     print(f"{loc['location_name']}: {loc['search_count']} searches")
        """
        try:
            logger.info(f"Getting top {limit} popular locations (last {days or 'all'} days)")
            
            # If days specified, only get recent searches
            if days:
                # Calculate cutoff date
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # TODO: Add date filtering to MongoDB aggregation
                # For now, use all-time data
                logger.warning(f"Date filtering not yet implemented, using all-time data")
            
            # Get popular locations from MongoDB
            popular = await self.mongo.get_top_searched_locations(limit=limit)
            
            logger.info(f"Retrieved {len(popular)} popular locations")
            
            return popular
            
        except Exception as e:
            logger.error(f"Error getting popular locations: {str(e)}")
            return []
    
    async def get_search_statistics(self) -> Dict:
        """
        Get overall search statistics.
        
        Returns:
            Dictionary with statistics:
            {
                'total_searches': int,
                'successful_searches': int,
                'failed_searches': int,
                'success_rate': float,
                'avg_response_time_ms': float,
                'top_search_sources': Dict,
                'searches_today': int,
                'searches_this_week': int
            }
        
        Example:
            >>> stats = await analytics.get_search_statistics()
            >>> print(f"Success rate: {stats['success_rate']}%")
        """
        try:
            logger.info("Calculating search statistics")
            
            # Get analytics collection
            collection = self.mongo.analytics_collection
            
            # Total searches
            total_searches = await collection.count_documents({})
            
            # Successful searches
            successful = await collection.count_documents({'success': True})
            
            # Failed searches
            failed = await collection.count_documents({'success': False})
            
            # Success rate
            success_rate = (successful / total_searches * 100) if total_searches > 0 else 0
            
            # Average response time
            # Aggregation pipeline to calculate average
            pipeline = [
                {'$match': {'response_time_ms': {'$exists': True, '$ne': None}}},
                {'$group': {
                    '_id': None,
                    'avg_response_time': {'$avg': '$response_time_ms'}
                }}
            ]
            
            cursor = collection.aggregate(pipeline)
            result = await cursor.to_list(length=1)
            avg_response_time = result[0]['avg_response_time'] if result else 0
            
            # Search sources breakdown
            pipeline_sources = [
                {'$group': {
                    '_id': '$search_source',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}}
            ]
            
            cursor_sources = collection.aggregate(pipeline_sources)
            sources = await cursor_sources.to_list(length=None)
            top_sources = {item['_id']: item['count'] for item in sources}
            
            # Searches today
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            searches_today = await collection.count_documents({
                'timestamp': {'$gte': today_start}
            })
            
            # Searches this week
            week_start = datetime.utcnow() - timedelta(days=7)
            searches_week = await collection.count_documents({
                'timestamp': {'$gte': week_start}
            })
            
            # Build statistics dictionary
            stats = {
                'total_searches': total_searches,
                'successful_searches': successful,
                'failed_searches': failed,
                'success_rate': round(success_rate, 2),
                'avg_response_time_ms': round(avg_response_time, 2),
                'top_search_sources': top_sources,
                'searches_today': searches_today,
                'searches_this_week': searches_week
            }
            
            logger.info(f"Statistics calculated: {total_searches} total searches, {success_rate:.1f}% success rate")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            # Return empty statistics on error
            return {
                'total_searches': 0,
                'successful_searches': 0,
                'failed_searches': 0,
                'success_rate': 0,
                'avg_response_time_ms': 0,
                'top_search_sources': {},
                'searches_today': 0,
                'searches_this_week': 0
            }
    
    async def get_trending_queries(self, limit: int = 10, days: int = 7) -> List[Dict]:
        """
        Get trending search queries (most searched in recent days).
        
        Args:
            limit: Number of trending queries to return
            days: Consider searches from last N days
        
        Returns:
            List of trending queries with counts
            [
                {
                    'query': 'library',
                    'count': 45,
                    'growth': 25.5  # Percentage increase from previous period
                },
                ...
            ]
        
        Example:
            >>> trending = await analytics.get_trending_queries(limit=5, days=7)
            >>> for item in trending:
            ...     print(f"{item['query']}: {item['count']} searches (↑{item['growth']}%)")
        """
        try:
            logger.info(f"Getting top {limit} trending queries from last {days} days")
            
            # Calculate date range
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Aggregation pipeline to count queries
            pipeline = [
                # Only recent searches
                {'$match': {
                    'timestamp': {'$gte': cutoff_date},
                    'success': True  # Only successful searches
                }},
                
                # Group by query and count
                {'$group': {
                    '_id': '$query',
                    'count': {'$sum': 1}
                }},
                
                # Sort by count
                {'$sort': {'count': -1}},
                
                # Limit results
                {'$limit': limit}
            ]
            
            collection = self.mongo.analytics_collection
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=limit)
            
            # Format results
            trending = [
                {
                    'query': item['_id'],
                    'count': item['count'],
                    'growth': 0  # TODO: Calculate growth vs previous period
                }
                for item in results
            ]
            
            logger.info(f"Retrieved {len(trending)} trending queries")
            
            return trending
            
        except Exception as e:
            logger.error(f"Error getting trending queries: {str(e)}")
            return []
    
    def create_session_id(self) -> str:
        """
        Create a unique session ID for tracking user journeys.
        
        Returns:
            Unique session ID string
        
        Example:
            >>> session = analytics.create_session_id()
            >>> print(session)
            "session_1732467890_abc123"
        """
        # Generate session ID using timestamp + random component
        import uuid
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        
        session_id = f"session_{timestamp}_{unique_id}"
        
        return session_id


# Testing and usage example
if __name__ == "__main__":
    import asyncio
    
    async def test_analytics():
        """Test Analytics Service"""
        print("=" * 70)
        print("ANALYTICS SERVICE TEST")
        print("=" * 70)
        
        try:
            # Initialize service
            analytics = AnalyticsService()
            print("✅ Service initialized\n")
            
            # Connect to database
            connected = await analytics.connect()
            if not connected:
                print("❌ Failed to connect")
                return
            
            print("✅ Connected to database\n")
            
            # Test 1: Log a search
            print("Test 1: Log Search")
            print("-" * 70)
            
            session = analytics.create_session_id()
            print(f"Session ID: {session}")
            
            logged = await analytics.log_search(
                query="library",
                location_id="loc_123",
                location_name="University Library",
                location_type="building",
                is_on_campus=True,
                search_source="qdrant",
                session_id=session,
                response_time_ms=150,
                success=True
            )
            
            print(f"Search logged: {logged}")
            
            # Test 2: Get popular locations
            print("\nTest 2: Popular Locations")
            print("-" * 70)
            
            popular = await analytics.get_popular_locations(limit=5)
            print(f"Top {len(popular)} popular locations:")
            for i, loc in enumerate(popular, 1):
                print(f"{i}. {loc.get('location_name', 'Unknown')}: {loc.get('count', 0)} searches")
            
            # Test 3: Get statistics
            print("\nTest 3: Search Statistics")
            print("-" * 70)
            
            stats = await analytics.get_search_statistics()
            print(f"Total searches: {stats['total_searches']}")
            print(f"Success rate: {stats['success_rate']}%")
            print(f"Avg response time: {stats['avg_response_time_ms']}ms")
            print(f"Searches today: {stats['searches_today']}")
            
            # Test 4: Trending queries
            print("\nTest 4: Trending Queries")
            print("-" * 70)
            
            trending = await analytics.get_trending_queries(limit=5, days=7)
            print(f"Top {len(trending)} trending queries:")
            for i, item in enumerate(trending, 1):
                print(f"{i}. '{item['query']}': {item['count']} searches")
            
            # Disconnect
            await analytics.disconnect()
            
            print("\n" + "=" * 70)
            print("✅ All tests completed!")
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Run async test
    asyncio.run(test_analytics())