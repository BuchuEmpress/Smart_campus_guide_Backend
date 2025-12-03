"""
Analytics Service Module

This module tracks and analyzes user search behavior to:
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
from services.mongodb_service import MongoDBService # Assuming this uses Motor/AsyncIO

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
    """
    
    def __init__(self, mongo_service: Optional[MongoDBService] = None):
        """
        Initialize Analytics Service.
        """
        # Use provided MongoDB service or create new one
        self.mongo = mongo_service or MongoDBService()
        
        logger.info("Analytics service initialized")
    
    async def connect(self) -> bool:
        """
        Connect to MongoDB database.
        """
        try:
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
            
            # FIX: Use the async analytics_collection and 'await'
            await self.mongo.analytics_collection.insert_one(search_data) 
            
            logger.debug(f"Search logged: '{query}' → {location_name or 'not found'}")
            
            return True
            
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
        """
        try:
            logger.info(f"Getting top {limit} popular locations (last {days or 'all'} days)")
            
            pipeline_match = {}
            # FIX: Implement date filtering 
            if days:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                pipeline_match['timestamp'] = {'$gte': cutoff_date}
                logger.info(f"Filtering data after: {cutoff_date}")
            else:
                 logger.info(f"Using all-time data for popular locations.")

            # We must only match searches that were successful and returned a location.
            pipeline_match['location_id'] = {'$exists': True, '$ne': None}

            
            # Use the async analytics collection
            collection = self.mongo.analytics_collection
            
            pipeline = [
                {'$match': pipeline_match}, # Apply the date filter and location filter
                {'$group': {
                    '_id': '$location_id',
                    'location_name': {'$first': '$location_name'},
                    'location_type': {'$first': '$location_type'},
                    'is_on_campus': {'$first': '$is_on_campus'},
                    'search_count': {'$sum': 1} # Changed 'count' to 'search_count' for clarity
                }},
                {'$sort': {'search_count': -1}},
                {'$limit': limit},
                # Project the output to match the desired format
                {'$project': {
                    '_id': 0, 
                    'location_id': '$_id', 
                    'location_name': 1, 
                    'location_type': 1, 
                    'is_on_campus': 1, 
                    'search_count': 1
                }}
            ]
            
            # FIX: Execute aggregation asynchronously
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=limit)
            
            logger.info(f"Retrieved {len(results)} popular locations")
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting popular locations: {str(e)}")
            return []
    
    async def get_search_statistics(self) -> Dict:
        """
        Get overall search statistics.
        """
        try:
            logger.info("Calculating search statistics")
            
            # FIX: Use the async analytics collection
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
            pipeline = [
                {'$match': {'response_time_ms': {'$exists': True, '$ne': None}}},
                {'$group': {
                    '_id': None,
                    'avg_response_time': {'$avg': '$response_time_ms'}
                }}
            ]
            
            # FIX: Execute aggregation asynchronously
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
            
            # FIX: Execute aggregation asynchronously
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
        
        Note: The 'growth' calculation is marked as a future TODO.
        """
        try:
            logger.info(f"Getting top {limit} trending queries from last {days} days")
            
            # Calculate date range
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Aggregation pipeline to count queries
            pipeline = [
                # Only recent and successful searches
                {'$match': {
                    'timestamp': {'$gte': cutoff_date},
                    'success': True 
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
            
            # FIX: Use the async analytics collection
            collection = self.mongo.analytics_collection
            
            # FIX: Execute aggregation asynchronously
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
        """
        # Generate session ID using timestamp + random component
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
            
            # Log a few more for testing the count logic
            await analytics.log_search(query="library", location_id="loc_123", location_name="University Library", success=True)
            await analytics.log_search(query="gate", location_id="loc_456", location_name="Main Gate", success=True)
            
            popular = await analytics.get_popular_locations(limit=5)
            print(f"Top {len(popular)} popular locations:")
            for i, loc in enumerate(popular, 1):
                print(f"{i}. {loc.get('location_name', 'Unknown')}: {loc.get('search_count', 0)} searches")
            
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
            print("✅ All tests completed! The service is now fully asynchronous.")
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Run async test
    asyncio.run(test_analytics())