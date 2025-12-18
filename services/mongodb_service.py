"""
MongoDB Service Module

This module provides connection and operations for MongoDB Atlas database.
Handles all database operations for:
- Topics (final year project topics)
- Search analytics (tracking frequently searched locations)
- Popular locations cache (offline support)

Uses Motor for async MongoDB operations (required for FastAPI).
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

# Import environment variable support
from dotenv import load_dotenv

# Import Motor (async MongoDB driver for Python)
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

# Import PyMongo for some utility functions
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MongoDBService:
    """
    Service for MongoDB database operations.
    
    This service handles all database operations for the Smart Campus Guide:
    - Connection management
    - CRUD operations for topics
    - Analytics tracking
    - Popular locations caching
    
    Attributes:
        client: Motor async MongoDB client
        db: Database instance
        topics_collection: Collection for topics
        analytics_collection: Collection for search analytics
        popular_collection: Collection for popular locations
    
    Example:
        >>> mongo = MongoDBService()
        >>> await mongo.connect()
        >>> topic = await mongo.add_topic(topic_data)
        >>> await mongo.disconnect()
    """
    
    def __init__(self, connection_uri: Optional[str] = None, database_name: str = "smart_campus_db"):
        """
        Initialize MongoDB service.
        
        Args:
            connection_uri: MongoDB connection string. If None, reads from environment.
            database_name: Name of the database to use
        
        Raises:
            ValueError: If connection URI is not provided or found in environment.
        """
        # Get connection URI from parameter or environment
        self.connection_uri = connection_uri or os.getenv('MONGODB_URI')
        
        # Validate connection URI
        if not self.connection_uri:
            logger.error("MongoDB connection URI not found")
            raise ValueError(
                "MongoDB URI is required. "
                "Set MONGODB_URI environment variable or pass connection_uri parameter."
            )
        
        # Store database name
        self.database_name = database_name
        
        # Initialize client and database (will be set in connect())
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        
        # Collection references (will be set in connect())
        self.topics_collection: Optional[AsyncIOMotorCollection] = None
        self.analytics_collection: Optional[AsyncIOMotorCollection] = None
        self.popular_collection: Optional[AsyncIOMotorCollection] = None
        
        logger.info("MongoDB service initialized (not yet connected)")
    
    async def connect(self) -> bool:
        """
        Connect to MongoDB and setup collections.
        
        Returns:
            True if connection successful, False otherwise
        
        Example:
            >>> mongo = MongoDBService()
            >>> await mongo.connect()
            True
        """
        try:
            logger.info(f"Connecting to MongoDB: {self.database_name}")
            
            # Create Motor async client
            self.client = AsyncIOMotorClient(self.connection_uri)
            
            # Test the connection by running a simple command
            await self.client.admin.command('ping')
            
            # Get database reference
            self.db = self.client[self.database_name]
            
            # Get collection references
            self.topics_collection = self.db['topics']
            self.analytics_collection = self.db['search_analytics']
            self.popular_collection = self.db['popular_locations']
            self.chat_collection = self.db['chat_history']
            
            # Create indexes for better query performance
            await self._create_indexes()
            
            logger.info("✅ Successfully connected to MongoDB")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {str(e)}")
            return False
    
    async def disconnect(self):
        """
        Disconnect from MongoDB.
        
        Always call this when shutting down the application to close connections properly.
        """
        if self.client:
            logger.info("Closing MongoDB connection")
            self.client.close()
            logger.info("✅ MongoDB connection closed")
    
    async def _create_indexes(self):
        """
        Create database indexes for better query performance.
        
        Indexes speed up queries on frequently searched fields.
        This is called automatically during connect().
        """
        try:
            logger.info("Creating database indexes...")
            
            # Topics collection indexes
            await self.topics_collection.create_index([("department", ASCENDING)])
            await self.topics_collection.create_index([("option", ASCENDING)])
            await self.topics_collection.create_index([("year", DESCENDING)])
            await self.topics_collection.create_index([("status", ASCENDING)])
            await self.topics_collection.create_index([("topic_id", ASCENDING)], unique=True)
            
            # Analytics collection indexes
            await self.analytics_collection.create_index([("timestamp", DESCENDING)])
            await self.analytics_collection.create_index([("location_id", ASCENDING)])
            await self.analytics_collection.create_index([("is_on_campus", ASCENDING)])
            
            # Popular locations collection indexes
            await self.popular_collection.create_index([("location_id", ASCENDING)], unique=True)
            await self.popular_collection.create_index([("search_count", DESCENDING)])
            await self.popular_collection.create_index([("rank", ASCENDING)])

            # Chat history indexes
            await self.db['chat_history'].create_index([("session_id", ASCENDING)])
            await self.db['chat_history'].create_index([("timestamp", ASCENDING)])
            
            logger.info("✅ Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
    
    # ============================================================================
    # TOPICS OPERATIONS
    # ============================================================================
    
    async def add_topic(self, topic_data: Dict) -> Optional[str]:
        """
        Add a new topic to the database.
        
        Args:
            topic_data: Dictionary containing topic information:
                {
                    'topic_id': str,
                    'title': str,
                    'department': str,
                    'option': str,
                    'student': {'name': str, 'matric': str},
                    'year': int,
                    'status': str ('in_progress', 'completed', 'reserved'),
                    'keywords': List[str],
                    'supervisor': str,
                    'description': str
                }
        
        Returns:
            Topic ID if successful, None if failed
        
        Example:
            >>> topic = {
            ...     'topic_id': 'topic_001',
            ...     'title': 'Smart Campus Guide',
            ...     'department': 'Computer Engineering',
            ...     'option': 'Software Engineering',
            ...     'year': 2024,
            ...     'status': 'in_progress'
            ... }
            >>> topic_id = await mongo.add_topic(topic)
        """
        try:
            # Add timestamps
            topic_data['created_at'] = datetime.utcnow()
            topic_data['updated_at'] = datetime.utcnow()
            
            # Insert into database
            result = await self.topics_collection.insert_one(topic_data)
            
            logger.info(f"Topic added: {topic_data.get('title')} (ID: {topic_data.get('topic_id')})")
            
            return topic_data.get('topic_id')
            
        except DuplicateKeyError:
            logger.error(f"Topic with ID {topic_data.get('topic_id')} already exists")
            return None
        except Exception as e:
            logger.error(f"Error adding topic: {str(e)}")
            return None
    
    async def get_topics(
        self,
        department: Optional[str] = None,
        option: Optional[str] = None,
        status: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[Dict]:
        """
        Get topics with optional filtering.
        
        Args:
            department: Filter by department (optional)
            option: Filter by option (optional)
            status: Filter by status (optional)
            year: Filter by year (optional)
        
        Returns:
            List of topic dictionaries
        
        Example:
            >>> # Get all Software Engineering topics from 2024
            >>> topics = await mongo.get_topics(
            ...     option="Software Engineering",
            ...     year=2024
            ... )
        """
        try:
            # Build query filter
            query = {}
            
            if department:
                query['department'] = department
            
            if option:
                query['option'] = option
            
            if status:
                query['status'] = status
            
            if year:
                query['year'] = year
            
            # Execute query
            cursor = self.topics_collection.find(query).sort('created_at', DESCENDING)
            
            # Convert to list
            topics = await cursor.to_list(length=None)
            
            logger.info(f"Retrieved {len(topics)} topics")
            
            # Convert ObjectId to string for JSON serialization
            for topic in topics:
                topic['_id'] = str(topic['_id'])
            
            return topics
            
        except Exception as e:
            logger.error(f"Error retrieving topics: {str(e)}")
            return []
    
    async def check_topic_exists(self, title: str) -> Optional[Dict]:
        """
        Check if a topic with similar title already exists.
        
        Args:
            title: Topic title to check
        
        Returns:
            Existing topic dictionary if found, None otherwise
        
        Example:
            >>> existing = await mongo.check_topic_exists("Smart Campus Guide")
            >>> if existing:
            ...     print("Topic already taken!")
        """
        try:
            # Search for case-insensitive match
            query = {'title': {'$regex': f'^{title}$', '$options': 'i'}}
            
            result = await self.topics_collection.find_one(query)
            
            if result:
                logger.info(f"Topic exists: {title}")
                result['_id'] = str(result['_id'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking topic: {str(e)}")
            return None
    
    # ============================================================================
    # ANALYTICS OPERATIONS (For tracking frequently searched locations)
    # ============================================================================
    
    async def log_search(self, search_data: Dict) -> bool:
        """
        Log a search query for analytics.
        
        This is called EVERY TIME a user searches for a location.
        We track what they search for to identify popular locations.
        
        Args:
            search_data: Dictionary containing search information:
                {
                    'query': str,                       # What user searched
                    'location_id': str,                 # Qdrant location ID (if found)
                    'location_name': str,               # Location name
                    'location_type': str,               # building, landmark, etc.
                    'is_on_campus': bool,
                    'user_location': {'lat': float, 'lon': float},
                    'response_time_ms': int,
                    'search_source': str,               # 'qdrant' or 'google_maps'
                    'session_id': str
                }
        
        Returns:
            True if logged successfully, False otherwise
        
        Example:
            >>> search_info = {
            ...     'query': 'library',
            ...     'location_id': 'loc_123',
            ...     'location_name': 'University Library',
            ...     'is_on_campus': True,
            ...     'search_source': 'qdrant'
            ... }
            >>> await mongo.log_search(search_info)
        """
        try:
            # Add timestamp
            search_data['timestamp'] = datetime.utcnow()
            
            # Insert into analytics collection
            await self.analytics_collection.insert_one(search_data)
            
            logger.debug(f"Search logged: {search_data.get('query')} → {search_data.get('location_name')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging search: {str(e)}")
            return False
    
    async def get_top_searched_locations(self, limit: int = 20) -> List[Dict]:
        """
        Get the most frequently searched locations.
        
        This aggregates search analytics to find popular locations.
        Used for generating the offline cache.
        
        Args:
            limit: Number of top locations to return (default: 20)
        
        Returns:
            List of location dictionaries with search counts
        
        Example:
            >>> top_locations = await mongo.get_top_searched_locations(10)
            >>> for loc in top_locations:
            ...     print(f"{loc['location_name']}: {loc['count']} searches")
        """
        try:
            logger.info(f"Getting top {limit} searched locations")
            
            # Aggregation pipeline to count searches per location
            pipeline = [
                # Only include searches with location_id (successful searches)
                {'$match': {'location_id': {'$exists': True, '$ne': None}}},
                
                # Group by location and count
                {'$group': {
                    '_id': '$location_id',
                    'location_name': {'$first': '$location_name'},
                    'location_type': {'$first': '$location_type'},
                    'is_on_campus': {'$first': '$is_on_campus'},
                    'count': {'$sum': 1}
                }},
                
                # Sort by count (most searched first)
                {'$sort': {'count': DESCENDING}},
                
                # Limit results
                {'$limit': limit}
            ]
            
            # Execute aggregation
            cursor = self.analytics_collection.aggregate(pipeline)
            results = await cursor.to_list(length=limit)
            
            logger.info(f"Retrieved {len(results)} top locations")
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting top locations: {str(e)}")
            return []
    
    # ============================================================================
    # POPULAR LOCATIONS CACHE (For offline support)
    # ============================================================================
    
    async def update_popular_location(self, location_data: Dict) -> bool:
        """
        Update or insert a popular location in the cache.
        
        This is called by the cache generation service (nightly job).
        
        Args:
            location_data: Dictionary containing:
                {
                    'location_id': str,
                    'location_name': str,
                    'location_type': str,
                    'coordinates': {'lat': float, 'lon': float},
                    'description': str,
                    'search_count': int,
                    'search_count_week': int,
                    'search_count_month': int,
                    'last_searched': datetime,
                    'cached_directions': Dict,
                    'is_trending': bool,
                    'rank': int
                }
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add update timestamp
            location_data['updated_at'] = datetime.utcnow()
            
            # Upsert (update if exists, insert if not)
            await self.popular_collection.update_one(
                {'location_id': location_data['location_id']},
                {'$set': location_data},
                upsert=True
            )
            
            logger.debug(f"Popular location updated: {location_data.get('location_name')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating popular location: {str(e)}")
            return False
    
    async def get_popular_locations(self, limit: int = 20) -> List[Dict]:
        """
        Get the cached popular locations for offline use.
        
        Returns pre-computed data with directions included.
        
        Args:
            limit: Number of locations to return
        
        Returns:
            List of popular location dictionaries
        
        Example:
            >>> popular = await mongo.get_popular_locations(20)
            >>> # Send this to mobile app for offline cache
        """
        try:
            # Get popular locations sorted by rank
            cursor = self.popular_collection.find().sort('rank', ASCENDING).limit(limit)
            
            locations = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string
            for loc in locations:
                loc['_id'] = str(loc['_id'])
            
            logger.info(f"Retrieved {len(locations)} popular locations")
            
            return locations
            
        except Exception as e:
            logger.error(f"Error getting popular locations: {str(e)}")
            return []

    # ============================================================================
    # CHAT HISTORY OPERATIONS (For the Chatbots)
    # ============================================================================

    async def save_chat_message(self, session_id: str, chat_type: str, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Save a message (either user or bot) to the chat history.
        
        Args:
            session_id: The unique identifier for the user session
            chat_type: 'location' or 'topics'
            role: 'user' or 'assistant'
            content: The text message
            metadata: Any extra info (coordinates, intent, etc.)
        """
        try:
            message = {
                "session_id": session_id,
                "chat_type": chat_type,
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow(),
                "metadata": metadata or {}
            }
            await self.chat_collection.insert_one(message)
            logger.debug(f"Saved {role} message for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving chat message: {e}")
            return False

    async def get_chat_history(self, session_id: str, chat_type: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve recent chat history for a session.
        
        Args:
            session_id: Unique session ID
            chat_type: 'location' or 'topics'
            limit: How many messages to retrieve (default: 10)
        """
        try:
            cursor = self.chat_collection.find({
                "session_id": session_id,
                "chat_type": chat_type
            }).sort("timestamp", DESCENDING).limit(limit)
            
            history = await cursor.to_list(length=limit)
            
            # Sort chronologically for the AI
            history.reverse()
            
            # Formatted list of role/content
            return [{"role": msg["role"], "content": msg["content"]} for msg in history]
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []


# Testing and usage example
if __name__ == "__main__":
    import asyncio
    
    async def test_mongodb():
        """Test MongoDB service"""
        print("=" * 70)
        print("MONGODB SERVICE TEST")
        print("=" * 70)
        
        try:
            # Initialize service
            mongo = MongoDBService()
            print("✅ Service initialized\n")
            
            # Connect to database
            connected = await mongo.connect()
            if not connected:
                print("❌ Failed to connect to MongoDB")
                return
            
            print("✅ Connected to MongoDB\n")
            
            # Test 1: Add a topic
            print("Test 1: Add Topic")
            print("-" * 70)
            
            topic = {
                'topic_id': 'test_topic_001',
                'title': 'Smart Campus Navigation System',
                'department': 'Computer Engineering',
                'option': 'Software Engineering',
                'student': {'name': 'Test Student', 'matric': 'CE2024/001'},
                'year': 2024,
                'status': 'in_progress',
                'keywords': ['AI', 'Navigation', 'Mobile'],
                'supervisor': 'Dr. Test',
                'description': 'AI-powered campus guide system'
            }
            
            topic_id = await mongo.add_topic(topic)
            print(f"Topic added: {topic_id}")
            
            # Test 2: Get topics
            print("\nTest 2: Get Topics")
            print("-" * 70)
            
            topics = await mongo.get_topics(department='Computer Engineering')
            print(f"Found {len(topics)} topics")
            
            # Test 3: Log a search
            print("\nTest 3: Log Search")
            print("-" * 70)
            
            search = {
                'query': 'library',
                'location_id': 'loc_123',
                'location_name': 'University Library',
                'location_type': 'building',
                'is_on_campus': True,
                'search_source': 'qdrant',
                'session_id': 'test_session'
            }
            
            logged = await mongo.log_search(search)
            print(f"Search logged: {logged}")
            
            # Disconnect
            await mongo.disconnect()
            
            print("\n" + "=" * 70)
            print("✅ All tests completed!")
            print("=" * 70)
            
        except ValueError as e:
            print(f"❌ Configuration error: {str(e)}")
            print("Make sure MONGODB_URI is set in your .env file")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Run async test
    asyncio.run(test_mongodb())