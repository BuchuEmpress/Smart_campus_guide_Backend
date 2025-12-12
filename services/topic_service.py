"""
Topic Service Module - COMPLETE VERSION

Handles all CRUD operations for final year project topics.
Now includes ALL methods required by topics.py routes.

Features:
- Add, update, delete topics
- Search with filters
- Check duplicates
- Track views and searches
- Get statistics
"""

import os
import logging
import hashlib
from typing import Dict, List, Optional
from datetime import datetime
from bson import ObjectId

from services.mongodb_service import MongoDBService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicService:
    """
    Complete service for managing final year project topics.
    
    All methods are SYNCHRONOUS (not async) to match MongoDB service.
    """
    
    def __init__(self, mongo_service: Optional[MongoDBService] = None):
        """Initialize Topic Service with sync MongoDB."""
        if mongo_service and hasattr(mongo_service, 'db') and mongo_service.db:
            self.mongo = mongo_service
        else:
            from pymongo import MongoClient
            from dotenv import load_dotenv
            load_dotenv()
            
            uri = os.getenv('MONGODB_URI')
            if not uri:
                logger.warning("MONGODB_URI not set")
                self.mongo = type('obj', (object,), {'db': None})()
                return
            
            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                client.admin.command('ping')
                self.mongo = type('obj', (object,), {
                    'db': client['smart_campus_db'],
                    'client': client
                })()
                logger.info("✅ MongoDB connected")
            except Exception as e:
                logger.error(f"MongoDB failed: {e}")
                self.mongo = type('obj', (object,), {'db': None})()
        logger.info("Topic service initialized")
    
    def add_topic(self, topic_data: Dict) -> Optional[str]:
        """
        Add a new topic to the database.
        
        Args:
            topic_data: Dictionary with topic information
        
        Returns:
            Topic ID if successful, None if failed
        """
        try:
            # Generate topic_id if not provided
            if 'topic_id' not in topic_data or not topic_data['topic_id']:
                timestamp = int(datetime.utcnow().timestamp())
                title_hash = hashlib.md5(topic_data['title'].encode()).hexdigest()[:8]
                topic_data['topic_id'] = f"topic_{timestamp}_{title_hash}"
            
            # Add timestamps
            topic_data['created_at'] = datetime.utcnow()
            topic_data['updated_at'] = datetime.utcnow()
            
            # Add default stats
            if 'views' not in topic_data:
                topic_data['views'] = 0
            if 'searches' not in topic_data:
                topic_data['searches'] = 0
            
            # Insert into MongoDB
            collection = self.mongo.db['topics']
            result = collection.insert_one(topic_data)
            
            logger.info(f"Topic added: {topic_data['title']} (ID: {result.inserted_id})")
            return topic_data['topic_id']
            
        except Exception as e:
            logger.error(f"Error adding topic: {str(e)}")
            return None
    
    def get_topic_by_id(self, topic_id: str) -> Optional[Dict]:
        """
        Get a specific topic by its custom topic_id.
        
        Args:
            topic_id: Custom topic ID string
        
        Returns:
            Topic dictionary if found, None otherwise
        """
        try:
            collection = self.mongo.db['topics']
            # Query by custom topic_id, not _id
            topic = collection.find_one({'topic_id': topic_id})
            return topic
        except Exception as e:
            logger.error(f"Error getting topic {topic_id}: {str(e)}")
            return None
    
    def list_topics(
        self,
        filter_query: Optional[Dict] = None,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict]:
        """
        List topics with optional filtering and pagination.
        
        Args:
            filter_query: MongoDB filter query
            limit: Maximum results
            skip: Number to skip (pagination)
        
        Returns:
            List of topics
        """
        try:
            collection = self.mongo.db['topics']
            query = filter_query or {}
            
            # Convert string year to int if present in query
            if 'year' in query and isinstance(query['year'], str):
                if query['year'].isdigit():
                    query['year'] = int(query['year'])

            cursor = collection.find(query).skip(skip).limit(limit)
            topics = list(cursor)
            
            logger.info(f"Retrieved {len(topics)} topics")
            return topics
            
        except Exception as e:
            logger.error(f"Error listing topics: {str(e)}")
            return []
    
    def search_topics(
        self,
        query: Optional[str] = None,
        department: Optional[str] = None,
        option: Optional[str] = None,
        year: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Search topics by keyword with filters.
        
        Args:
            query: Search query
            department: Filter by department
            option: Filter by option
            year: Filter by year
            status: Filter by status
            limit: Maximum results
        
        Returns:
            List of matching topics
        """
        try:
            collection = self.mongo.db['topics']
            
            # Build search filter
            search_filter = {}
            
            if query:
                search_filter['title'] = {'$regex': query, '$options': 'i'}
            
            if department:
                search_filter['department'] = department
            if option:
                search_filter['option'] = option
            if year:
                search_filter['year'] = year
            if status:
                search_filter['status'] = status
                
            cursor = collection.find(search_filter).limit(limit)
            results = list(cursor)
            
            logger.info(f"Search '{query}' found {len(results)} topics")
            return results
            
        except Exception as e:
            logger.error(f"Error searching topics: {str(e)}")
            return []
    
    def check_duplicate(self, title: str, category: Optional[str] = None) -> Optional[Dict]:
        """
        Check if a topic with similar title already exists.
        
        NOTE: This performs a strict title check. For semantic similarity, 
        TopicIntelligenceService should be used.
        
        Args:
            title: Topic title to check
            category: Optional category filter (ignored, kept for compatibility)
        
        Returns:
            Existing topic if found, None otherwise
        """
        try:
            collection = self.mongo.db['topics']
            
            # Build query - Case-insensitive partial match
            query = {'title': {'$regex': title, '$options': 'i'}}
            
            existing = collection.find_one(query)
            
            if existing:
                logger.info(f"Possible duplicate found for '{title}' (Title match)")
            
            return existing
            
        except Exception as e:
            logger.error(f"Error checking duplicate: {str(e)}")
            return None

    def find_duplicate(
        self,
        title: str,
        department: str,
        option: str,
        year: int
    ) -> Optional[Dict]:
        """
        Find exact duplicate topic.
        
        Args:
            title: Topic title
            department: Department
            option: Option
            year: Year
            
        Returns:
            Duplicate topic if found
        """
        try:
            collection = self.mongo.db['topics']
            query = {
                'title': title,
                'department': department,
                'option': option,
                'year': year
            }
            return collection.find_one(query)
        except Exception as e:
            logger.error(f"Error finding duplicate: {str(e)}")
            return None
    
    def update_topic(self, topic_id: str, update_data: Dict) -> bool:
        """
        Update an existing topic.
        
        Args:
            topic_id: Custom topic ID string
            update_data: Fields to update
        
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.mongo.db['topics']
            
            # Add updated timestamp
            update_data['updated_at'] = datetime.utcnow()
            
            result = collection.update_one(
                {'topic_id': topic_id},
                {'$set': update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"Topic {topic_id} updated")
                return True
            else:
                logger.warning(f"Topic {topic_id} not found or no changes")
                return False
                
        except Exception as e:
            logger.error(f"Error updating topic: {str(e)}")
            return False
    
    def delete_topic(self, topic_id: str) -> bool:
        """
        Delete a topic.
        
        Args:
            topic_id: Custom topic ID string
        
        Returns:
            True if deleted, False otherwise
        """
        try:
            collection = self.mongo.db['topics']
            
            result = collection.delete_one({'topic_id': topic_id})
            
            if result.deleted_count > 0:
                logger.info(f"Topic {topic_id} deleted")
                return True
            else:
                logger.warning(f"Topic {topic_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting topic: {str(e)}")
            return False
    
    def increment_views(self, topic_id: str) -> bool:
        """
        Increment view count for a topic.
        
        Args:
            topic_id: Custom topic ID string
        
        Returns:
            True if successful
        """
        try:
            collection = self.mongo.db['topics']
            
            collection.update_one(
                {'topic_id': topic_id},
                {'$inc': {'views': 1}}
            )
            
            logger.debug(f"Incremented views for topic {topic_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error incrementing views: {str(e)}")
            return False
    
    def increment_searches(self, topic_id: str) -> bool:
        """
        Increment search count for a topic.
        
        Args:
            topic_id: Custom topic ID string
        
        Returns:
            True if successful
        """
        try:
            collection = self.mongo.db['topics']
            
            collection.update_one(
                {'topic_id': topic_id},
                {'$inc': {'searches': 1}}
            )
            
            logger.debug(f"Incremented searches for topic {topic_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error incrementing searches: {str(e)}")
            return False
    
    def get_unique_categories(self) -> List[str]:
        """
        Get list of all unique categories.
        
        Returns:
            List of category names
        """
        try:
            collection = self.mongo.db['topics']
            # We don't have categories anymore, but keeping method for compatibility
            # returning empty list or maybe departments?
            return []
            
        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict:
        """
        Get topic statistics.
        
        Returns:
            Statistics dictionary
        """
        try:
            collection = self.mongo.db['topics']
            
            # Total count
            total = collection.count_documents({})
            
            # Aggregation for breakdowns
            pipeline = [
                {
                    '$facet': {
                        'by_status': [{'$group': {'_id': '$status', 'count': {'$sum': 1}}}],
                        'by_department': [{'$group': {'_id': '$department', 'count': {'$sum': 1}}}],
                        'by_option': [{'$group': {'_id': '$option', 'count': {'$sum': 1}}}],
                        'by_year': [{'$group': {'_id': '$year', 'count': {'$sum': 1}}}]
                    }
                }
            ]
            
            agg_result = list(collection.aggregate(pipeline))
            result = agg_result[0] if agg_result else {}
            
            stats = {
                'total_topics': total,
                'by_status': {item['_id']: item['count'] for item in result.get('by_status', []) if item['_id']},
                'by_department': {item['_id']: item['count'] for item in result.get('by_department', []) if item['_id']},
                'by_option': {item['_id']: item['count'] for item in result.get('by_option', []) if item['_id']},
                'by_year': {str(item['_id']): item['count'] for item in result.get('by_year', []) if item['_id']}
            }
            
            logger.info(f"Statistics: {total} total topics")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {
                'total_topics': 0, 
                'by_status': {},
                'by_department': {},
                'by_option': {},
                'by_year': {}
            }


if __name__ == "__main__":
    # Test
    print("Topic Service - All methods implemented ✅")