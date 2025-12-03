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

import logging
import hashlib # ✅ FIXED: Moved import to the top
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
        """Initialize Topic Service."""
        self.mongo = mongo_service or MongoDBService()
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
                # hashlib is now imported at the top of the file
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
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error adding topic: {str(e)}")
            return None
    
    def get_topic_by_id(self, topic_id: str) -> Optional[Dict]:
        """
        Get a specific topic by its MongoDB ObjectId.
        
        Args:
            topic_id: MongoDB ObjectId as string
        
        Returns:
            Topic dictionary if found, None otherwise
        """
        try:
            collection = self.mongo.db['topics']
            topic = collection.find_one({'_id': ObjectId(topic_id)})
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
            List of topic dictionaries
        """
        try:
            collection = self.mongo.db['topics']
            query = filter_query or {}
            
            cursor = collection.find(query).skip(skip).limit(limit).sort('created_at', -1)
            topics = list(cursor)
            
            logger.info(f"Retrieved {len(topics)} topics")
            return topics
            
        except Exception as e:
            logger.error(f"Error listing topics: {str(e)}")
            return []
    
    def search_topics(
        self,
        query: str,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Search topics by keyword with filters.
        
        Args:
            query: Search query
            category: Filter by category
            difficulty: Filter by difficulty
            limit: Maximum results
        
        Returns:
            List of matching topics
        """
        try:
            collection = self.mongo.db['topics']
            
            # Build search filter
            search_filter = {
                '$or': [
                    {'title': {'$regex': query, '$options': 'i'}},
                    {'description': {'$regex': query, '$options': 'i'}},
                    {'tags': {'$regex': query, '$options': 'i'}}
                ]
            }
            
            # Add category filter
            if category:
                search_filter['category'] = category
            
            # Add difficulty filter
            if difficulty:
                search_filter['difficulty'] = difficulty
            
            # Execute search
            cursor = collection.find(search_filter).limit(limit).sort('searches', -1)
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
            category: Optional category filter
        
        Returns:
            Existing topic if found, None otherwise
        """
        try:
            collection = self.mongo.db['topics']
            
            # Build query - Case-insensitive partial match
            query = {'title': {'$regex': title, '$options': 'i'}}
            
            if category:
                query['category'] = category
            
            existing = collection.find_one(query)
            
            if existing:
                logger.info(f"Possible duplicate found for '{title}' (Title match)")
            
            return existing
            
        except Exception as e:
            logger.error(f"Error checking duplicate: {str(e)}")
            return None
    
    def update_topic(self, topic_id: str, update_data: Dict) -> bool:
        """
        Update an existing topic.
        
        Args:
            topic_id: Topic MongoDB ObjectId
            update_data: Fields to update
        
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.mongo.db['topics']
            
            # Add updated timestamp
            update_data['updated_at'] = datetime.utcnow()
            
            result = collection.update_one(
                {'_id': ObjectId(topic_id)},
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
            topic_id: Topic MongoDB ObjectId
        
        Returns:
            True if deleted, False otherwise
        """
        try:
            collection = self.mongo.db['topics']
            
            result = collection.delete_one({'_id': ObjectId(topic_id)})
            
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
            topic_id: Topic MongoDB ObjectId
        
        Returns:
            True if successful
        """
        try:
            collection = self.mongo.db['topics']
            
            collection.update_one(
                {'_id': ObjectId(topic_id)},
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
            topic_id: Topic MongoDB ObjectId
        
        Returns:
            True if successful
        """
        try:
            collection = self.mongo.db['topics']
            
            collection.update_one(
                {'_id': ObjectId(topic_id)},
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
            
            categories = collection.distinct('category')
            
            logger.info(f"Found {len(categories)} unique categories")
            return categories
            
        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
            return []
    
    def get_statistics(self, category: Optional[str] = None) -> Dict:
        """
        Get topic statistics.
        
        Args:
            category: Optional category filter
        
        Returns:
            Statistics dictionary
        """
        try:
            collection = self.mongo.db['topics']
            
            # Build filter
            query = {}
            if category:
                query['category'] = category
            
            # Total count
            total = collection.count_documents(query)
            
            # By difficulty
            pipeline = [
                {'$match': query},
                {'$group': {
                    '_id': '$difficulty',
                    'count': {'$sum': 1}
                }}
            ]
            
            by_difficulty = {}
            for result in collection.aggregate(pipeline):
                by_difficulty[result['_id']] = result['count']
            
            stats = {
                'total_topics': total,
                'by_difficulty': by_difficulty
            }
            
            logger.info(f"Statistics: {total} total topics")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {'total_topics': 0, 'by_difficulty': {}}


if __name__ == "__main__":
    # Test
    print("Topic Service - All methods implemented ✅")