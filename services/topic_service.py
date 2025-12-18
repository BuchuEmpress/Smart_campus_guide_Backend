"""
Topic Service Module - COMPLETE VERSION WITH CASE-INSENSITIVE FIXES

Handles all CRUD operations for final year project topics.
NOW WITH: Case-insensitive searching for department, option, and status.

Features:
- Add, update, delete topics
- Search with filters (CASE-INSENSITIVE)
- Check duplicates (CASE-INSENSITIVE)
- Track views and searches
- Get statistics
"""

import os
import logging
import hashlib
import re
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
    NOW WITH CASE-INSENSITIVE SUPPORT!
    """
    
    def __init__(self, mongo_service: Optional[MongoDBService] = None):
        """Initialize Topic Service with sync MongoDB."""
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
    
    @staticmethod
    def _escape_regex(text: str) -> str:
        """
        Escape special regex characters to prevent regex injection.
        
        Args:
            text: Text to escape
        
        Returns:
            Escaped text safe for regex
        """
        return re.escape(text)
    
    @staticmethod
    def _make_case_insensitive_filter(field: str, value: str) -> Dict:
        """
        Create a case-insensitive match filter for MongoDB.
        
        Args:
            field: Field name
            value: Value to match (case-insensitive)
        
        Returns:
            MongoDB filter dict with regex
        
        Example:
            >>> _make_case_insensitive_filter('department', 'Computer')
            {'department': {'$regex': 'Computer', '$options': 'i'}}
        """
        escaped = TopicService._escape_regex(value)
        # Removed anchors ^ and $ to allow partial matches and fix strictness issues
        return {field: {'$regex': f'{escaped}', '$options': 'i'}}
    
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
        CASE-INSENSITIVE LOOKUP!
        
        Args:
            topic_id: Custom topic ID string
        
        Returns:
            Topic dictionary if found, None otherwise
        """
        try:
            collection = self.mongo.db['topics']
            # Regular expression for case-insensitive ID match
            escaped_id = self._escape_regex(topic_id)
            topic = collection.find_one({'topic_id': {'$regex': f'^{escaped_id}$', '$options': 'i'}})
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
        NOW WITH CASE-INSENSITIVE SUPPORT!
        
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
            
            # Convert string filters to case-insensitive
            processed_query = {}
            for key, value in query.items():
                if key in ['department', 'option', 'status'] and isinstance(value, str):
                    # Make case-insensitive
                    processed_query.update(self._make_case_insensitive_filter(key, value))
                elif key == 'year' and isinstance(value, str) and value.isdigit():
                    # Convert string year to int
                    processed_query[key] = int(value)
                else:
                    # Keep as-is (including $in, $ne, etc.)
                    processed_query[key] = value

            cursor = collection.find(processed_query).skip(skip).limit(limit)
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
        ALL FILTERS ARE NOW CASE-INSENSITIVE!
        
        Args:
            query: Search query (searches in title - case-insensitive)
            department: Filter by department (case-insensitive)
            option: Filter by option (case-insensitive)
            year: Filter by year (exact match)
            status: Filter by status (case-insensitive)
            limit: Maximum results
        
        Returns:
            List of matching topics
        """
        try:
            collection = self.mongo.db['topics']
            
            # Build search filter
            search_filter = {}
            
            # Title search - case-insensitive partial match
            if query:
                escaped_query = self._escape_regex(query)
                search_filter['title'] = {'$regex': escaped_query, '$options': 'i'}
            
            # Department - case-insensitive exact match
            if department:
                search_filter.update(self._make_case_insensitive_filter('department', department))
            
            # Option - case-insensitive exact match
            if option:
                search_filter.update(self._make_case_insensitive_filter('option', option))
            
            # Year - exact match (no case-insensitive needed for numbers)
            if year:
                search_filter['year'] = year
            
            # Status - case-insensitive exact match
            if status:
                search_filter.update(self._make_case_insensitive_filter('status', status))
                
            cursor = collection.find(search_filter).limit(limit)
            results = list(cursor)
            
            logger.info(f"Search '{query}' with filters found {len(results)} topics")
            return results
            
        except Exception as e:
            logger.error(f"Error searching topics: {str(e)}")
            return []
    
    def check_duplicate(self, title: str, option: Optional[str] = None) -> Optional[Dict]:
        """
        Check if a topic with similar title already exists.
        NOW WITH CASE-INSENSITIVE TITLE AND OPTION MATCHING!
        
        NOTE: This performs a title check. For semantic similarity, 
        TopicIntelligenceService should be used.
        
        Args:
            title: Topic title to check (case-insensitive)
            option: Optional option filter (case-insensitive)
        
        Returns:
            Existing topic if found, None otherwise
        """
        try:
            collection = self.mongo.db['topics']
            
            # Build query - Case-insensitive partial match on title
            escaped_title = self._escape_regex(title)
            query = {'title': {'$regex': escaped_title, '$options': 'i'}}
            
            # Add option filter if provided (case-insensitive)
            if option:
                query.update(self._make_case_insensitive_filter('option', option))
            
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
        NOW WITH CASE-INSENSITIVE MATCHING!
        
        Args:
            title: Topic title (case-insensitive)
            department: Department (case-insensitive)
            option: Option (case-insensitive)
            year: Year (exact match)
            
        Returns:
            Duplicate topic if found
        """
        try:
            collection = self.mongo.db['topics']
            
            # Build case-insensitive query
            escaped_title = self._escape_regex(title)
            query = {
                'title': {'$regex': f'^{escaped_title}$', '$options': 'i'},
                'year': year
            }
            
            # Add case-insensitive department and option
            query.update(self._make_case_insensitive_filter('department', department))
            query.update(self._make_case_insensitive_filter('option', option))
            
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
                        'by_year': [{'$group': {'_id': '$year', 'count': {'$sum': 1}}}],
                        'totals': [{'$group': {
                            '_id': None, 
                            'total_views': {'$sum': '$views'},
                            'total_searches': {'$sum': '$searches'}
                        }}]
                    }
                }
            ]
            
            agg_result = list(collection.aggregate(pipeline))
            result = agg_result[0] if agg_result else {}
            
            totals = result.get('totals', [{}])[0] if result.get('totals') else {}
            
            stats = {
                'total_topics': total,
                'total_views': totals.get('total_views', 0),
                'total_searches': totals.get('total_searches', 0),
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
    # Test case-insensitive functionality
    print("✅ Topic Service - Complete with Case-Insensitive Support!")
    print("\nExample usage:")
    print("  search_topics(department='computer engineering')  # Matches 'Computer Engineering'")
    print("  search_topics(option='sen')                      # Matches 'SEN'")
    print("  search_topics(status='taken')                    # Matches 'Taken'")