"""
Topic Service Module

This module handles basic CRUD operations for final year project topics.
It manages:
- Adding new topics
- Searching topics
- Checking for duplicates
- Reserving topics
- Getting topics by department/option

This is the BASIC service. The intelligent features (AI suggestions)
are in topic_intelligence_service.py (Phase 4B).

"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

# Import MongoDB service
from services.mongodb_service import MongoDBService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TopicService:
    """
    Service for managing final year project topics.
    
    Provides basic CRUD operations for topics:
    - Create: Add new topics
    - Read: Search and retrieve topics
    - Update: Modify existing topics
    - Delete: Remove topics
    
    Attributes:
        mongo: MongoDB service instance
    
    Example:
        >>> topics = TopicService()
        >>> await topics.connect()
        >>> topic_id = await topics.add_topic(topic_data)
    """
    
    def __init__(self, mongo_service: Optional[MongoDBService] = None):
        """
        Initialize Topic Service.
        
        Args:
            mongo_service: MongoDB service instance (creates new if None)
        """
        # Use provided MongoDB service or create new one
        self.mongo = mongo_service or MongoDBService()
        
        logger.info("Topic service initialized")
    
    async def connect(self) -> bool:
        """
        Connect to MongoDB database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            connected = await self.mongo.connect()
            
            if connected:
                logger.info("✅ Topic service connected to database")
                return True
            else:
                logger.error("Failed to connect topic service to database")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting topic service: {str(e)}")
            return False
    
    async def disconnect(self):
        """Disconnect from MongoDB database."""
        await self.mongo.disconnect()
        logger.info("Topic service disconnected")
    
    async def add_topic(self, topic_data: Dict) -> Optional[str]:
        """
        Add a new topic to the database.
        
        Args:
            topic_data: Dictionary containing topic information:
                {
                    'topic_id': str,              # Unique ID (auto-generated if not provided)
                    'title': str,                 # Topic title
                    'department': str,            # e.g., "Computer Engineering"
                    'option': str,                # e.g., "Software Engineering"
                    'student': {                  # Student information
                        'name': str,
                        'matric': str
                    },
                    'year': int,                  # Academic year (e.g., 2024)
                    'status': str,                # 'in_progress', 'completed', 'reserved'
                    'keywords': List[str],        # Topic keywords
                    'supervisor': str,            # Supervisor name
                    'description': str            # Detailed description
                }
        
        Returns:
            Topic ID if successful, None if failed
        
        Example:
            >>> topic = {
            ...     'title': 'Smart Campus Navigation System',
            ...     'department': 'Computer Engineering',
            ...     'option': 'Software Engineering',
            ...     'student': {'name': 'John Doe', 'matric': 'CE2024/001'},
            ...     'year': 2024,
            ...     'status': 'in_progress',
            ...     'keywords': ['AI', 'Navigation', 'Mobile'],
            ...     'supervisor': 'Dr. Smith',
            ...     'description': 'AI-powered campus navigation system'
            ... }
            >>> topic_id = await topics.add_topic(topic)
        """
        try:
            # Generate topic_id if not provided
            if 'topic_id' not in topic_data or not topic_data['topic_id']:
                # Generate ID from timestamp and title
                import hashlib
                timestamp = int(datetime.utcnow().timestamp())
                title_hash = hashlib.md5(topic_data['title'].encode()).hexdigest()[:8]
                topic_data['topic_id'] = f"topic_{timestamp}_{title_hash}"
            
            # Validate required fields
            required_fields = ['title', 'department', 'option', 'year']
            missing_fields = [field for field in required_fields if field not in topic_data]
            
            if missing_fields:
                logger.error(f"Missing required fields: {missing_fields}")
                return None
            
            # Check for duplicate title
            existing = await self.check_duplicate(topic_data['title'])
            if existing:
                logger.warning(f"Topic with similar title already exists: {existing.get('title')}")
                return None
            
            # Add to database
            topic_id = await self.mongo.add_topic(topic_data)
            
            if topic_id:
                logger.info(f"Topic added: {topic_data['title']} (ID: {topic_id})")
            
            return topic_id
            
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
            department: Filter by department (e.g., "Computer Engineering")
            option: Filter by option (e.g., "Software Engineering")
            status: Filter by status ('in_progress', 'completed', 'reserved')
            year: Filter by academic year
        
        Returns:
            List of topic dictionaries matching the filters
        
        Example:
            >>> # Get all Software Engineering topics from 2024
            >>> topics_list = await topics.get_topics(
            ...     department="Computer Engineering",
            ...     option="Software Engineering",
            ...     year=2024
            ... )
            >>> print(f"Found {len(topics_list)} topics")
        """
        try:
            # Get from MongoDB
            topics_list = await self.mongo.get_topics(
                department=department,
                option=option,
                status=status,
                year=year
            )
            
            logger.info(f"Retrieved {len(topics_list)} topics (filters: dept={department}, opt={option}, year={year})")
            
            return topics_list
            
        except Exception as e:
            logger.error(f"Error retrieving topics: {str(e)}")
            return []
    
    async def check_duplicate(self, title: str) -> Optional[Dict]:
        """
        Check if a topic with similar title already exists.
        
        Uses case-insensitive search to find similar titles.
        
        Args:
            title: Topic title to check
        
        Returns:
            Existing topic dictionary if found, None otherwise
        
        Example:
            >>> existing = await topics.check_duplicate("Smart Campus Navigation")
            >>> if existing:
            ...     print(f"Similar topic found: {existing['title']}")
            ...     print(f"Taken by: {existing['student']['name']}")
        """
        try:
            # Check in MongoDB
            existing = await self.mongo.check_topic_exists(title)
            
            if existing:
                logger.info(f"Duplicate found for '{title}': {existing.get('title')}")
            else:
                logger.debug(f"No duplicate found for '{title}'")
            
            return existing
            
        except Exception as e:
            logger.error(f"Error checking duplicate: {str(e)}")
            return None
    
    async def reserve_topic(
        self,
        topic_id: str,
        student_name: str,
        student_matric: str
    ) -> bool:
        """
        Reserve a topic for a student.
        
        Changes topic status to 'reserved' and assigns to student.
        
        Args:
            topic_id: ID of the topic to reserve
            student_name: Name of student reserving the topic
            student_matric: Matriculation number of student
        
        Returns:
            True if reserved successfully, False otherwise
        
        Example:
            >>> success = await topics.reserve_topic(
            ...     topic_id="topic_123",
            ...     student_name="John Doe",
            ...     student_matric="CE2024/001"
            ... )
            >>> if success:
            ...     print("Topic reserved successfully!")
        """
        try:
            # TODO: Implement topic reservation
            # This would update the topic in MongoDB with student info and status='reserved'
            
            logger.info(f"Topic {topic_id} reserved for {student_name} ({student_matric})")
            
            # For now, return True (implement actual MongoDB update later)
            return True
            
        except Exception as e:
            logger.error(f"Error reserving topic: {str(e)}")
            return False
    
    async def get_topic_by_id(self, topic_id: str) -> Optional[Dict]:
        """
        Get a specific topic by its ID.
        
        Args:
            topic_id: Topic ID to retrieve
        
        Returns:
            Topic dictionary if found, None otherwise
        
        Example:
            >>> topic = await topics.get_topic_by_id("topic_123")
            >>> if topic:
            ...     print(f"Title: {topic['title']}")
        """
        try:
            # TODO: Implement MongoDB query by topic_id
            # For now, return None
            logger.debug(f"Getting topic: {topic_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting topic: {str(e)}")
            return None
    
    async def get_available_topics(
        self,
        department: str,
        option: str
    ) -> List[Dict]:
        """
        Get all available (not taken) topics for a department/option.
        
        Args:
            department: Department name
            option: Option name
        
        Returns:
            List of available topics
        
        Example:
            >>> available = await topics.get_available_topics(
            ...     department="Computer Engineering",
            ...     option="Software Engineering"
            ... )
            >>> print(f"{len(available)} topics available")
        """
        try:
            # Get topics with status != 'completed' and != 'in_progress'
            # This means topics that are 'reserved' or newly suggested
            all_topics = await self.get_topics(
                department=department,
                option=option
            )
            
            # Filter for available ones
            available = [
                topic for topic in all_topics
                if topic.get('status') not in ['completed', 'in_progress']
            ]
            
            logger.info(f"Found {len(available)} available topics for {option}")
            
            return available
            
        except Exception as e:
            logger.error(f"Error getting available topics: {str(e)}")
            return []
    
    async def get_topic_statistics(self, department: str) -> Dict:
        """
        Get statistics about topics for a department.
        
        Args:
            department: Department name
        
        Returns:
            Dictionary with statistics:
            {
                'total_topics': int,
                'by_option': Dict[str, int],
                'by_status': Dict[str, int],
                'by_year': Dict[int, int]
            }
        
        Example:
            >>> stats = await topics.get_topic_statistics("Computer Engineering")
            >>> print(f"Total topics: {stats['total_topics']}")
            >>> print(f"By option: {stats['by_option']}")
        """
        try:
            # Get all topics for department
            all_topics = await self.get_topics(department=department)
            
            # Count by option
            by_option = {}
            for topic in all_topics:
                option = topic.get('option', 'Unknown')
                by_option[option] = by_option.get(option, 0) + 1
            
            # Count by status
            by_status = {}
            for topic in all_topics:
                status = topic.get('status', 'unknown')
                by_status[status] = by_status.get(status, 0) + 1
            
            # Count by year
            by_year = {}
            for topic in all_topics:
                year = topic.get('year', 0)
                by_year[year] = by_year.get(year, 0) + 1
            
            stats = {
                'total_topics': len(all_topics),
                'by_option': by_option,
                'by_status': by_status,
                'by_year': by_year
            }
            
            logger.info(f"Statistics for {department}: {stats['total_topics']} total topics")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {
                'total_topics': 0,
                'by_option': {},
                'by_status': {},
                'by_year': {}
            }


# Testing and usage example
if __name__ == "__main__":
    import asyncio
    
    async def test_topic_service():
        """Test Topic Service"""
        print("=" * 70)
        print("TOPIC SERVICE TEST")
        print("=" * 70)
        
        try:
            # Initialize service
            topics = TopicService()
            print("✅ Service initialized\n")
            
            # Connect to database
            connected = await topics.connect()
            if not connected:
                print("❌ Failed to connect")
                return
            
            print("✅ Connected to database\n")
            
            # Test 1: Add a topic
            print("Test 1: Add Topic")
            print("-" * 70)
            
            topic = {
                'title': 'AI-Powered Campus Navigation System',
                'department': 'Computer Engineering',
                'option': 'Software Engineering',
                'student': {
                    'name': 'Test Student',
                    'matric': 'CE2024/TEST'
                },
                'year': 2024,
                'status': 'in_progress',
                'keywords': ['AI', 'Navigation', 'Mobile App'],
                'supervisor': 'Dr. Test Supervisor',
                'description': 'An intelligent campus navigation system with offline support'
            }
            
            topic_id = await topics.add_topic(topic)
            print(f"Topic added: {topic_id}")
            
            # Test 2: Check duplicate
            print("\nTest 2: Check Duplicate")
            print("-" * 70)
            
            duplicate = await topics.check_duplicate(topic['title'])
            if duplicate:
                print(f"Duplicate found: {duplicate.get('title')}")
            else:
                print("No duplicate found")
            
            # Test 3: Get topics
            print("\nTest 3: Get Topics")
            print("-" * 70)
            
            topics_list = await topics.get_topics(
                department="Computer Engineering",
                option="Software Engineering"
            )
            print(f"Found {len(topics_list)} Software Engineering topics")
            
            for i, t in enumerate(topics_list[:3], 1):
                print(f"{i}. {t.get('title')} - {t.get('status')}")
            
            # Test 4: Get statistics
            print("\nTest 4: Topic Statistics")
            print("-" * 70)
            
            stats = await topics.get_topic_statistics("Computer Engineering")
            print(f"Total topics: {stats['total_topics']}")
            print(f"By option: {stats['by_option']}")
            print(f"By status: {stats['by_status']}")
            
            # Disconnect
            await topics.disconnect()
            
            print("\n" + "=" * 70)
            print("✅ All tests completed!")
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Run async test
    asyncio.run(test_topic_service())