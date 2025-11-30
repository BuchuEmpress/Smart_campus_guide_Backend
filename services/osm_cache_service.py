"""
Cache Service Module

This module generates and manages offline cache for:
- Popular locations (frequently searched)
- Pre-generated directions
- Offline availability

The cache allows students to access popular locations even without internet.
This is CRITICAL for campuses with poor connectivity.

"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Import our services
from services.mongodb_service import MongoDBService
from services.analytics_service import AnalyticsService
from services.qdrant_service import QdrantService
from services.osm_routing_service import MapsService
from services.gemini_service import GeminiService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CacheService:
    """
    Service for generating and managing offline cache.
    
    This service:
    1. Analyzes search analytics to find popular locations
    2. Generates pre-computed directions from common starting points
    3. Creates offline cache JSON files
    4. Updates popular locations in MongoDB
    
    The cache enables offline functionality for students with poor connectivity.
    
    Attributes:
        mongo: MongoDB service instance
        analytics: Analytics service instance
        qdrant: Qdrant service instance (for location data)
        maps: Google Maps service instance (for directions)
        gemini: Gemini AI service instance (for humanized directions)
    
    Example:
        >>> cache = CacheService()
        >>> await cache.connect_all()
        >>> await cache.generate_offline_cache(top_n=20)
    """
    
    def __init__(
        self,
        mongo_service: Optional[MongoDBService] = None,
        analytics_service: Optional[AnalyticsService] = None,
        qdrant_service: Optional[QdrantService] = None
    ):
        """
        Initialize Cache Service.
        
        Args:
            mongo_service: MongoDB service instance (creates new if None)
            analytics_service: Analytics service instance (creates new if None)
            qdrant_service: Qdrant service instance (creates new if None)
        """
        # Initialize or use provided services
        self.mongo = mongo_service or MongoDBService()
        self.analytics = analytics_service or AnalyticsService(self.mongo)
        self.qdrant = qdrant_service or QdrantService()
        
        # These will be initialized when needed
        self.maps: Optional[MapsService] = None
        self.gemini: Optional[GeminiService] = None
        
        # Cache output directory
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Cache service initialized")
    
    async def connect_all(self) -> bool:
        """
        Connect all required services.
        
        Returns:
            True if all connections successful, False otherwise
        """
        try:
            logger.info("Connecting cache service dependencies...")
            
            # Connect MongoDB
            mongo_connected = await self.mongo.connect()
            if not mongo_connected:
                logger.error("Failed to connect to MongoDB")
                return False
            
            # Connect Analytics (uses same MongoDB connection)
            analytics_connected = await self.analytics.connect()
            if not analytics_connected:
                logger.error("Failed to connect analytics")
                return False
            
            # Initialize Maps service (if needed for directions)
            try:
                self.maps = MapsService()
                logger.info("Maps service ready")
            except Exception as e:
                logger.warning(f"Maps service unavailable: {str(e)}")
            
            # Initialize Gemini service (if needed for humanized directions)
            try:
                self.gemini = GeminiService()
                logger.info("Gemini service ready")
            except Exception as e:
                logger.warning(f"Gemini service unavailable: {str(e)}")
            
            logger.info("✅ All cache service dependencies connected")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting cache services: {str(e)}")
            return False
    
    async def disconnect_all(self):
        """Disconnect all services."""
        await self.mongo.disconnect()
        await self.analytics.disconnect()
        logger.info("Cache service disconnected")
    
    async def generate_offline_cache(
        self,
        top_n: int = 20,
        common_starting_points: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate offline cache for top N most searched locations.
        
        This is the MAIN function that creates the offline cache.
        It should be run as a scheduled job (e.g., daily at midnight).
        
        Args:
            top_n: Number of popular locations to include in cache
            common_starting_points: List of common starting points for pre-generating directions
                                   e.g., [{'name': 'Main Gate', 'lat': 6.01, 'lon': 10.25}]
        
        Returns:
            Dictionary with generation statistics:
            {
                'success': bool,
                'locations_cached': int,
                'directions_generated': int,
                'cache_file': str,
                'generated_at': datetime
            }
        
        Example:
            >>> # Run this daily to update cache
            >>> starting_points = [
            ...     {'name': 'Main Gate', 'lat': 6.010317, 'lon': 10.258816},
            ...     {'name': 'Cafeteria', 'lat': 6.010500, 'lon': 10.259000}
            ... ]
            >>> result = await cache.generate_offline_cache(20, starting_points)
            >>> print(f"Cached {result['locations_cached']} locations")
        """
        try:
            logger.info(f"Generating offline cache for top {top_n} locations...")
            
            # Step 1: Get popular locations from analytics
            logger.info("Step 1: Analyzing search data...")
            popular_locations = await self.analytics.get_popular_locations(limit=top_n)
            
            if not popular_locations:
                logger.warning("No popular locations found in analytics")
                return {
                    'success': False,
                    'locations_cached': 0,
                    'directions_generated': 0,
                    'error': 'No popular locations found'
                }
            
            logger.info(f"Found {len(popular_locations)} popular locations")
            
            # Step 2: Enrich each location with full data from Qdrant
            logger.info("Step 2: Enriching location data...")
            cached_locations = []
            directions_count = 0
            
            for i, pop_loc in enumerate(popular_locations, 1):
                logger.info(f"Processing {i}/{len(popular_locations)}: {pop_loc.get('location_name')}")
                
                try:
                    # Get full location data from Qdrant (if available)
                    location_id = pop_loc.get('_id')  # This is actually location_id from aggregation
                    
                    # Build cached location object
                    cached_location = {
                        'location_id': location_id,
                        'location_name': pop_loc.get('location_name', 'Unknown'),
                        'location_type': pop_loc.get('location_type', 'unknown'),
                        'is_on_campus': pop_loc.get('is_on_campus', True),
                        'search_count': pop_loc.get('count', 0),
                        'rank': i,
                        'cached_directions': {},
                        'description': '',  # Will be filled from Qdrant if available
                        'coordinates': None  # Will be filled from Qdrant if available
                    }
                    
                    # Try to get full details from Qdrant
                    # Note: This would require implementing a get_by_id method in qdrant_service
                    # For now, we'll use the basic data we have
                    
                    # Step 3: Pre-generate directions from common starting points
                    if common_starting_points and self.maps and cached_location['coordinates']:
                        logger.debug(f"Generating directions for {cached_location['location_name']}")
                        
                        for start_point in common_starting_points:
                            try:
                                # Get route from Google Maps
                                route = self.maps.get_directions(
                                    origin=(start_point['lat'], start_point['lon']),
                                    destination=(
                                        cached_location['coordinates']['lat'],
                                        cached_location['coordinates']['lon']
                                    ),
                                    mode='walking'
                                )
                                
                                if route:
                                    # Generate humanized directions with Gemini
                                    if self.gemini:
                                        humanized = self.gemini.humanize_directions(route)
                                    else:
                                        humanized = f"Walk {route['total_distance']['text']}, takes about {route['total_duration']['text']}"
                                    
                                    # Store in cached directions
                                    cached_location['cached_directions'][start_point['name']] = {
                                        'distance': route['total_distance']['text'],
                                        'duration': route['total_duration']['text'],
                                        'directions': humanized
                                    }
                                    
                                    directions_count += 1
                                    
                            except Exception as e:
                                logger.warning(f"Failed to generate directions from {start_point['name']}: {str(e)}")
                    
                    # Add to cached locations list
                    cached_locations.append(cached_location)
                    
                    # Update popular_locations collection in MongoDB
                    await self.mongo.update_popular_location({
                        'location_id': location_id,
                        'location_name': cached_location['location_name'],
                        'location_type': cached_location['location_type'],
                        'coordinates': cached_location['coordinates'],
                        'description': cached_location['description'],
                        'search_count': cached_location['search_count'],
                        'search_count_week': 0,  # TODO: Calculate from analytics
                        'search_count_month': 0,  # TODO: Calculate from analytics
                        'last_searched': datetime.utcnow(),
                        'cached_directions': cached_location['cached_directions'],
                        'is_trending': i <= 10,  # Top 10 are "trending"
                        'rank': i
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing location {pop_loc.get('location_name')}: {str(e)}")
                    continue
            
            # Step 4: Save to JSON file for offline download
            logger.info("Step 4: Saving cache file...")
            
            cache_data = {
                'generated_at': datetime.utcnow().isoformat(),
                'version': '1.0',
                'total_locations': len(cached_locations),
                'locations': cached_locations
            }
            
            # Generate filename with timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            cache_filename = f"popular_places_{timestamp}.json"
            cache_filepath = self.cache_dir / cache_filename
            
            # Also save as "latest" for easy access
            latest_filepath = self.cache_dir / "popular_places_latest.json"
            
            # Write cache files
            with open(cache_filepath, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            with open(latest_filepath, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Cache generated successfully!")
            logger.info(f"   Locations cached: {len(cached_locations)}")
            logger.info(f"   Directions generated: {directions_count}")
            logger.info(f"   Cache file: {cache_filename}")
            
            return {
                'success': True,
                'locations_cached': len(cached_locations),
                'directions_generated': directions_count,
                'cache_file': str(cache_filepath),
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating offline cache: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_latest_cache(self) -> Optional[Dict]:
        """
        Get the latest generated cache data.
        
        Returns:
            Cache data dictionary or None if not found
        
        Example:
            >>> cache_data = await cache.get_latest_cache()
            >>> if cache_data:
            ...     print(f"Cache has {cache_data['total_locations']} locations")
        """
        try:
            latest_file = self.cache_dir / "popular_places_latest.json"
            
            if not latest_file.exists():
                logger.warning("No cache file found")
                return None
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            logger.info(f"Loaded cache with {cache_data.get('total_locations', 0)} locations")
            
            return cache_data
            
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return None
    
    def get_cache_info(self) -> Dict:
        """
        Get information about available cache files.
        
        Returns:
            Dictionary with cache information:
            {
                'has_cache': bool,
                'latest_generated': datetime,
                'total_locations': int,
                'cache_size_mb': float,
                'cache_files': List[str]
            }
        """
        try:
            latest_file = self.cache_dir / "popular_places_latest.json"
            
            if not latest_file.exists():
                return {
                    'has_cache': False,
                    'message': 'No cache generated yet'
                }
            
            # Get file size
            file_size_bytes = latest_file.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Load cache to get details
            with open(latest_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # List all cache files
            cache_files = [f.name for f in self.cache_dir.glob("popular_places_*.json")]
            
            return {
                'has_cache': True,
                'latest_generated': cache_data.get('generated_at'),
                'total_locations': cache_data.get('total_locations', 0),
                'cache_size_mb': round(file_size_mb, 2),
                'cache_files': cache_files
            }
            
        except Exception as e:
            logger.error(f"Error getting cache info: {str(e)}")
            return {
                'has_cache': False,
                'error': str(e)
            }


# Testing and usage example
if __name__ == "__main__":
    import asyncio
    
    async def test_cache():
        """Test Cache Service"""
        print("=" * 70)
        print("CACHE SERVICE TEST")
        print("=" * 70)
        
        try:
            # Initialize service
            cache = CacheService()
            print("✅ Service initialized\n")
            
            # Connect all services
            connected = await cache.connect_all()
            if not connected:
                print("❌ Failed to connect services")
                return
            
            print("✅ All services connected\n")
            
            # Test 1: Check existing cache
            print("Test 1: Cache Info")
            print("-" * 70)
            
            info = cache.get_cache_info()
            print(f"Has cache: {info.get('has_cache')}")
            if info.get('has_cache'):
                print(f"Last generated: {info.get('latest_generated')}")
                print(f"Locations: {info.get('total_locations')}")
                print(f"Size: {info.get('cache_size_mb')} MB")
            
            # Test 2: Generate new cache
            print("\nTest 2: Generate Cache")
            print("-" * 70)
            
            # Define common starting points (example)
            starting_points = [
                {'name': 'Main Gate', 'lat': 6.010317, 'lon': 10.258816},
                {'name': 'Cafeteria', 'lat': 6.010500, 'lon': 10.259000}
            ]
            
            result = await cache.generate_offline_cache(
                top_n=10,  # Just top 10 for testing
                common_starting_points=starting_points
            )
            
            print(f"Generation success: {result.get('success')}")
            print(f"Locations cached: {result.get('locations_cached', 0)}")
            print(f"Directions generated: {result.get('directions_generated', 0)}")
            
            # Test 3: Load latest cache
            print("\nTest 3: Load Latest Cache")
            print("-" * 70)
            
            cache_data = await cache.get_latest_cache()
            if cache_data:
                print(f"Cache version: {cache_data.get('version')}")
                print(f"Total locations: {cache_data.get('total_locations')}")
                print(f"Generated at: {cache_data.get('generated_at')}")
                
                # Show first location
                if cache_data.get('locations'):
                    first = cache_data['locations'][0]
                    print(f"\nFirst location: {first.get('location_name')}")
                    print(f"  Rank: {first.get('rank')}")
                    print(f"  Search count: {first.get('search_count')}")
            
            # Disconnect
            await cache.disconnect_all()
            
            print("\n" + "=" * 70)
            print("✅ All tests completed!")
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Run async test
    asyncio.run(test_cache())