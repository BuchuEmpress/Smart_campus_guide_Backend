"""
Cache Service Module - FIXED IMPORT

CRITICAL FIX: Line 11 changed from MapsService to OSMRoutingService
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from services.mongodb_service import MongoDBService
from services.analytics_service import AnalyticsService
from services.qdrant_service import QdrantService
from services.osm_routing_service import OSMRoutingService # ✅ FIXED!
from services.gemini_service import GeminiService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheService:
    """Service for generating and managing offline cache."""
    
    def __init__(
        self,
        mongo_service: Optional[MongoDBService] = None,
        analytics_service: Optional[AnalyticsService] = None,
        qdrant_service: Optional[QdrantService] = None
    ):
        """Initialize Cache Service."""
        self.mongo = mongo_service or MongoDBService()
        self.analytics = analytics_service or AnalyticsService(self.mongo)
        self.qdrant = qdrant_service or QdrantService()
        
        # Initialize OSM and Gemini when needed
        self.osm: Optional[OSMRoutingService] = None
        self.gemini: Optional[GeminiService] = None
        
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Cache service initialized")
    
    def load_cache(self) -> Optional[Dict]:
        """Load the latest cache file."""
        try:
            cache_file = self.cache_dir / "popular_places_latest.json"
            
            if not cache_file.exists():
                logger.warning("No cache file found")
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            logger.info(f"Loaded cache with {cache_data.get('total_locations', 0)} locations")
            return cache_data
            
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return None
    
    def generate_cache(self, popular_locations: List[Dict]) -> bool:
        """
        Generate offline cache file.
        
        Args:
            popular_locations: List of popular locations from analytics
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Generating cache for {len(popular_locations)} locations")
            
            # Build cache data
            cache_data = {
                'metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'version': '1.0',
                    'total_locations': len(popular_locations)
                },
                'locations': popular_locations
            }
            
            # Save cache files
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            cache_filename = f"popular_places_{timestamp}.json"
            cache_filepath = self.cache_dir / cache_filename
            latest_filepath = self.cache_dir / "popular_places_latest.json"
            
            # Write files
            with open(cache_filepath, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            with open(latest_filepath, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Cache generated: {cache_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating cache: {str(e)}")
            return False


if __name__ == "__main__":
    print("Cache Service - Import Fixed ✅")