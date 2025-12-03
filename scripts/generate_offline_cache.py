"""
Generate Offline Cache for Smart Campus Guide

Creates a cache file with popular locations for offline access.
Run this daily or weekly to keep cache fresh.

USAGE:
    python scripts/generate_offline_cache.py

OUTPUT:
    data/cache/popular_places_latest.json
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.analytics_service import AnalyticsService
from services.qdrant_service import QdrantService


def generate_cache(min_searches: int = 5, limit: int = 100):
    """
    Generate offline cache with popular locations.
    
    Args:
        min_searches: Minimum searches for a location to be cached
        limit: Maximum locations to cache
    """
    print("\n" + "="*80)
    print("🚀 GENERATING OFFLINE CACHE")
    print("="*80 + "\n")
    
    try:
        # Initialize services
        print("📊 Initializing analytics service...")
        analytics = AnalyticsService()
        
        print("🗄️  Initializing Qdrant service...")
        # ✅ FIX: Changed load_model to True because qdrant.search() requires the embedding model.
        qdrant = QdrantService(load_model=True) 
        
        # Get popular locations from analytics
        print(f"\n🔍 Finding popular locations (min {min_searches} searches)...")
        popular = analytics.get_popular_locations(
            min_searches=min_searches,
            limit=limit
        )
        
        print(f"✅ Found {len(popular)} popular locations")
        
        # Get full details from Qdrant
        print("\n📍 Fetching location details from Qdrant...")
        cached_locations = []
        
        for i, pop in enumerate(popular, 1):
            try:
                # Search Qdrant for this location
                results = qdrant.search(pop['location_name'], limit=1)
                
                if results:
                    location = results[0]
                    
                    # Build cache entry
                    cache_entry = {
                        'id': location.get('id', f'cache_{i}'),
                        'name': location.get('name', pop['location_name']),
                        'description': location.get('description', ''),
                        'type': location.get('type', 'location'),
                        'coordinates': {
                            'lat': location.get('latitude'),
                            'lng': location.get('longitude')
                        },
                        'search_count': pop['count'],
                        'popularity_rank': i
                    }
                    
                    cached_locations.append(cache_entry)
                    
                    if i % 10 == 0:
                        print(f"   Processed {i}/{len(popular)} locations...")
                
            except Exception as e:
                print(f"   ⚠️  Error processing {pop['location_name']}: {str(e)}")
                continue
        
        print(f"\n✅ Successfully cached {len(cached_locations)} locations")
        
        # Build cache data
        cache_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=7)).isoformat(),
                'version': '1.0',
                'total_locations': len(cached_locations),
                'min_searches': min_searches
            },
            'locations': cached_locations
        }
        
        # Create cache directory
        cache_dir = Path('data/cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cache file
        cache_file = cache_dir / 'popular_places_latest.json'
        
        print(f"\n💾 Saving cache to {cache_file}...")
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        # Get file size
        file_size = os.path.getsize(cache_file) / 1024  # KB
        
        print(f"✅ Cache saved successfully!")
        print(f"   File size: {file_size:.2f} KB")
        print(f"   Locations: {len(cached_locations)}")
        print(f"   Valid until: {cache_data['metadata']['expires_at']}")
        
        # Print top 10 popular locations
        print(f"\n📊 Top 10 Most Popular Locations:")
        print("-" * 80)
        for i, loc in enumerate(cached_locations[:10], 1):
            print(f"   {i}. {loc['name']}")
            print(f"      Searches: {loc['search_count']}")
            print(f"      Type: {loc['type']}")
        
        print("\n" + "="*80)
        print("✅ CACHE GENERATION COMPLETE!")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Cache generation failed")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate offline cache')
    parser.add_argument('--min-searches', type=int, default=5,
                        help='Minimum searches for location to be cached (default: 5)')
    parser.add_argument('--limit', type=int, default=100,
                        help='Maximum locations to cache (default: 100)')
    
    args = parser.parse_args()
    
    success = generate_cache(
        min_searches=args.min_searches,
        limit=args.limit
    )
    
    sys.exit(0 if success else 1)