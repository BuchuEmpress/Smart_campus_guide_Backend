"""
Qdrant Service - Vector Database Management for Smart Campus Guide

USAGE:
------
1. Upload embeddings to Qdrant:
   python services/qdrant_service.py upload

2. Search for locations:
   python services/qdrant_service.py search "main gate"
   python services/qdrant_service.py search "computer lab"

3. Get collection info:
   python services/qdrant_service.py info

REQUIREMENTS:
-------------
- config/qdrant_config.json with your Qdrant Cloud credentials
- data/locations/campus_locations_embeddings.json with embedded location data
- qdrant-client installed: pip install qdrant-client

TEAM NOTES:
-----------
- The script automatically creates/recreates the 'campus_locations' collection
- Embeddings are 384-dimensional vectors (from all-MiniLM-L6-v2 model)
- Uses Cosine similarity for semantic search
- All metadata (type, lat/long, description) is stored in Qdrant payload
- Model only loads when needed (for search), not during upload (saves memory!)
"""

import json
import os
import sys
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue


class QdrantService:
    """Handles all Qdrant vector database operations"""
    
    def __init__(self, config_path: str = 'config/qdrant_config.json', load_model: bool = False):
        """
        Initialize Qdrant service with cloud connection
        
        Args:
            config_path: Path to JSON config file with 'url' and 'api_key'
            load_model: If True, loads SentenceTransformer model (needed for search)
                       If False, skips model loading (for upload only - saves memory!)
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is missing required fields
        """
        print(f"[INFO] Loading Qdrant config from {config_path}...")
        
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Please create it with:\n"
                f'{{\n  "url": "https://your-cluster.qdrant.io",\n  '
                f'"api_key": "your-api-key"\n}}'
            )
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate config
        if 'url' not in config or 'api_key' not in config:
            raise ValueError("Config must contain 'url' and 'api_key' fields")
        
        # Initialize Qdrant client
        print(f"[INFO] Connecting to Qdrant Cloud: {config['url']}")
        try:
            self.client = QdrantClient(
                url=config['url'],
                api_key=config['api_key']
            )
            print("[SUCCESS] Connected to Qdrant Cloud!")
        except Exception as e:
            print(f"[ERROR] Failed to connect to Qdrant: {str(e)}")
            raise
        
        # Only load model if needed (for search operations)
        # This saves memory during upload operations
        self.model = None
        if load_model:
            print("[INFO] Loading sentence transformer model...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[SUCCESS] Model loaded!")
        else:
            print("[INFO] Skipping model load (not needed for upload - saves memory!)")
        
        self.collection_name = "campus_locations"
        self.vector_size = 384  # Dimension for all-MiniLM-L6-v2
    
    def _ensure_model_loaded(self):
        """
        Lazy load the model only when needed for search
        This prevents memory errors during upload operations
        """
        if self.model is None:
            print("[INFO] Loading sentence transformer model for search...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[SUCCESS] Model loaded!")
    
    def create_collection(self, force_recreate: bool = True):
        """
        Create or recreate the campus_locations collection
        
        Args:
            force_recreate: If True, deletes existing collection before creating
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                if force_recreate:
                    print(f"[INFO] Deleting existing collection '{self.collection_name}'...")
                    self.client.delete_collection(self.collection_name)
                    print("[SUCCESS] Existing collection deleted")
                else:
                    print(f"[INFO] Collection '{self.collection_name}' already exists, skipping creation")
                    return
            
            # Create new collection with payload index for filtering
            from qdrant_client.models import PayloadSchemaType
            
            print(f"[INFO] Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE  # Best for semantic similarity
                )
            )
            
            # Create payload index for 'type' field to enable filtering
            print(f"[INFO] Creating payload index for 'type' field...")
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="type",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            print(f"[SUCCESS] Collection '{self.collection_name}' created with {self.vector_size}D vectors (Cosine similarity)")
            print(f"[SUCCESS] Payload index created for 'type' field - filtering enabled!")
            
        except Exception as e:
            print(f"[ERROR] Failed to create collection: {str(e)}")
            raise
    
    def upload_embeddings(self, embeddings_file: str = 'data/locations/campus_locations_embeddings.json'):
        """
        Load embeddings from JSON and upload to Qdrant
        NO MODEL LOADING - uses pre-computed embeddings from file
        
        Args:
            embeddings_file: Path to JSON file with location embeddings
        
        Raises:
            FileNotFoundError: If embeddings file doesn't exist
            ValueError: If embeddings are missing from data
        """
        print(f"\n[INFO] Loading embeddings from {embeddings_file}...")
        
        # Check file exists
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_file}\n"
                f"Please run embedding_service.py first to generate embeddings"
            )
        
        # Load data
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            locations = json.load(f)
        
        print(f"[INFO] Found {len(locations)} locations in file")
        
        # Validate data has embeddings
        if not locations:
            raise ValueError("Embeddings file is empty!")
        
        if 'embedding' not in locations[0]:
            raise ValueError(
                "No 'embedding' field found in data! "
                "Please run embedding_service.py first"
            )
        
        # Prepare points for upload
        print("[INFO] Preparing data for Qdrant upload...")
        points = []
        
        for i, location in enumerate(locations):
            # Build payload with all metadata (everything except embedding)
            payload = {
                "id": location.get('id', f'loc_{i}'),  # Fallback ID if missing
                "type": location.get('type', 'unknown'),
                "latitude": location.get('latitude') or location.get('lat'),  # Handle both formats
                "longitude": location.get('longitude') or location.get('lon'),  # Handle both formats
                "description": location.get('description', '')
            }
            
            # Add optional fields if present
            if 'timestamp' in location:
                payload['timestamp'] = location['timestamp']
            if 'embedding_text' in location:
                payload['embedding_text'] = location['embedding_text']
            if 'name' in location:
                payload['name'] = location['name']
            
            # Handle OSM tags format
            if 'tags' in location:
                payload['tags'] = location['tags']
                if 'name' in location['tags'] and 'name' not in payload:
                    payload['name'] = location['tags']['name']
            
            # Create Qdrant point
            point = PointStruct(
                id=i,  # Sequential ID for Qdrant (different from location ID)
                vector=location['embedding'],
                payload=payload
            )
            points.append(point)
            
            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == len(locations):
                print(f"[PROGRESS] Prepared {i + 1}/{len(locations)} points...")
        
        # Upload to Qdrant in batches (more memory efficient)
        print(f"[INFO] Uploading {len(points)} points to Qdrant in batches...")
        batch_size = 100
        try:
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                print(f"[PROGRESS] Uploaded batch {i // batch_size + 1}/{(len(points) + batch_size - 1) // batch_size}")
            
            print(f"[SUCCESS] Successfully uploaded {len(points)} locations to Qdrant!")
        except Exception as e:
            print(f"[ERROR] Upload failed: {str(e)}")
            raise
    
    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: Optional[float] = None,
        filter_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform semantic search for locations
        
        Args:
            query: Natural language search query (e.g., "main gate", "computer lab")
            limit: Number of results to return
            score_threshold: Minimum similarity score (0-1), filters out low matches
            filter_type: Optional filter by location type (e.g., 'landmark', 'building')
        
        Returns:
            List of dicts with keys: score, id, type, description, latitude, longitude
        """
        print(f"\n[SEARCH] Query: '{query}'")
        
        # Ensure model is loaded for search
        self._ensure_model_loaded()
        
        # Generate query embedding
        print("[INFO] Generating query embedding...")
        query_vector = self.model.encode(query).tolist()
        
        # Build filter if type specified
        query_filter = None
        if filter_type:
            print(f"[INFO] Filtering by type: {filter_type}")
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=filter_type)
                    )
                ]
            )
        
        # Perform search
        print(f"[INFO] Searching for top {limit} matches...")
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            ).points
            
            print(f"[SUCCESS] Found {len(results)} results\n")
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted = {
                    'rank': i,
                    'score': result.score,
                    'id': result.payload.get('id'),
                    'type': result.payload.get('type'),
                    'name': result.payload.get('name', ''),
                    'description': result.payload.get('description', ''),
                    'latitude': result.payload.get('latitude'),
                    'longitude': result.payload.get('longitude')
                }
                formatted_results.append(formatted)
            
            return formatted_results
            
        except Exception as e:
            print(f"[ERROR] Search failed: {str(e)}")
            raise
    
    def print_search_results(self, results: List[Dict]):
        """Pretty print search results"""
        if not results:
            print("No results found.")
            return
        
        print("=" * 80)
        print(f"SEARCH RESULTS ({len(results)} matches)")
        print("=" * 80)
        
        for result in results:
            # Handle missing type field
            loc_type = result.get('type', 'location')
            if loc_type:
                loc_type = loc_type.upper()
            else:
                loc_type = 'LOCATION'
            
            print(f"\n{result['rank']}. [{loc_type}] Score: {result['score']:.4f}")
            print(f"   ID: {result['id']}")
            if result.get('name'):
                print(f"   Name: {result['name']}")
            print(f"   Location: ({result['latitude']}, {result['longitude']})")
            desc = result.get('description', '')
            if desc:
                print(f"   Description: {desc[:100]}...")
                if len(desc) > 100:
                    print(f"                {desc[100:200]}...")
        
        print("\n" + "=" * 80)
    
    def get_collection_info(self):
        """Display collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            
            print("\n" + "=" * 80)
            print(f"COLLECTION INFO: {self.collection_name}")
            print("=" * 80)
            print(f"Points count:    {info.points_count}")
            print(f"Vector size:     {info.config.params.vectors.size}")
            print(f"Distance metric: {info.config.params.vectors.distance}")
            print(f"Status:          {info.status}")
            print("=" * 80 + "\n")
            
            return info
        except Exception as e:
            print(f"[ERROR] Failed to get collection info: {str(e)}")
            return None


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for Qdrant operations"""
    
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nERROR: No command provided")
        print("\nAvailable commands:")
        print("  upload              - Create collection and upload embeddings")
        print("  search <query>      - Search for locations")
        print("  info                - Show collection information")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    try:
        if command == 'upload':
            print("\n" + "=" * 80)
            print("UPLOADING EMBEDDINGS TO QDRANT")
            print("=" * 80)
            
            # Initialize service WITHOUT loading model (saves memory for upload)
            service = QdrantService(load_model=False)
            
            # Create collection (force recreate to ensure clean state)
            service.create_collection(force_recreate=True)
            
            # Upload embeddings
            service.upload_embeddings()
            
            # Show final stats
            service.get_collection_info()
            
            print("\n✅ Upload complete! You can now search with:")
            print('   python services/qdrant_service.py search "your query"')
        
        elif command == 'search':
            if len(sys.argv) < 3:
                print("ERROR: Search query required")
                print('Usage: python services/qdrant_service.py search "your query"')
                sys.exit(1)
            
            query = ' '.join(sys.argv[2:])
            
            # Initialize service WITH model (needed for search)
            service = QdrantService(load_model=True)
            
            # Perform search
            results = service.search(query, limit=5)
            
            # Print results
            service.print_search_results(results)
        
        elif command == 'info':
            # Initialize service without model (just checking info)
            service = QdrantService(load_model=False)
            service.get_collection_info()
        
        else:
            print(f"ERROR: Unknown command '{command}'")
            print("\nAvailable commands: upload, search, info")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ OPERATION FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()