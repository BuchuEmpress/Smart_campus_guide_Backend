"""
Qdrant Service - Fully Asynchronous Version
Now works with:
- search()  <-- correct async search
- scroll() for ID-exact lookup
- robust logging, errors, lazy model loading
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from services.gemini_service import GeminiService

load_dotenv()


class QdrantService:
    """Handles all Qdrant vector database operations (async + robust)."""

    # Class-level singleton for the heavy model
    _model = None

    def __init__(self, collection_name: str = "campus_locations"):
        # Load env values
        self.qdrant_url = os.getenv("QDRANT_HOST")
        self.qdrant_key = os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url or not self.qdrant_key:
            raise ValueError(
                "Missing QDRANT_HOST or QDRANT_API_KEY in environment variables."
            )

        # Async client
        self.client = AsyncQdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_key
        )

        self.collection_name = collection_name
        self.vector_size = 768  # GEMINI EMBEDDINGS SIZE
        self.gemini = GeminiService()

    # Removed local model loading to save 500MB+ RAM

    # -----------------------------------------------------
    # SEMANTIC SEARCH (FIXED)
    # -----------------------------------------------------
    async def search(
        self,
        query: str,
        limit: int = 5,
        filter_params: Optional[Filter] = None
    ) -> List[Dict]:
        """
        Perform semantic vector search using correct async API:
        self.client.query_points()
        """
        try:
            # Use Gemini for embeddings (remote call)
            vector = await self.gemini.embed_text(query)

            # Run async search
            results = await self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                query_filter=filter_params,
                with_payload=True
            )
            
            # query_points returns a response with .points
            results = results.points

            formatted = []
            for hit in results:
                payload = dict(hit.payload)
                payload["score"] = float(hit.score)
                formatted.append(payload)

            return formatted

        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []

    # -----------------------------------------------------
    # LOOKUP BY LOCATION ID (EXACT MATCH)
    # -----------------------------------------------------
    async def get_by_id(self, location_id: str) -> Optional[Dict]:
        """
        Get exact location by ID using server-side filtering.
        """
        try:
            # Create filter for exact ID match
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="id",
                        match=MatchValue(value=location_id)
                    )
                ]
            )
            
            # Use scroll with filter
            results, _ = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=1,
                with_payload=True
            )

            if results:
                return results[0].payload
            
            return None

        except Exception as e:
            # Self-healing: Create index if missing
            if "Index required" in str(e):
                print("Index missing for 'id'. Creating index now...")
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="id",
                    field_schema="keyword"
                )
                # Retry once
                return await self.get_by_id(location_id)

            print(f"Qdrant get_by_id error: {e}")
            return None

    # -----------------------------------------------------
    # CREATE COLLECTION
    # -----------------------------------------------------
    async def create_collection(self, force_recreate: bool = True):
        """Creates collection if missing."""
        exists = await self.client.collection_exists(self.collection_name)

        if exists and force_recreate:
            await self.client.delete_collection(self.collection_name)

        if not exists or force_recreate:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            return True

        return False

    # -----------------------------------------------------
    # UPSERT POINTS
    # -----------------------------------------------------
    async def upload_points(self, items: List[Dict]):
        """Upload vectorized locations to Qdrant.

        This method respects the Gemini batch limit (<=100 requests per batch)
        and uploads points to Qdrant in chunks. It raises on failure so callers
        can detect and react to upload problems.
        """
        try:
            batch_size = 100
            total = len(items)
            if total == 0:
                return

            for start in range(0, total, batch_size):
                batch = items[start:start + batch_size]

                # Prepare texts for embedding
                texts = [f"{it.get('name','')} {it.get('description','')}" for it in batch]

                # Request embeddings for this batch
                vectors = await self.gemini.embed_batch(texts)
                if not vectors or len(vectors) != len(batch):
                    raise RuntimeError(f"Embedding batch failed or returned unexpected size: expected {len(batch)}, got {len(vectors) if vectors is not None else 0}")

                points = []
                for idx, (item, vector) in enumerate(zip(batch, vectors)):
                    points.append(
                        PointStruct(
                            id=item.get("id", f"{start + idx}"),
                            vector=vector,
                            payload=item
                        )
                    )

                # Upsert this batch
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

                # small throttle to avoid API rate limits
                await asyncio.sleep(0.05)

        except Exception as e:
            # Log and re-raise so callers (sync) notice failures
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error uploading points: {e}")
            raise

    # -----------------------------------------------------
    # TEXT SEARCH (PAYLOAD LOOKUP)
    # -----------------------------------------------------
    async def search_by_text(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for text across multiple payload fields (name, description, id, etc.)
        This is an 'aggressive' search to find 'in-between the lines' as requested by user.
        """
        try:
            from qdrant_client.models import MatchText, MatchValue
            
            # Create a broad filter to check multiple fields
            text_filter = Filter(
                should=[
                    FieldCondition(key="name", match=MatchText(text=query)),
                    FieldCondition(key="description", match=MatchText(text=query)),
                    FieldCondition(key="id", match=MatchText(text=query)) # Search in IDs too
                ]
            )

            # Execution with auto-retry for indexing
            try:
                results, _ = await self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=text_filter,
                    limit=limit,
                    with_payload=True
                )
            except Exception as e:
                # Handle missing text indexes
                msg = str(e)
                if any(x in msg for x in ["Index required", "Standard index", "text"]):
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Indexing issue detected: {msg}. Retrying with fresh indexes...")
                    await self._create_text_indexes()
                    
                    # Retry once
                    results, _ = await self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=text_filter,
                        limit=limit,
                        with_payload=True
                    )
                else:
                    raise e

            formatted = []
            for hit in results:
                payload = dict(hit.payload)
                # Assign high score for direct text matches
                payload["score"] = 0.99
                formatted.append(payload)

            return formatted

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Aggressive text search failed: {e}")
            return []

        except Exception as e:
            # Self-healing: Create text indexes if missing
            if "Index required" in str(e):
                print("Text index missing. Creating indexes for 'name' and 'description'...")
                try:
                    await self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="name",
                        field_schema="text"
                    )
                    await self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="description",
                        field_schema="text"
                    )
                    # Retry once
                    return await self.search_by_text(query, limit)
                except Exception as idx_err:
                    print(f"Failed to create indexes: {idx_err}")
            
            print(f"Qdrant text search error: {e}")
            return []

    async def _create_text_indexes(self):
        """Creates text indexes for name and description."""
        try:
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="name",
                field_schema="text"
            )
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="description",
                field_schema="text"
            )
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="id",
                field_schema="text"
            )
        except Exception as e:
            print(f"Error creating indexes: {e}")

    # -----------------------------------------------------
    # COLLECTION INFO
    # -----------------------------------------------------
    async def get_collection_info(self):
        """Returns Qdrant info or readable error."""
        try:
            return await self.client.get_collection(self.collection_name)
        except Exception as e:
            return {"error": str(e)}

    async def sync_missing_topics(self):
        """
        Sync missing topics from MongoDB to Qdrant.
        Ensures all MongoDB topics exist in Qdrant with correct embeddings.
        """
        try:
            from services.topic_service import TopicService
            import uuid
            
            # Initialize TopicService
            topic_service = TopicService()
            
            # Get all topics from MongoDB
            all_topics = topic_service.list_topics(limit=10000)  # Large limit to get all
            print(f"Found {len(all_topics)} topics in MongoDB")
            
            if not all_topics:
                print("No topics to sync")
                return
            
            # Get existing topic IDs from Qdrant
            existing_ids = set()
            try:
                # Scroll through all points to get existing topic_ids
                offset = None
                while True:
                    results, next_offset = await self.client.scroll(
                        collection_name=self.collection_name,
                        offset=offset,
                        limit=1000,
                        with_payload=["topic_id"]
                    )
                    
                    for point in results:
                        if point.payload.get("topic_id"):
                            existing_ids.add(str(point.payload["topic_id"]))
                    
                    if next_offset is None:
                        break
                    offset = next_offset
                    
            except Exception as e:
                print(f"Error getting existing Qdrant topics: {e}")
                # If collection doesn't exist, recreate it
                await self.create_collection(force_recreate=True)
                existing_ids = set()
            
            print(f"Found {len(existing_ids)} topics already in Qdrant")
            
            # Find missing topics
            missing_topics = []
            for topic in all_topics:
                topic_id = str(topic.get('topic_id', ''))
                if topic_id and topic_id not in existing_ids:
                    missing_topics.append(topic)
            
            print(f"Found {len(missing_topics)} missing topics to sync")
            
            if not missing_topics:
                print("All topics are already synced")
                return
            
            # Prepare points for upload
            points = []
            for topic in missing_topics:
                # Create text for embedding
                text = f"{topic.get('title', '')}. {topic.get('description', '')}. Tags: {', '.join(topic.get('tags', []))}"
                
                # Use topic_id as payload id
                t_id = topic.get('topic_id')
                
                # Generate deterministic UUID from topic_id
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(t_id)))
                
                points.append({
                    "id": point_id,
                    "name": text,  # Used for embedding generation
                    "topic_id": t_id,
                    "title": topic.get('title'),
                    "department": topic.get('department'),
                    "option": topic.get('option'),
                    "year": topic.get('year'),
                    "status": topic.get('status')
                })
            
            # Upload missing topics
            print(f"Uploading {len(points)} missing topics to Qdrant...")
            await self.upload_points(points)
            print("✓ Sync completed!")
            
        except Exception as e:
            print(f"Error syncing missing topics: {e}")
            raise

    # -----------------------------------------------------
    # OPTIONAL: PRINT RESULTS
    # -----------------------------------------------------
    def print_search_results(self, results: List[Dict]):
        for r in results:
            print(
                f"[{r.get('score'):.3f}] {r.get('name', 'Unknown')} "
                f"({r.get('type', 'N/A')})"
            )