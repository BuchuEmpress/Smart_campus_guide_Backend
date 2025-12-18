"""
Qdrant Service - Fully Fixed Asynchronous Version
Now works with:
- search_points()  <-- correct async search
- scroll() for ID-exact lookup
- robust logging, errors, lazy model loading
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

import asyncio
from qdrant_client import QdrantClient
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

        # Sync client (wrapped in async methods)
        self.client = QdrantClient(
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
        self.client.search_points()
        """
        try:
            # Use Gemini for embeddings (remote call)
            vector = await self.gemini.embed_text(query)

            # Run sync query in thread pool
            results = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                query_filter=filter_params,
                with_payload=True
            )
            
            # Extract points from response
            results = results.points if hasattr(results, 'points') else results

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
            results, _ = await asyncio.to_thread(
                self.client.scroll,
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
                await asyncio.to_thread(
                    self.client.create_payload_index,
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
        exists = await asyncio.to_thread(self.client.collection_exists, self.collection_name)

        if exists and force_recreate:
            await asyncio.to_thread(self.client.delete_collection, self.collection_name)

        if not exists or force_recreate:
            await asyncio.to_thread(
                self.client.create_collection,
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
        """Upload vectorized locations to Qdrant."""
        try:
            points = []
            # Batch encode for efficiency
            texts = [f"{item['name']} {item.get('description', '')}" for item in items]
            vectors = await self.gemini.embed_batch(texts)

            for idx, (item, vector) in enumerate(zip(items, vectors)):
                points.append(
                    PointStruct(
                        id=item.get("id", idx),
                        vector=vector,
                        payload=item
                    )
                )

            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection_name,
                points=points
            )

        except Exception as e:
            print(f"Error uploading points: {e}")

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
                results, _ = await asyncio.to_thread(
                    self.client.scroll,
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
                    results, _ = await asyncio.to_thread(
                        self.client.scroll,
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
                    await asyncio.to_thread(
                        self.client.create_payload_index,
                        collection_name=self.collection_name,
                        field_name="name",
                        field_schema="text"
                    )
                    await asyncio.to_thread(
                        self.client.create_payload_index,
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
            await asyncio.to_thread(
                self.client.create_payload_index,
                collection_name=self.collection_name,
                field_name="name",
                field_schema="text"
            )
            await asyncio.to_thread(
                self.client.create_payload_index,
                collection_name=self.collection_name,
                field_name="description",
                field_schema="text"
            )
            await asyncio.to_thread(
                self.client.create_payload_index,
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
            return await asyncio.to_thread(self.client.get_collection, self.collection_name)
        except Exception as e:
            return {"error": str(e)}

    # -----------------------------------------------------
    # OPTIONAL: PRINT RESULTS
    # -----------------------------------------------------
    def print_search_results(self, results: List[Dict]):
        for r in results:
            print(
                f"[{r.get('score'):.3f}] {r.get('name', 'Unknown')} "
                f"({r.get('type', 'N/A')})"
            )
