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

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
)

load_dotenv()


class QdrantService:
    """Handles all Qdrant vector database operations (async + robust)."""

    def __init__(self, load_model: bool = False):
        # Load env values
        self.qdrant_url = os.getenv("QDRANT_HOST")
        self.qdrant_key = os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url or not self.qdrant_key:
            raise ValueError(
                "Missing QDRANT_HOST or QDRANT_API_KEY in environment variables."
            )

        # Async client (correct for FastAPI)
        self.client = AsyncQdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_key
        )

        self.model = None
        self.collection_name = "campus_locations"
        self.vector_size = 384

        if load_model:
            self._load_model()

    # -----------------------------------------------------
    # MODEL LOADING
    # -----------------------------------------------------
    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _ensure_model_loaded(self):
        if self.model is None:
            self._load_model()

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
            self._ensure_model_loaded()

            vector = self.model.encode(query).tolist()

            results = await self.client.search_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                filter=filter_params,
                with_payload=True
            )

            formatted = []
            for hit in results:
                payload = dict(hit.payload)
                payload["score"] = float(hit.score)
                formatted.append(payload)

            return formatted

        except Exception as e:
            print(f"❌ Qdrant search error: {e}")
            return []

    # -----------------------------------------------------
    # LOOKUP BY LOCATION ID (EXACT MATCH)
    # -----------------------------------------------------
    async def get_by_id(self, location_id: str) -> Optional[Dict]:
        """
        Qdrant cannot vector-search for IDs.
        So we use scroll() to find exact match by payload.
        """
        try:
            scroll_result, _ = await self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=1000  # safe campus size
            )

            for point in scroll_result:
                payload = point.payload
                if str(payload.get("id")) == str(location_id):
                    return payload

            return None

        except Exception as e:
            print(f"❌ Qdrant get_by_id error: {e}")
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
        """Upload vectorized locations to Qdrant."""
        try:
            self._ensure_model_loaded()

            points = []
            for idx, item in enumerate(items):
                vector = self.model.encode(item["name"]).tolist()

                points.append(
                    PointStruct(
                        id=item.get("id", idx),
                        vector=vector,
                        payload=item
                    )
                )

            await self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        except Exception as e:
            print(f"❌ Error uploading points: {e}")

    # -----------------------------------------------------
    # COLLECTION INFO
    # -----------------------------------------------------
    async def get_collection_info(self):
        """Returns Qdrant info or readable error."""
        try:
            return await self.client.get_collection(self.collection_name)
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
