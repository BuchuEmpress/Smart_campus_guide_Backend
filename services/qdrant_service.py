"""
Qdrant Service - Vector Database Management for Smart Campus Guide
Asynchronous + robust version with full compatibility for unit tests.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# IMPORTANT — matches test patch path
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

load_dotenv()


class QdrantService:
    """Handles all Qdrant vector database operations (async + robust)."""

    def __init__(self, load_model: bool = False):
        # Load env values
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_key = os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url or not self.qdrant_key:
            raise ValueError(
                "Missing QDRANT_URL or QDRANT_API_KEY in environment variables."
            )

        # Async client = test-compatible
        self.client = AsyncQdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_key
        )

        self.model = None
        self.collection_name = "campus_locations"
        self.vector_size = 384

        if load_model:
            self._load_model()

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _ensure_model_loaded(self):
        """Lazy load for memory-safe startup."""
        if self.model is None:
            self._load_model()

    # =============================================================================
    # SEARCH (ASYNC)
    # =============================================================================
    async def search(
        self,
        query: str,
        limit: int = 5,
        filter_params: Optional[Filter] = None
    ) -> List[Dict]:
        """Perform semantic search in Qdrant."""
        self._ensure_model_loaded()

        vector = self.model.encode(query).tolist()

        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=filter_params,
            limit=limit,
            with_payload=True
        )

        formatted = []
        for hit in results:
            payload = dict(hit.payload)
            payload["score"] = float(hit.score)
            formatted.append(payload)

        return formatted

    # =============================================================================
    # CREATE COLLECTION (SAFE)
    # =============================================================================
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

    # =============================================================================
    # UPSERT POINTS
    # =============================================================================
    async def upload_points(self, items: List[Dict]):
        """Upload vectorized locations to Qdrant."""
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

    # =============================================================================
    # COLLECTION INFO
    # =============================================================================
    async def get_collection_info(self):
        """Returns Qdrant info or readable error."""
        try:
            return await self.client.get_collection(self.collection_name)
        except Exception as e:
            return {"error": str(e)}

    # =============================================================================
    # PRINT SEARCH RESULTS (Human friendly)
    # =============================================================================
    def print_search_results(self, results: List[Dict]):
        for r in results:
            print(
                f"[{r.get('score'):.3f}] {r.get('name', 'Unknown')} "
                f"({r.get('type', 'N/A')}) - {r.get('department', '')}"
            )


# =============================================================================
# CLI (Optional)
# =============================================================================

async def main():
    print("Qdrant Service CLI - Not Implemented")
    pass


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
