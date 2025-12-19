import asyncio
from services.qdrant_service import QdrantService

async def test():
    q = QdrantService('topics')
    info = await q.get_collection_info()
    print('Vector size:', info.config.params.vectors.size)
    print('Points count:', info.points_count)

asyncio.run(test())