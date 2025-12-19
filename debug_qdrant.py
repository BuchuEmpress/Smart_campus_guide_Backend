import asyncio
from services.qdrant_service import QdrantService

async def test():
    q = QdrantService('topics')
    info = await q.get_collection_info()
    with open('debug.txt', 'w') as f:
        f.write(f'Points count: {getattr(info, "points_count", "N/A")}\n')
        if hasattr(info, 'config'):
            f.write(f'Vector size: {info.config.params.vectors.size}\n')
        else:
            f.write('No config\n')
    
    results = await q.search('database', 3)
    with open('debug.txt', 'a') as f:
        f.write(f'Qdrant search returned {len(results)} results\n')
        for r in results:
            f.write(f'- {r.get("title", "No title")}\n')

asyncio.run(test())