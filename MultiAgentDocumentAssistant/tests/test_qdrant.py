import asyncio

from qdrant_client import AsyncQdrantClient, models as m

from research.memory import Memory, QdrantSemantic
from research.rag import PaperIndex


class Embeddings:
    size = 3

    def encode(self, texts):
        return [[float('alpha' in t), float('beta' in t), .1] for t in texts]

    def lexical(self, texts):
        return [m.SparseVector(indices=[1 if 'alpha' in t else 2], values=[1.]) for t in texts]


class Reranker:
    def rank(self, query, chunks, limit):
        return sorted(chunks, key=lambda c: query in c['text'], reverse=True)[:limit]


def test_qdrant_real_hybrid_and_memory_isolation():
    async def scenario():
        client = AsyncQdrantClient(':memory:')
        try:
            index = PaperIndex(client, Embeddings(), Reranker())
            await index.setup()
            pages = [{'doc_id': 'd', 'page': 1, 'title': 'P', 'text': 'Introduction\nalpha evidence'}]
            await index.ingest(pages, 'a')
            await index.ingest(pages, 'a')
            await index.ingest(pages, 'b')
            assert (await client.count(index.collection)).count == 2
            result = await index.search('alpha', 'a')
            assert len(result) == 1 and result[0]['namespace'] == 'a'
            assert await index.search('alpha', 'unknown') == []
            store = QdrantSemantic(client, Embeddings())
            await store.setup()
            await store.put(Memory('id', 'a', 'semantic', 'alpha', .9))
            await store.put(Memory('id', 'b', 'semantic', 'beta', .9))
            assert (await store.search('a', 'alpha', 5))[0].text == 'alpha'
            await store.delete('a', 'id')
            assert await store.list('a') == [] and len(await store.list('b')) == 1
        finally:
            await client.close()
    asyncio.run(scenario())
