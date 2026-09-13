"""Three durable memory stores with provenance-preserving retention."""
import asyncio
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field, replace


@dataclass
class Memory:
    id: str
    namespace: str
    kind: str
    text: str
    importance: float
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    accesses: int = 0
    level: int = 0
    sources: list[str] = field(default_factory=list)


def importance_score(confidence, novelty, usefulness):
    return sum(w * min(1., max(0., float(v))) for w, v in
               zip((.4, .2, .4), (confidence, novelty, usefulness)))


def retention(memory, now=None, half_life_days=30):
    if half_life_days <= 0:
        raise ValueError('half_life_days must be positive')
    age = max(0, (time.time() if now is None else now) - memory.accessed_at) / 86400
    return min(1., memory.importance + .03 * math.log1p(memory.accesses)) * 2 ** (-age / half_life_days)


def compress(memory, score):
    """L0 raw -> L1 extractive sentences -> L2 compact excerpt, retaining provenance."""
    target = 2 if score < .15 else 1 if score < .4 else 0
    if target <= memory.level:
        return memory
    budget = 400 if target == 2 else 1600
    import re
    sentences = re.split(r'(?<=[.!?。！？])\s*|\n+', memory.text)
    selected, used = [], 0
    for sentence in sentences:
        if sentence and used + len(sentence) + bool(selected) <= budget:
            used += bool(selected)
            selected.append(sentence)
            used += len(sentence)
    excerpt = '\n'.join(selected) or memory.text[:budget]
    return replace(memory, text=excerpt, level=target)


class PostgresProcedural:
    def __init__(self, pool):
        self.pool = pool

    async def setup(self):
        async with self.pool.connection() as conn:
            await conn.execute('CREATE TABLE IF NOT EXISTS research_procedures '
                               '(namespace TEXT NOT NULL, id TEXT NOT NULL, data JSONB NOT NULL, '
                               'PRIMARY KEY(namespace,id))')

    async def put(self, memory):
        from psycopg.types.json import Jsonb
        async with self.pool.connection() as conn:
            await conn.execute('INSERT INTO research_procedures VALUES (%s,%s,%s) '
                'ON CONFLICT(namespace,id) DO UPDATE SET data=EXCLUDED.data',
                (memory.namespace, memory.id, Jsonb(asdict(memory))))

    async def list(self, namespace):
        async with self.pool.connection() as conn:
            cursor = await conn.execute('SELECT data FROM research_procedures WHERE namespace=%s', (namespace,))
            return [Memory(**row[0]) for row in await cursor.fetchall()]

    async def delete(self, namespace, memory_id):
        async with self.pool.connection() as conn:
            await conn.execute('DELETE FROM research_procedures WHERE namespace=%s AND id=%s', (namespace, memory_id))


class Neo4jEpisodic:
    def __init__(self, driver):
        self.driver = driver

    async def setup(self):
        async with self.driver.session() as session:
            await session.run('CREATE CONSTRAINT research_episode IF NOT EXISTS '
                              'FOR (e:ResearchEpisode) REQUIRE (e.namespace,e.id) IS UNIQUE')

    async def put(self, memory):
        async with self.driver.session() as session:
            await session.run('MERGE (e:ResearchEpisode {namespace:$namespace,id:$id}) '
                'SET e.data=$data WITH e UNWIND $sources AS source '
                'MERGE (p:ResearchSource {namespace:$namespace,id:source}) MERGE (e)-[:CITES]->(p)',
                namespace=memory.namespace, id=memory.id, data=json.dumps(asdict(memory)), sources=memory.sources)

    async def list(self, namespace):
        async with self.driver.session() as session:
            result = await session.run('MATCH (e:ResearchEpisode {namespace:$namespace}) RETURN e.data AS data', namespace=namespace)
            return [Memory(**json.loads(row['data'])) async for row in result]

    async def delete(self, namespace, memory_id):
        async with self.driver.session() as session:
            await session.run('MATCH (e:ResearchEpisode {namespace:$namespace,id:$id}) DETACH DELETE e',
                              namespace=namespace, id=memory_id)


class QdrantSemantic:
    collection = 'research_semantic'

    def __init__(self, client, embeddings):
        self.client, self.embeddings = client, embeddings

    async def setup(self):
        from qdrant_client import models as m
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(self.collection,
                vectors_config=m.VectorParams(size=self.embeddings.size, distance=m.Distance.COSINE))

    @staticmethod
    def scope(namespace):
        from qdrant_client import models as m
        return m.Filter(must=[m.FieldCondition(key='namespace', match=m.MatchValue(value=namespace))])

    async def put(self, memory):
        from qdrant_client import models as m
        vector = (await asyncio.to_thread(self.embeddings.encode, [memory.text]))[0]
        await self.client.upsert(self.collection, [m.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, memory.namespace + ':' + memory.id)),
            vector=vector, payload=asdict(memory))], wait=True)

    async def list(self, namespace):
        records, offset = [], None
        while True:
            points, offset = await self.client.scroll(self.collection, scroll_filter=self.scope(namespace),
                offset=offset, limit=100, with_payload=True, with_vectors=False)
            records.extend(Memory(**p.payload) for p in points)
            if offset is None:
                return records

    async def search(self, namespace, query, limit):
        vector = (await asyncio.to_thread(self.embeddings.encode, [query]))[0]
        result = await self.client.query_points(self.collection, query=vector,
            query_filter=self.scope(namespace), limit=limit)
        return [Memory(**p.payload) for p in result.points]

    async def delete(self, namespace, memory_id):
        from qdrant_client import models as m
        await self.client.delete(self.collection, m.PointIdsList(points=[
            str(uuid.uuid5(uuid.NAMESPACE_URL, namespace + ':' + memory_id))]), wait=True)


class MemoryManager:
    def __init__(self, episodic, semantic, procedural):
        self.stores = {'episodic': episodic, 'semantic': semantic, 'procedural': procedural}

    async def setup(self):
        for store in self.stores.values():
            await store.setup()

    async def recall(self, namespace, query, limit=4):
        recalled = []
        for kind, store in self.stores.items():
            if kind == 'semantic':
                memories = await store.search(namespace, query, limit)
            else:
                memories = await store.list(namespace)
                terms = set(query.lower().split())
                memories.sort(key=lambda m: retention(m) * (1 + len(terms & set(m.text.lower().split()))), reverse=True)
                memories = memories[:limit]
            for memory in memories:
                memory.accesses += 1
                memory.accessed_at = time.time()
                await store.put(memory)
                recalled.append(asdict(memory))
        return recalled

    async def remember(self, namespace, run_id, question, answer, sources, verified):
        # Deterministic IDs make checkpoint replay idempotent across stores.
        values = [('episodic', f'Question: {question}\nOutcome: {answer}', .65)]
        if verified:
            values.append(('semantic', answer, importance_score(.95, .5, .8)))
        for kind, text, importance in values:
            store = self.stores[kind]
            await store.put(Memory(f'{run_id}:{kind}', namespace, kind, text, importance, sources=sources))
        await self.maintain(namespace)

    async def seed_skills(self, namespace):
        from research.skills import SKILLS, skill_instructions
        existing = {m.id for m in await self.stores['procedural'].list(namespace)}
        for name in SKILLS:
            if name not in existing:
                await self.stores['procedural'].put(Memory(name, namespace, 'procedural',
                    skill_instructions(name), .95, sources=[f'skills/{name}/SKILL.md']))

    async def maintain(self, namespace):
        stats = {'compressed': 0, 'deleted': 0}
        for kind, store in self.stores.items():
            for memory in await store.list(namespace):
                if kind == 'procedural' and memory.importance >= .9:
                    continue  # Authored SKILL instructions are pinned.
                score = retention(memory)
                if score < .03 and memory.importance < .8:
                    await store.delete(namespace, memory.id)
                    stats['deleted'] += 1
                else:
                    compact = compress(memory, score)
                    if compact != memory:
                        await store.put(compact)
                        stats['compressed'] += 1
        return stats
