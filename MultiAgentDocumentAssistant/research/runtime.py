import asyncio
from contextlib import AsyncExitStack, asynccontextmanager

from research.config import Settings
from research.memory import MemoryManager, Neo4jEpisodic, PostgresProcedural, QdrantSemantic
from research.rag import BGEReranker, Embeddings, PaperIndex
from research.routing import ModelRouter


@asynccontextmanager
async def runtime():
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from neo4j import AsyncGraphDatabase
    from psycopg_pool import AsyncConnectionPool
    from qdrant_client import AsyncQdrantClient
    settings = Settings.from_env()
    async with AsyncExitStack() as stack:
        pool = await stack.enter_async_context(AsyncConnectionPool(settings.postgres_uri, open=False))
        driver = await stack.enter_async_context(AsyncGraphDatabase.driver(settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)))
        client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_key)
        stack.push_async_callback(client.close)
        saver = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(settings.postgres_uri))
        await saver.setup()
        embeddings = await asyncio.to_thread(Embeddings)
        reranker = await asyncio.to_thread(BGEReranker)
        memory = MemoryManager(Neo4jEpisodic(driver), QdrantSemantic(client, embeddings), PostgresProcedural(pool))
        index = PaperIndex(client, embeddings, reranker)
        await memory.setup()
        await index.setup()
        yield ModelRouter(), index, memory, saver
