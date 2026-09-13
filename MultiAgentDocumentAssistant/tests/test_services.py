"""Opt-in integration: RESEARCH_TEST_POSTGRES_URI / RESEARCH_TEST_NEO4J_URI.

Use disposable databases. Each test isolates data under a random namespace.
"""
import asyncio
import os
import sys
import uuid

import pytest


@pytest.mark.skipif(not os.getenv('RESEARCH_TEST_POSTGRES_URI'), reason='Postgres test service not configured')
def test_postgres_checkpoint_survives_connection_restart():
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.types import Command
    from research.workflow import build_graph, initial_state
    from test_workflow import Index, Memory, Router

    async def scenario():
        uri = os.environ['RESEARCH_TEST_POSTGRES_URI']
        config = {'configurable': {'thread_id': 'integration-' + uuid.uuid4().hex}, 'recursion_limit': 80}
        router, memory = Router(), Memory()
        async with AsyncPostgresSaver.from_conn_string(uri) as saver:
            await saver.setup()
            graph = build_graph(router, Index(), memory, saver)
            result = await graph.ainvoke(initial_state('test', 'run', 'question', task='qa', pause=True), config)
            assert result['__interrupt__']
        before = list(router.calls)
        async with AsyncPostgresSaver.from_conn_string(uri) as saver:
            graph = build_graph(router, Index(), memory, saver)
            result = await graph.ainvoke(Command(resume=True), config)
            assert result['approved'] and before == router.calls
            await saver.adelete_thread(config['configurable']['thread_id'])
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(scenario())


@pytest.mark.skipif(not os.getenv('RESEARCH_TEST_POSTGRES_URI'), reason='Postgres test service not configured')
def test_procedural_postgres_roundtrip():
    from psycopg_pool import AsyncConnectionPool
    from research.memory import Memory, PostgresProcedural

    async def scenario():
        async with AsyncConnectionPool(os.environ['RESEARCH_TEST_POSTGRES_URI'], open=False) as pool:
            store = PostgresProcedural(pool)
            await store.setup()
            namespace = uuid.uuid4().hex
            item = Memory('id', namespace, 'procedural', 'procedure', .9)
            await store.put(item)
            await store.put(item)
            assert await store.list(namespace) == [item]
            await store.delete(namespace, item.id)
            assert await store.list(namespace) == []
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(scenario())


@pytest.mark.skipif(not os.getenv('RESEARCH_TEST_NEO4J_URI'), reason='Neo4j test service not configured')
def test_episodic_neo4j_roundtrip():
    from neo4j import AsyncGraphDatabase
    from research.memory import Memory, Neo4jEpisodic

    async def scenario():
        async with AsyncGraphDatabase.driver(os.environ['RESEARCH_TEST_NEO4J_URI'], auth=(
            os.getenv('NEO4J_USER', 'neo4j'), os.environ['NEO4J_PASSWORD'])) as driver:
            store = Neo4jEpisodic(driver)
            await store.setup()
            namespace = uuid.uuid4().hex
            item = Memory('id', namespace, 'episodic', 'episode', .9)
            await store.put(item)
            assert await store.list(namespace) == [item]
            await store.delete(namespace, item.id)
            assert await store.list(namespace) == []
    asyncio.run(scenario())
