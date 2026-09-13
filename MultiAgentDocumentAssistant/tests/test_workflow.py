import asyncio
import json
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from research.workflow import ROLES, build_graph, initial_state


SOURCE = 'a' * 24


class SupervisorModel(BaseChatModel):
    @property
    def _llm_type(self):
        return 'test-supervisor'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        done = {m.name for m in messages if isinstance(m, AIMessage)}
        role = next((r for r in ROLES if r not in done), None)
        message = AIMessage(content='done') if role is None else AIMessage(content='',
            tool_calls=[{'name': 'transfer_to_' + role, 'args': {}, 'id': 'handoff-' + role}])
        return ChatResult(generations=[ChatGeneration(message=message)])


class Router:
    def __init__(self, bad=False):
        self.calls = []
        self.bad = bad

    def get(self, *args, **kwargs):
        return SupervisorModel()

    async def ask(self, role, prompt, **kwargs):
        self.calls.append(role)
        if prompt.startswith('Audit'):
            return json.dumps({'claims': [{'claim': 'claim', 'source_id': SOURCE,
                                          'quote': 'fake' if self.bad else 'real evidence', 'supported': True}]})
        if prompt.startswith('Extract'):
            return json.dumps([dict(hypothesis='proposal', motivation='evidence', experiment='test',
                                    limitations='unknown novelty', source_ids=[SOURCE])])
        return 'claim [' + SOURCE + ']'


class Index:
    async def search(self, *args):
        return [{'id': SOURCE, 'text': 'real evidence'}]


class Memory:
    def __init__(self):
        self.saved = []

    async def seed_skills(self, *args):
        pass

    async def recall(self, *args):
        return []

    async def remember(self, *args):
        self.saved.append(args)


def test_real_supervisor_four_workers_and_interrupt_after_rebuild():
    async def scenario():
        saver, memory, router = InMemorySaver(), Memory(), Router()
        config = {'configurable': {'thread_id': 'research'}, 'recursion_limit': 80}
        graph = build_graph(router, Index(), memory, saver)
        result = await graph.ainvoke(initial_state('a', 'run', 'Research question', pause=True), config)
        assert set(result['reports']) == set(ROLES)
        assert result['audit']['valid'] and result['__interrupt__']
        assert memory.saved == []
        before = list(router.calls)
        # A fresh compiled graph resumes from persisted state without repeating LLM work.
        graph = build_graph(router, Index(), memory, saver)
        result = await graph.ainvoke(Command(resume=True), config)
        assert result['approved']
        assert router.calls == before
        assert len(memory.saved) == 1 and memory.saved[0][-1] is True
    asyncio.run(scenario())


def test_fast_path_and_bounded_failed_audit():
    async def scenario():
        memory, router = Memory(), Router(bad=True)
        graph = build_graph(router, Index(), memory, InMemorySaver())
        result = await graph.ainvoke(initial_state('a', 'run', 'Question', task='qa'),
                                    {'configurable': {'thread_id': 'fast'}})
        assert set(result['reports']) == {'analyst'}
        assert result['attempts'] == 2 and not result['audit']['valid']
        assert '无法提供可靠结论' in result['answer']
        assert memory.saved[0][-1] is False
    asyncio.run(scenario())


def test_citation_fast_path_and_reject():
    async def scenario():
        memory, router = Memory(), Router()
        graph = build_graph(router, Index(), memory, InMemorySaver())
        config = {'configurable': {'thread_id': 'citation'}}
        result = await graph.ainvoke(initial_state('a', 'run', 'Format', task='citation',
            citation={'title': 'Paper', 'authors': ['Author'], 'year': '2024'}, pause=True), config)
        assert router.calls == [] and result['audit']['scope'].startswith('format_only')
        await graph.ainvoke(Command(resume=False), config)
        assert memory.saved == []
    asyncio.run(scenario())


def test_failed_node_resumes_without_repeating_specialist():
    class FailOnce(Router):
        failed = False

        async def ask(self, role, prompt, **kwargs):
            if prompt.startswith('Answer the question') and not self.failed:
                self.failed = True
                raise ConnectionError('temporary provider failure')
            return await super().ask(role, prompt, **kwargs)

    async def scenario():
        router, memory, saver = FailOnce(), Memory(), InMemorySaver()
        config = {'configurable': {'thread_id': 'failure'}}
        graph = build_graph(router, Index(), memory, saver)
        with pytest.raises(ConnectionError):
            await graph.ainvoke(initial_state('a', 'run', 'Question', task='qa'), config)
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ('compose',)
        assert router.calls == ['analyst']
        rebuilt = build_graph(router, Index(), memory, saver)
        result = await rebuilt.ainvoke(None, config)
        assert result['audit']['valid']
        assert router.calls == ['analyst', 'analyst', 'reviewer']
        assert len(memory.saved) == 1
    asyncio.run(scenario())
