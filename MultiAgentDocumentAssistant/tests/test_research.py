import asyncio
import json
from dataclasses import replace

from research.memory import Memory, MemoryManager, compress, importance_score, retention
from research.rag import section_chunks, verify_citations
from research.routing import select_tier
from research.skills import format_citation
import pytest


def run(coro):
    return asyncio.run(coro)


def test_sections_provenance_overlap():
    pages = [{'doc_id': 'doc', 'title': 'Paper', 'page': 1,
              'text': '1 Introduction\n' + 'a' * 200 + '\n2 Methods\n' + 'b' * 100},
             {'doc_id': 'doc', 'title': 'Paper', 'page': 2, 'text': 'method continued'}]
    chunks = section_chunks(pages, size=80, overlap=10)
    assert chunks == section_chunks(pages, size=80, overlap=10)
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(len(c.text) <= 80 for c in chunks)
    assert chunks[0].text[-10:] == chunks[1].text[:10]
    assert not any('a' in c.text and 'b' in c.text for c in chunks)
    assert chunks[-1].page == 2 and chunks[-1].section == '2 Methods'


def test_retention_and_compression():
    m = Memory('id', 'n', 'episodic', 'Evidence sentence. ' * 200, .8,
               created_at=0, accessed_at=0, sources=['paper:1'])
    assert retention(m, now=30 * 86400) == .4
    assert retention(replace(m, accesses=10), now=0) > retention(m, now=0)
    compact = compress(m, .1)
    assert compact.level == 2 and len(compact.text) <= 400
    assert compact.sources == m.sources
    assert compress(compact, .9) == compact
    assert 0 <= importance_score(-1, 2, .5) <= 1


def test_routes_and_citations():
    assert select_tier('librarian', 'short') == 'qwen'
    assert select_tier('analyst', 'short') == 'deepseek'
    assert select_tier('librarian', 'x' * 25000) == 'claude'
    assert select_tier('analyst', 'short', retry=True) == 'claude'
    assert 'doi.org' not in format_citation('Title', ['A'], '2024')
    assert '\\{' in format_citation('{Title}', ['A'], '2024', style='bibtex')


class AuditRouter:
    def __init__(self, quote='real evidence'):
        self.quote = quote

    async def ask(self, *args, **kwargs):
        return json.dumps({'claims': [{'claim': 'claim', 'source_id': 'a' * 24,
                                      'quote': self.quote, 'supported': True}]})


def test_audit_rejects_fabricated_quote_unknown_and_missing_sources():
    evidence = [{'id': 'a' * 24, 'text': 'real evidence'}]
    assert run(verify_citations('claim [' + 'a' * 24 + ']', evidence, AuditRouter()))['valid']
    assert not run(verify_citations('claim [' + 'a' * 24 + ']', evidence, AuditRouter('fake')))['valid']
    assert not run(verify_citations('claim [' + 'b' * 24 + ']', evidence, AuditRouter()))['valid']
    assert not run(verify_citations('uncited', evidence, AuditRouter()))['valid']


class Store:
    def __init__(self):
        self.items = {}

    async def put(self, memory):
        self.items[(memory.namespace, memory.id)] = memory

    async def list(self, namespace):
        return [m for (n, _), m in self.items.items() if n == namespace]

    async def search(self, namespace, query, limit):
        return (await self.list(namespace))[:limit]

    async def delete(self, namespace, memory_id):
        del self.items[(namespace, memory_id)]


def test_memory_replay_isolation_and_unverified_gate():
    async def scenario():
        stores = [Store() for _ in range(3)]
        manager = MemoryManager(*stores)
        for _ in range(2):
            await manager.remember('a', 'run', 'q', 'answer', ['source'], True)
        assert len(await stores[0].list('a')) == 1
        assert len(await stores[1].list('a')) == 1
        await manager.remember('a', 'bad', 'q', 'bad', [], False)
        assert len(await stores[1].list('a')) == 1
        assert await manager.recall('b', 'q') == []
        await stores[0].put(Memory('old', 'a', 'episodic', 'old', .2, accessed_at=0))
        stats = await manager.maintain('a')
        assert stats['deleted'] == 1
    run(scenario())


def test_pdf_skill_preserves_page_and_rejects_image_only(tmp_path):
    import pymupdf
    from research.skills import parse_pdf
    path = tmp_path / 'paper.pdf'
    with pymupdf.open() as document:
        for text in ['Introduction', 'Methods']:
            page = document.new_page()
            page.insert_text((72, 72), text)
        document.save(path)
    pages = parse_pdf(str(path))
    assert [p['page'] for p in pages] == [1, 2]
    assert pages[0]['doc_id'] == pages[1]['doc_id']
    assert 'Methods' in pages[1]['text']
    blank = tmp_path / 'blank.pdf'
    with pymupdf.open() as document:
        document.new_page()
        document.save(blank)
    with pytest.raises(ValueError, match='OCR'):
        parse_pdf(str(blank))


def test_idea_skill_rejects_invented_sources():
    from research.skills import extract_ideas

    class Ideas:
        async def ask(self, *args):
            return json.dumps([dict(hypothesis='idea', motivation='m', experiment='e',
                                    limitations='l', source_ids=['b' * 24])])
    with pytest.raises(ValueError, match='unknown source'):
        run(extract_ideas('a' * 24, Ideas()))
