import argparse
import asyncio
import hashlib
import json
import sys


def parser():
    p = argparse.ArgumentParser(description='Multi-agent scientific paper assistant')
    p.add_argument('--namespace', required=True, help='Workspace/user isolation key')
    sub = p.add_subparsers(dest='command', required=True)
    ingest = sub.add_parser('ingest')
    ingest.add_argument('pdf', nargs='+')
    run = sub.add_parser('run')
    run.add_argument('--thread', required=True)
    run.add_argument('--task', choices=['research', 'qa', 'ideas', 'citation'], default='research')
    run.add_argument('--pause', action='store_true')
    run.add_argument('--citation-json', default='{}')
    run.add_argument('question')
    resume = sub.add_parser('resume')
    resume.add_argument('--thread', required=True)
    resume.add_argument('--decision', choices=['accept', 'reject'])
    status = sub.add_parser('status')
    status.add_argument('--thread', required=True)
    sub.add_parser('maintain')
    return p


async def main(args):
    from langgraph.types import Command
    from research.runtime import runtime
    from research.skills import parse_pdf
    from research.workflow import build_graph, initial_state
    async with runtime() as (router, index, memory, saver):
        if args.command == 'ingest':
            for path in args.pdf:
                pages = await asyncio.to_thread(parse_pdf, path)
                print(json.dumps({'file': path, 'chunks': await index.ingest(pages, args.namespace)}, ensure_ascii=False))
            return
        if args.command == 'maintain':
            print(await memory.maintain(args.namespace))
            return
        thread_id = hashlib.sha256(json.dumps([args.namespace, args.thread]).encode()).hexdigest()
        config = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 80}
        graph = build_graph(router, index, memory, saver)
        snapshot = await graph.aget_state(config)
        if args.command == 'status':
            print(json.dumps({'next': snapshot.next, 'answer': snapshot.values.get('answer'),
                              'tasks': [str(t.interrupts) for t in snapshot.tasks]}, ensure_ascii=False))
            return
        if args.command == 'run':
            if snapshot.values:
                raise ValueError('Thread already exists. Use resume or choose a new thread ID.')
            request = initial_state(args.namespace, thread_id, args.question, args.task,
                                    json.loads(args.citation_json), args.pause)
        else:
            if not snapshot.values or not snapshot.next:
                raise ValueError('No pending task for this namespace/thread')
            interrupted = any(t.interrupts for t in snapshot.tasks)
            if interrupted and args.decision is None:
                raise ValueError('Paused task requires --decision accept or reject')
            if not interrupted and args.decision is not None:
                raise ValueError('Failed task resumes without --decision')
            request = Command(resume=args.decision == 'accept') if interrupted else None
        result = await graph.ainvoke(request, config)
        print(json.dumps({'answer': result.get('answer'), 'audit': result.get('audit'),
                          'interrupt': result.get('__interrupt__')}, ensure_ascii=False, default=str))


if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(parser().parse_args()))
