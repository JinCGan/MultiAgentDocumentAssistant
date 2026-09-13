# Multi-Agent Research Paper Assistant

An evidence-grounded research assistant built with LangGraph Supervisor. It coordinates four specialized agents for literature review, method analysis, research idea generation, and citation verification, with durable execution, long-term memory, hybrid RAG, and dynamic model routing.

## Features

- **Four specialized agents** coordinated by LangGraph Supervisor:
  - Literature Agent organizes papers and identifies missing evidence.
  - Analysis Agent compares methods, experiments, baselines, and limitations.
  - Idea Agent proposes testable hypotheses and follow-up experiments.
  - Review Agent checks evidence quality and unsupported claims.
- **Durable workflows** powered by `AsyncPostgresSaver`, including checkpointing, failure recovery, and human-in-the-loop interrupts.
- **Fast paths** for focused Q&A, idea extraction, and deterministic citation formatting.
- **Three-layer long-term memory**:
  - Episodic memory in Neo4j.
  - Semantic memory in Qdrant.
  - Procedural memory in PostgreSQL.
- **Dynamic memory retention** using importance scoring, time decay, access frequency, and multi-level extractive compression.
- **Optimized RAG pipeline** with section-aware chunking, multiple vector representations, BM25, reciprocal rank fusion, and BGE reranking.
- **Citation self-reflection** that validates source IDs and supporting excerpts before conclusions enter semantic memory.
- **Reusable skills** for PDF parsing, citation formatting, and research idea extraction.
- **Dynamic model routing** across Qwen, DeepSeek, and Claude according to task role, prompt size, and retry stage.

## Architecture

```mermaid
flowchart TD
    A[Research request] --> B[Memory recall and paper retrieval]
    B --> C{Task type}
    C -->|research| S[LangGraph Supervisor]
    S --> L[Literature Agent]
    S --> N[Analysis Agent]
    S --> I[Idea Agent]
    S --> R[Review Agent]
    L --> D[Answer synthesis]
    N --> D
    I --> D
    R --> D
    C -->|qa / ideas| F[Specialist fast path]
    F --> D
    C -->|citation| G[Citation formatter]
    D --> V[Citation validation]
    V -->|Failed, up to two repairs| P[Claude repair]
    P --> V
    V --> H[Optional human review]
    G --> H
    H --> M[Persist and maintain memory]
```

## Requirements

- Python 3.11 or 3.12
- Docker and Docker Compose
- PostgreSQL 16+
- Neo4j 5.26+
- Qdrant 1.15+
- API credentials for Qwen, DeepSeek, and Claude

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv-research
```

```powershell
# Windows PowerShell
.\.venv-research\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv-research/bin/activate
```

Install the dependencies and start the local services:

```bash
python -m pip install -r requirements-research.txt
docker compose -f compose.research.yaml up -d
```

## Configuration

Use [.env.research.example](.env.research.example) as a configuration reference. The application does not automatically load `.env` files, so export the variables in your shell or deployment environment.

```powershell
$env:POSTGRES_URI = 'postgresql://research:research@localhost:5432/research'
$env:NEO4J_URI = 'bolt://localhost:7687'
$env:NEO4J_USER = 'neo4j'
$env:NEO4J_PASSWORD = 'research-password'
$env:QDRANT_URL = 'http://localhost:6333'

$env:QWEN_API_KEY = '<your-key>'
$env:QWEN_MODEL = '<your-model-id>'
$env:QWEN_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

$env:DEEPSEEK_API_KEY = '<your-key>'
$env:DEEPSEEK_MODEL = '<your-model-id>'
$env:DEEPSEEK_BASE_URL = 'https://api.deepseek.com'

$env:CLAUDE_API_KEY = '<your-key>'
$env:CLAUDE_MODEL = '<your-model-id>'
```

Use model IDs available to your provider accounts. Replace the development database credentials before deployment.

## Usage

### Ingest papers

PDF files must contain a text layer. Scanned documents require OCR before ingestion.

```bash
python -m research --namespace lab-a ingest paper1.pdf paper2.pdf
```

### Run a full research task

```bash
python -m research --namespace lab-a run --thread research-001 \
  "Compare the papers' methods and limitations, then propose testable follow-up ideas."
```

### Use a fast path

```bash
# Focused question answering
python -m research --namespace lab-a run --thread qa-001 --task qa \
  "Which experimental baselines were used?"

# Research idea extraction
python -m research --namespace lab-a run --thread ideas-001 --task ideas \
  "Propose testable follow-up experiments."
```

### Pause and resume

```bash
# Pause before committing the result to long-term memory
python -m research --namespace lab-a run --thread review-001 --pause \
  "Assess whether the conclusions are sufficiently supported."

# Inspect the saved state
python -m research --namespace lab-a status --thread review-001

# Accept or reject the paused result
python -m research --namespace lab-a resume --thread review-001 --decision accept
python -m research --namespace lab-a resume --thread review-001 --decision reject

# Resume after a process or service failure
python -m research --namespace lab-a resume --thread research-001
```

`--namespace` isolates documents, memories, and workflow state. Use a new `--thread` value for each new task, and avoid running concurrent processes for the same thread.

### Maintain long-term memory

Maintenance runs automatically after a result is stored. It can also be triggered manually:

```bash
python -m research --namespace lab-a maintain
```

## Retrieval Pipeline

1. PDF text is split without crossing detected section or page boundaries.
2. Each chunk stores document identity, title, section, page number, and a stable source ID.
3. Qdrant indexes two dense representations (`body` and `context`) and one BM25 sparse representation (`lexical`).
4. The three retrieval branches are combined using reciprocal rank fusion.
5. `BAAI/bge-reranker-v2-m3` reranks the fused candidates.
6. Generated answers cite evidence using `[chunk_id]` references.
7. The review stage verifies every factual claim against an exact excerpt from the cited source.

The default dense encoder is `BAAI/bge-small-en-v1.5`. Set `RESEARCH_EMBED_MODEL` to another model supported by FastEmbed when needed. Rebuild the collection when changing embedding models.

## Long-Term Memory

| Layer | Store | Purpose |
| --- | --- | --- |
| Episodic | Neo4j | Questions, outcomes, and source relationships |
| Semantic | Qdrant | Citation-verified conclusions for vector recall |
| Procedural | PostgreSQL | Reusable skills and operating procedures |

Retention combines importance, access count, and a 30-day half-life. Low-scoring memories are progressively compressed to 1,600-character and 400-character excerpts. Low-importance records are deleted only after their retention score falls below the cleanup threshold. Source provenance is retained during compression.

## Model Routing

| Condition | Provider |
| --- | --- |
| Short literature organization tasks | Qwen |
| Supervisor, analysis, idea generation, or prompts over 6,000 characters | DeepSeek |
| Review, repair, retries, or prompts over 24,000 characters | Claude |

Provider failures do not silently fall back to another account. Fix the provider configuration and resume the saved workflow instead.

## Reusable Skills

| Skill | Purpose |
| --- | --- |
| [PDF parsing](skills/pdf-parse/SKILL.md) | Extract text while preserving document and page provenance |
| [Citation formatting](skills/citation-format/SKILL.md) | Format supplied metadata as simplified APA or BibTeX |
| [Idea extraction](skills/idea-extract/SKILL.md) | Produce testable hypotheses, experiments, limitations, and source IDs |

The executable implementations are registered in `research.skills.SKILLS`.

## Project Structure

```text
research/
  __main__.py              Command-line interface
  workflow.py              Supervisor, agents, and fast paths
  runtime.py               Service and checkpoint lifecycle
  memory.py                Long-term memory and retention policies
  rag.py                   Chunking, retrieval, reranking, and validation
  routing.py               Dynamic model routing
  skills.py                Executable skill registry
  config.py                Environment configuration
skills/                    Reusable skill instructions
tests/                     Unit, workflow, and service integration tests
docs/                      Detailed design and operations guide
compose.research.yaml       Local database services
requirements-research.txt  Python dependencies
```

## Testing

Run the offline test suite:

```bash
python -m pytest tests -q
```

Optional PostgreSQL and Neo4j integration tests require disposable test services:

```powershell
$env:RESEARCH_TEST_POSTGRES_URI = $env:POSTGRES_URI
$env:RESEARCH_TEST_NEO4J_URI = $env:NEO4J_URI
python -m pytest tests/test_services.py -q
```

The offline tests cover agent coordination, fast paths, checkpoint recovery, citation failure handling, namespace isolation, idempotency, PDF parsing, memory decay, compression, and local Qdrant retrieval.

## Current Limitations

- Full end-to-end behavior depends on live PostgreSQL, Neo4j, Qdrant, model APIs, and downloaded embedding/reranker weights.
- OCR and complex multi-column layout reconstruction are not implemented.
- The default embedding model primarily targets English papers; multilingual retrieval quality requires separate evaluation.
- Citation validation combines model judgment with exact excerpt checks and may still miss unsupported claims.
- Research idea novelty is marked as unverified until confirmed through a separate literature search.
- Namespaces provide data partitioning, not authentication.
- Writes across the three memory stores are not distributed transactions.

See the [detailed design and operations guide](docs/research-assistant.md) for persistence semantics, scoring formulas, recovery behavior, and implementation boundaries.

## License

Distributed under the [MIT License](LICENSE).
