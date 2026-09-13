---
name: pdf-parse
description: Extract scientific PDF text with document identity and page provenance before research indexing.
---

Call `research.skills.parse_pdf(path)` for a local PDF, then `PaperIndex.ingest(pages, namespace)`.
This reuses the project's PyMuPDF backend. Preserve the returned document ID, one-based page,
title and line breaks: section-aware splitting depends on them. Do not fabricate missing text.
An image-only PDF requires OCR before ingestion; the parser reports this explicitly.
CLI: `python -m research --namespace demo ingest paper.pdf`.
