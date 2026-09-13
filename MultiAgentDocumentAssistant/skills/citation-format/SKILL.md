---
name: citation-format
description: Format supplied paper metadata as APA-style references or escaped BibTeX without inventing bibliographic fields.
---

Call `research.skills.format_citation(title, authors, year, doi='', style='apa')`.
`authors` is a list; `style` is `apa` or `bibtex`. Required fields must come from the caller
or a verified paper. This lightweight formatter does not implement every APA publication subtype.
Missing optional DOI stays omitted. Formatting success is not verification of the paper's existence.
Use the `citation` task for the deterministic fast path; do not add unsupported metadata.
