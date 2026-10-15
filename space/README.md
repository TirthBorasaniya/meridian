---
title: Meridian
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Meridian

A research assistant over a maintained corpus of arXiv papers on LLM reasoning and evaluation.
Every factual sentence of an answer cites a paper section; the assistant says when the corpus
does not support an answer. Excerpts are limited to two sentences, and every citation links to
the paper's arXiv abstract page.

The index is restored from the latest accepted daily snapshot at startup and refreshed hourly.
Web search fallback is disabled here; requests are limited to 10 per minute and 100 per day per
address. Source code: https://github.com/TirthBorasaniya/meridian
