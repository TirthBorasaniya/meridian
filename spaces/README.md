---
title: Meridian Agentic RAG
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Meridian

Meridian is an agentic retrieval-augmented generation system with hybrid
dense and sparse retrieval fused by Reciprocal Rank Fusion, cross-encoder
reranking with CRAG-style document grading, a LangGraph state machine with
three independent failure-mode recovery paths (retrieval fallback,
hallucination retry, off-target query rewrite), tiered LLM inference, and
cross-session conversation memory. Ask a question below; the response
includes the retrieved sources and the graph path taken to produce the
answer.

This Space queries a pre-built vector index and does not perform corpus
ingestion. Configure `GROQ_API_KEY`, `TAVILY_API_KEY`, `QDRANT_URL`, and
`QDRANT_API_KEY` as Space secrets before use.
