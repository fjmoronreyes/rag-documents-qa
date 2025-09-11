# RAG Architecture for Document QA

This project implements a Retrieval-Augmented Generation (RAG) pipeline for PDF-based Question Answering.

## Components
- **PDF ingestion** – load and parse documents.
- **Chunking** – recursive token-based splitting with overlap.
- **Embeddings** – vector representation of chunks.
- **Vector DB** – storage and semantic retrieval (top-k).
- **Re-ranking** – optional cross-encoder or improved embeddings.
- **LLM** – answer generation with source citations.

## Deliverables
- Reproducible pipeline (Poetry + Docker).
- Evaluation dataset (`eval.jsonl`) with expected answers and sources.
- Metrics: answer-level accuracy and citation validity.