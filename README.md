# rag-documents-qa

This repository implements a Retrieval-Augmented Generation (RAG) pipeline for answering natural language questions over PDF documents. It handles the full flow: PDF ingestion, text extraction, recursive chunking with overlap, embeddings generation, vector database storage, semantic retrieval, and LLM-based answer generation with source citations.

## Table of Contents
1. [Introduction](#introduction)
2. [Document Processor](#document-processor)
   - [Architectural Choices](#architectural-choices)
   - [Chunk size and overlap](#chunk-size-and-overlap)
3. [Vector Database](#vector-database)
   - [Architectural Choices](#architectural-choices-1)
   - [Why Open Source Models](#why-open-source-models)
   - [A Note on Model Size](#a-note-on-model-size)
   - [Why ChromaDB](#why-chromadb)
4. [GenAI](#genai)
5. [Main Pipeline](#main-pipeline)
6. [Evaluation](#evaluation)
   - [How it works](#how-it-works)
   - [Why We Didn’t Use Cross-Encoders or Embedding Improvements](#why-we-didnt-use-cross-encoders-or-embedding-improvements)
   - [Metrics](#metrics)
   - [Aggregated Metrics](#aggregated-metrics)
   - [Why this matters](#why-this-matters)

## Introduction

The goal of this project is to provide an end-to-end pipeline for **document question answering** using Retrieval-Augmented Generation. The system ingests PDF files, segments them into overlapping chunks, and stores vector representations in a persistent ChromaDB instance. Queries are embedded and matched against the database to retrieve relevant passages, which are then passed to a language model to generate final answers grounded in the original documents.  

The repository also includes an **evaluation framework** to benchmark answer quality against a gold dataset, measuring both response accuracy and the percentage of answers that cite correct supporting passages.

The implementation follows a **clean, modular architecture** designed for clarity, reproducibility, and extensibility. Each step of the pipeline is encapsulated in its own component, following the principle of *separation of concerns*. This makes it easy to swap out models, replace the vector database, or extend evaluation metrics without touching unrelated logic.  

Key design principles:
- **Object-Oriented Structure**: the pipeline is organized into classes that encapsulate state and behavior, providing clarity and modularity. While the design is object-oriented in structure, it does not rely on deep inheritance hierarchies or abstract interfaces instead, it favors composition and modular components over strict OOP patterns.
- **Single Responsibility**: each class (e.g., `DocumentLoader`, `TextExtractor`, `ChunkBuilder`, `TextEmbedder`, `IndexCreator`) is responsible for exactly one task.  
- **Modularity and Reusability**: embedding, indexing, and evaluation logic are decoupled, allowing independent testing and replacement.  
- **Transparency and Logging**: all major operations are logged for traceability and debugging.  
- **Reproducibility**: deterministic chunking, deduplication by `chunk_id`, and persistent storage ensure results can be reproduced across runs.  
- **Extensibility**: the pipeline is designed so new embedding models, retrievers, or evaluation metrics can be integrated with minimal changes.  

## Document Processor

The Document Processor module is responsible for preparing raw PDFs so they can be used in the RAG pipeline. It covers three key steps: loading the documents, extracting and splitting text, and converting those splits into structured chunks that can be indexed and retrieved later.

- **DocumentLoader**  
  Scans the configured directory for PDF files and provides an iterator over them. It validates the directory path and logs how many documents were found.

- **TextExtractor**  
  Uses `PyPDFLoader` to read PDFs and HuggingFace tokenizers with a recursive character splitter to break text into overlapping chunks. This ensures that each chunk fits within a token limit while preserving context across boundaries.

- **Chunk (dataclass)**  
  A lightweight representation of a chunk. It stores the chunk ID (hash of its content), text content, source path, page information, metadata, and an optional embedding.

- **ChunkBuilder**  
  Converts LangChain `Document` objects into `Chunk` instances. It can optionally save them as JSON files under the configured chunks directory. The builder handles filename derivation from metadata and ensures consistent chunk IDs using SHA-256 hashes.

Together, these components transform PDFs into standardized chunks of text that the vector database can embed and store.

### Architectural Choices

- Using **PyPDFLoader** and a HuggingFace tokenizer ensures compatibility across different types of documents while giving us full control over how text is split.  
- The **RecursiveCharacterTextSplitter** was picked because it balances token size and overlap in a way that keeps enough context for downstream models.  
- Representing each chunk as a **dataclass** makes it easy to extend with new metadata fields without changing the rest of the pipeline.  
- The **ChunkBuilder** persists chunks in JSON so we can inspect, debug, or reuse them independently of the vector database.

### Chunk size and overlap

We configured chunking with a size of **500 tokens** and an overlap of **50 tokens**. The reasoning was straightforward:

- A size of 500 keeps each chunk compact enough to fit comfortably within embedding and model context limits, while still carrying enough semantic weight to be meaningful on its own.  
- An overlap of 50 ensures continuity between chunks so that information crossing a boundary is not lost. This is especially important in documents like PDFs where paragraphs can flow without clear delimiters.  

The choice of these parameters balances **context preservation** with **efficiency**: smaller chunks increase precision in retrieval, while overlap prevents critical context from being truncated.

## Vector Database

The Vector Database module handles how chunks are represented as dense vectors and indexed into Chroma for fast semantic search. It covers two main responsibilities: embedding generation and index management.

- **TextEmbedder**  
  Wraps a `SentenceTransformer` model to generate embeddings for documents, queries, or arbitrary text. The embedder ensures a consistent vector dimension and logs each operation. It supports embedding single texts, batches of documents, and queries specifically for retrieval.

- **IndexCreator**  
  Provides a thin layer over ChromaDB for adding pre-embedded chunks. It performs deduplication to avoid re-indexing the same chunk twice, stores embeddings alongside metadata (such as `source` and `page_label`), and logs every indexing step. By separating the embedding logic from the indexing logic, it keeps the pipeline modular and easier to debug.

### Architectural Choices

For this module, we chose to keep embedding and indexing as separate responsibilities. This allows us to:
- Swap out the embedding model without touching database logic.
- Reuse the same indexer for different embedding strategies.
- Guarantee reproducibility by deduplicating on `chunk_id` before insertion.
- Maintain transparency, since every step (embedding, indexing, deduplication) is explicitly logged.

The system uses **SentenceTransformers** for embeddings because they provide a well-optimized interface for Hugging Face models, efficient batching, and broad support for retrieval-oriented architectures.


### Why Open Source Models

One of the key design choices in this project was to rely on **open-source models** instead of external APIs. Today, most RAG implementations simply call OpenAI endpoints, which makes them harder to run locally and more dependent on external costs and service availability. We wanted something different: a system that is original, reproducible, and able to run fully on CPU.

We selected the **Qwen family of models** for both embeddings and generation. From the start, i knew this project had to run on a very limited environment: a machine with only **8 GB of RAM** and **no GPU available**. In practice, this meant we couldn’t even consider the 4B or 8B variants of Qwen3, since they require a GPU and much more memory. Instead, we deliberately chose the **0.6B versions**, both for embeddings and for generation.  


- **Embeddings:** [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)  
  This model series was designed specifically for text embedding and ranking tasks. Despite being a smaller variant (0.6B parameters), it delivers outstanding results: it ranks very high on the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) across multiple tasks. Its multilingual support and long-context capability make it an excellent fit for document QA, even when running only on CPU.

- **Generation:** [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B#qwen3-highlights)  
  We also use the 0.6B generative variant for answer generation. The reason is practical: we run on a very limited machine, with no GPU available. Qwen3-0.6B is one of the few modern LLMs that can still run efficiently on CPU while maintaining acceptable performance for RAG tasks.

This combination gives us a **differentiator**: while most projects tie themselves to OpenAI or other cloud providers, this implementation proves that a fully functional RAG pipeline can be built on open-source models that are performant, multilingual, and lightweight enough to run locally.

In short, we prioritized:
- **Local execution** without GPUs.  
- **Competitive benchmarks** (Qwen ranks among the top models on MTEB for embeddings).  
- **Originality** by avoiding yet another OpenAI-dependent pipeline.

### A Note on Model Size

It goes without saying that a larger model would deliver stronger results. This is not something we needed to prove—benchmarking and research have already demonstrated the clear correlation between model size and downstream performance. Our goal here was different: to build a **lightweight, CPU-friendly RAG pipeline** that shows how far you can go without a GPU and with very limited memory.  

This project is not about pushing state-of-the-art numbers, but about **practicality, portability, and originality**. By selecting the smallest Qwen3 models, we proved that a fully functional RAG pipeline can be deployed in constrained environments without relying on external APIs or expensive hardware.

**The results are not meant to be spectacular, and we will discuss them later in detail.**

### Why ChromaDB

For the vector database layer, we decided to use **ChromaDB**. The decision was based on three simple but strong reasons:

- **Persistence by default**: unlike FAISS or some in-memory stores (as far as i know), ChromaDB provides an easy persistent client out of the box. This makes it straightforward to keep an index across sessions without extra integration work.
- **Lightweight and CPU-friendly**: ChromaDB runs locally without requiring any special dependencies or GPU acceleration, aligning with our constraint of building a pipeline that works on a modest machine.
- **Developer experience**: its Python API is clean and easy to integrate. The ability to add documents with metadata, query with embeddings, and retrieve structured results makes it a great fit for RAG prototyping.

There are more advanced options available in the ecosystem (e.g., Weaviate, Milvus, Vespa), but for this project the priority was to keep the setup lightweight, reproducible, and aligned with our resource constraints. ChromaDB offered exactly that balance.

## Genai

The Genai module is responsible for producing final answers from the query and the retrieved context. It wraps a causal language model (Qwen3-0.6B in our case) and enforces strict grounding rules so that every response is tied to the provided evidence.

- **AnswerGenerator**  
  Loads a HuggingFace causal LM and tokenizer, then generates answers by combining the query with the top-k retrieved passages. The generator is instructed to:
  - Never use external knowledge.  
  - Always provide explicit citations.  
  - Reference the source passage, the page number, and the `chunk_id`.  
  - Return *“The context does not provide this information.”* if the evidence is insufficient.

The design ensures that answers are not only fluent, but also verifiable: each piece of information is backed by a traceable citation to the source document.

## Main Pipeline

The `IndexingPipeline` is the main entry point of the system. It orchestrates all components and turns raw PDFs into a fully searchable vector database. In practice, running `main.py` executes the entire flow end-to-end:

1. **Load PDFs**  
   The pipeline starts with the `DocumentLoader`, which scans the configured directory and yields every PDF it finds. For demonstration purposes, we used PDFs exported from Wikipedia articles about videogames: they are public, varied in structure, and provide enough complexity to validate the pipeline.

2. **Extract text**  
   Each document is passed through the `TextExtractor`, which tokenizes and splits text into overlapping windows. This ensures that no relevant context is lost when a passage spans across chunk boundaries.

3. **Build chunks**  
   The `ChunkBuilder` wraps each passage into a `Chunk` object with a unique hash-based ID and rich metadata (source, page number, page label). Chunks are also stored as JSON files, making them easy to inspect or reuse outside the database.

4. **Generate embeddings**  
   The `TextEmbedder` converts each chunk into a dense vector representation. To keep memory usage under control, embeddings are computed in small batches (default: 8 chunks).

5. **Index into ChromaDB**  
   Finally, the `IndexCreator` writes the embeddings and metadata into a persistent Chroma collection. This gives us a lightweight but durable vector store ready for retrieval.

By the end of the run, the PDFs have been transformed into an indexed knowledge base inside ChromaDB. This is the backbone of the RAG system: every query will be matched against these chunks, ensuring answers are grounded in the original documents.

## Evaluation

The **RagEvaluator** is the component responsible for validating the quality of the full RAG pipeline. It takes the system outputs (answers.json) and compares them against a manually curated gold dataset (gold.json). The evaluation is twofold: first generating model answers with citations, and then computing metrics that assess both answer quality and evidence usage.

### How it works

1. **Answer generation**  
   The evaluator loads each query from the gold dataset and runs it through the full retrieval and generation pipeline:
   - The query is embedded with the same `TextEmbedder` used at indexing time.  
   - The top-k passages are retrieved from ChromaDB.  
   - These passages are concatenated into a context and passed to the `AnswerGenerator`.  
   - The model produces an **inferenced_answer**, with explicit citations to passages by their `chunk_id`.  
   - Retrieved passages and cited passages are stored alongside the answer for later inspection.  
   This process produces the file `answers.json`.

2. **Evaluation against gold**  
   Once the answers are generated, the evaluator computes per-query and global metrics by comparing `answers.json` to `gold.json`.  
   - For each query, it checks whether the answer text matches the expected reference and whether the correct passages were retrieved and cited.  
   - Results are aggregated into `eval.json`, containing both per-query breakdowns and global averages.

### Why We Didn’t Use Cross-Encoders or Embedding Improvements

We are fully aware of the potential improvements that could come from adding a **cross-encoder re-ranking stage**, using **higher-dimensional embeddings**, or even adopting **hybrid search strategies** (dense + sparse retrieval).  

- A **cross-encoder** works by jointly encoding the query and each candidate passage, assigning a precise relevance score instead of relying purely on cosine similarity of embeddings. This usually leads to a significant boost in retrieval accuracy, as the model can take into account fine-grained semantic nuances and contextual interactions between query and passage.  
- **Embedding improvements**, such as switching to larger models or instruction-tuned embeddings, would further increase recall and precision, capturing richer semantic representations.  
- **Hybrid search** (combining embeddings with keyword-based search like BM25) can also guard against cases where purely semantic retrieval fails, especially for rare terms or names.

However, introducing any of these techniques was **out of scope for this project**, and for good reasons:  

- **Resource constraints**: running cross-encoders or larger embedding models is significantly more memory-intensive. Our environment is deliberately limited: a CPU-only machine with just **7.2 GB of RAM** and no GPU. Cross-encoders in particular are designed for GPU acceleration and become impractically slow on CPU.  
- **Time constraints**: while hybrid retrieval and re-ranking are well-known techniques, they require careful tuning and additional engineering effort (e.g., managing multiple retrieval strategies, merging scores). The priority here was to deliver a fully working RAG pipeline end-to-end, not to optimize every stage.  
- **Scope clarity**: this project was about showing that a complete, **self-contained RAG pipeline** can be built with modest resources. Adding cross-encoders would shift the focus toward state-of-the-art performance, which was not the goal.  

In short, we did not omit these improvements out of ignorance but out of **pragmatism**. They represent clear next steps for future work, but in this context—**a lightweight, CPU-only RAG system**—they were not feasible.  

### Metrics

The evaluation is designed to capture three critical dimensions of a RAG pipeline: answer fidelity, citation correctness, and retriever quality.

- **answer_score**  
  Measures similarity between the model’s `inferenced_answer` and the `expected_answer` in the gold data. The score is computed with `difflib.SequenceMatcher`, which gives a ratio between 0.0 (no overlap) and 1.0 (identical).  
  *This metric answers: “How close is the model’s output to the reference wording?”*

- **has_valid_citation**  
  Checks whether the model cited at least one correct passage. A citation is valid if any `chunk_id` in the gold `source_passages` appears in the model’s cited `source_passages`.  
  *This metric answers: “Did the model explicitly ground its answer in the correct evidence?”*

- **retrieval_recall**  
  Checks whether the retriever surfaced at least one relevant passage, regardless of whether the model cited it. A retrieval is counted as a hit if any gold `chunk_id` appears in the model’s `retrieved_passages`.  
  *This metric answers: “Did the retriever bring the right evidence into the candidate pool?”*

### Aggregated Metrics

At the dataset level, three aggregated values summarize performance:

- **average_answer_score**: mean of all per-query answer scores. Reflects overall closeness of model answers to gold references.  
- **valid_citation_rate**: proportion of queries where `has_valid_citation=True`. Captures how consistently the model grounds its answers in the correct sources.  
- **retrieval_recall_rate**: proportion of queries where `retrieval_recall=True`. Captures how often the retriever made the right evidence available to the generator.  

The final output (`eval.json`) contains both the per-query results and these aggregated metrics.

### Why this matters

Unlike pure generative evaluation, this framework makes sure we are not only generating plausible answers, but also:
- Checking whether the model relies on correct evidence.  
- Verifying whether the retriever itself is strong enough to support the generation.  
- Highlighting failure points: sometimes the retriever fails, sometimes the generator ignores relevant context.  

This closes the loop on the RAG pipeline, giving us a clear picture of **where errors occur** and how the system behaves in practice.


