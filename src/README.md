# Running the Project

This project was designed to be lightweight and reproducible. We recommend using **Poetry** for dependency management, but if you prefer a plain `requirements.txt`, you can also install dependencies directly.

> **Note:** All commands below assume your working directory is `src/`.

> If you encounter issues with data paths, double-check your project structure.  
> The setup has been tested successfully under WSL with the repository located in `/home/.../rag-documents-qa/src` and works as expected.

---

## 1) Environment Setup

### Option A — Poetry (recommended)

Create and configure the environment with:

```
make poetry-init
```

This will:

- Create a fresh virtual environment in `.rag-documents/`
- Install all dependencies (including development ones)
- Configure Poetry to install packages directly into this environment

### Option B — requirements.txt

If you do not wish to use Poetry:

```
python3 -m venv .rag-documents
# Activate it (see section 2), then:
pip install -r requirements.txt
```

## 2) Activating the Environment

Activate the virtual environment before running any command.

Linux / macOS

```
source .rag-documents/bin/activate
```

Windows (PowerShell)

```
.\.rag-documents\Scripts\activate
```

## 3) Running the Pipeline (from `src/` with the Makefile)

Before running any command, ensure that the virtual environment is activated.

### Indexing documents into ChromaDB
Run:

```
make index-documents
```

> **Note:** The embedding model [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) is **not included in this repository**.  
> The first time you run the pipeline, it will be downloaded automatically from Hugging Face and stored in your local cache (default path: `~/.cache/huggingface/`).  
> After usage, it is recommended to clear this cache to save disk space.  
> This design choice avoids storing the model inside the repo or relying on private secrets.

This will:
- Load PDFs from your configured path  
- Split them into overlapping chunks  
- Generate embeddings  
- Persist the local vector database (path configurable via `.env`)

### Running evaluation
Run:
```
make evaluate
```

> **Note:** When running `make evaluate`, the generative model [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B#qwen3-highlights) will also be automatically downloaded from Hugging Face on first use and stored in the local cache (`~/.cache/huggingface/`).  
> As with the embedding model, it is recommended to clean the cache after use if you want to free up disk space.  
> This approach avoids committing the model into the repository or relying on external secrets.

This will:
- Build answers with the RAG pipeline  
- Compare them to the gold dataset (`eval.jsonl`)  
- Report metrics (answer quality and % of valid citations)

> **Note:** A reference version of the evaluation metrics (`eval.json`) is already included in the repository as a baseline.

### Where to run commands
All commands must be executed from the `src/` directory.

### If you don’t have `make` on Windows
You can run the underlying Python entry points directly:

```
python app/main.py # indexing
```

```
python app/launch_evaluation.py # evaluation
```

**Warning**:
The pipeline requires that PDFs are available in the `data/pdfs` folder (default location).  
It is strongly recommended **not to modify the default data paths** defined in the configuration (`DataStorageConfig`) unless strictly necessary.

21 minutes - index*