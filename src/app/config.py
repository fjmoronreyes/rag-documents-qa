import os
from typing import Optional

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

class DataStorageConfig(BaseModel):
    documents_path: str = os.getenv("DOCUMENTS_PATH", "data/pdfs")
    chunks_path: str = os.getenv("CHUNKS_PATH", "data/chunks")
    eval_path: str = os.getenv("EVAL_PATH", "data/metrics/eval.json")
    answer_path: str = os.getenv("ANSWER_PATH", "data/metrics/answers.json")

class ModelConfig(BaseModel):
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    embedding_dim: int = os.getenv("EMBEDDING_DIM", 512)
    generative_model: str = os.getenv("GENERATIVE_MODEL", "Qwen/Qwen3-0.6B")

class ChunkConfig(BaseModel):
    chunk_size: int = os.getenv("CHUNK_SIZE", 500)
    chunk_overlap: int = os.getenv("CHUNK_OVERLAP", 50)

class VectorDBConfig(BaseModel):
    chroma_path: str = os.getenv("CHROMA_PATH", "data/chroma_store")
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")

data_storage = DataStorageConfig()
model_config = ModelConfig()
chunk_config = ChunkConfig()
vector_db_config = VectorDBConfig()