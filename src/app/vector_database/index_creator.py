from typing import List, Dict, Any
import chromadb
from logger import get_logger
from config import vector_db_config


class IndexCreator:
    """
    Responsible only for creating and indexing chunks into ChromaDB.
    """

    def __init__(
        self,
        collection_name: str = vector_db_config.collection_name,
        persist_dir: str = vector_db_config.chroma_path,
    ):
        """
        Initialize Chroma client and collection.
        """
        self.logger = get_logger(name=self.__class__.__name__)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.logger.info(
            f"IndexCreator initialized for collection '{collection_name}' at {persist_dir}"
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add pre-embedded chunks into Chroma.
        Each chunk dict must include:
          - chunk_id
          - content
          - embedding
          - source
          - page_label
        """
        chunks = self._deduplicate_chunks(chunks)
        if not chunks:
            self.logger.info("No new chunks to index (all were duplicates)")
            return
        ids = [c["chunk_id"] for c in chunks]
        docs = [c["content"] for c in chunks]
        embs = [c["embedding"] for c in chunks]
        metas = [{"source": c["source"], "page_label": c["page_label"]} for c in chunks]
        self.collection.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
        self.logger.info(f"Indexed {len(chunks)} new chunks into Chroma collection")

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove chunks whose IDs already exist in the collection.
        """
        ids_to_check = [c["chunk_id"] for c in chunks]
        existing = self.collection.get(ids=ids_to_check)
        existing_ids = set(existing["ids"]) if existing and "ids" in existing else set()
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        skipped = len(chunks) - len(new_chunks)
        if skipped > 0:
            self.logger.info(f"Skipped {skipped} duplicate chunks")
        return new_chunks
