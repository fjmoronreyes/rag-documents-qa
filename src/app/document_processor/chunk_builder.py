import json
import re
import hashlib
from pathlib import Path
from typing import List
from document_processor.chunks import Chunk
from config import data_storage
from logger import get_logger

class ChunkBuilder:
    """
    Converts LangChain Documents into Chunks and optionally saves them locally.
    """

    def __init__(self, output_path: Path = Path(data_storage.chunks_path)):
        self.logger = get_logger(name=self.__class__.__name__)
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

    def build_chunks(self, documents: List, save: bool = True) -> List[Chunk]:
        """
        Convert a list of LangChain Documents into Chunks.

        Args:
            documents (List): List of LangChain Document objects.
            save (bool): Whether to save the chunks locally. Defaults to True.

        Returns:
            List[Chunk]: List of Chunk dataclass instances.
        """
        chunks = []
        for doc in documents:
            chunk = Chunk(
                chunk_id=self._make_chunk_id(doc.page_content),
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", -1),
                page_label=doc.metadata.get("page_label"),
                metadata=doc.metadata,
                embedding=None,
            )
            chunks.append(chunk)

        self.logger.info(f"Built {len(chunks)} chunks")

        if save and chunks:
            filename = self._derive_filename(chunks[0])
            self.save_chunks(chunks, filename)

        return chunks

    def save_chunks(self, chunks: List[Chunk], filename: str):
        """
        Save chunks to a JSON file in the configured chunks directory.

        Args:
            chunks (List[Chunk]): Chunks to save.
            filename (str): Output filename (without extension).
        """
        path = self.output_path / f"{filename}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([chunk.__dict__ for chunk in chunks], f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved {len(chunks)} chunks to {path}")

    def _derive_filename(self, chunk: Chunk) -> str:
        """
        Derive a filename from chunk metadata.

        Args:
            chunk (Chunk): A chunk whose metadata will be used.

        Returns:
            str: Sanitized filename without extension.
        """
        title = chunk.metadata.get("title")
        if title:
            base = title
        else:
            base = Path(chunk.source).stem
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", base).strip("_")

    @staticmethod
    def _make_chunk_id(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()