from typing import List
from sentence_transformers import SentenceTransformer
from config import model_config
from logger import get_logger
import torch


class TextEmbedder:
    """
    Generates embeddings from text using a SentenceTransformer model.
    """

    def __init__(
        self,
        model_name: str = model_config.embedding_model,
        embedding_dim: int = model_config.embedding_dim,
    ):
        self.logger = get_logger(name=self.__class__.__name__)
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        self.logger.info(
            f"Loading embedding model {self.model_name} with target dim={self.embedding_dim}"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Loading embedding model {self.model_name} on device={device} with target dim={self.embedding_dim}")
        self.model = SentenceTransformer(self.model_name, device=device)

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text, truncate_dim=self.embedding_dim)
        self.logger.info(
            f"Generated embedding of length {len(embedding)} for single text"
        )
        return embedding.tolist()

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(docs, truncate_dim=self.embedding_dim)
        self.logger.info(
            f"Generated {len(embeddings)} embeddings of length {len(embeddings[0])}"
        )
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query (for retrieval).
        """
        if not text:
            self.logger.warning("Empty query text provided")
            return []

        embedding = self.model.encode([text], truncate_dim=self.embedding_dim)[0]
        self.logger.info(f"Generated embedding of length {len(embedding)} for query")
        return embedding.tolist()
