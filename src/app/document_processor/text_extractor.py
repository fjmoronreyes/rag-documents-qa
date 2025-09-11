from pathlib import Path
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from config import model_config, chunk_config
from logger import get_logger

class TextExtractor:
    """
    Extractor for reading and splitting text from a single PDF file.
    """

    def __init__(
        self,
        model_name: str = model_config.embedding_model,
        chunk_size: int = chunk_config.chunk_size,
        chunk_overlap: int = chunk_config.chunk_overlap,
    ):
        """
        Initialize a TextExtractor with a tokenizer and splitter.

        Args:
            model_name (str): HuggingFace model identifier for the tokenizer. Defaults to config value.
            chunk_size (int): Maximum number of tokens per chunk. Defaults to config value.
            chunk_overlap (int): Number of overlapping tokens between chunks. Defaults to config value.
        """
        self.logger = get_logger(name=self.__class__.__name__)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def extract_text(self, file_path: str | Path) -> list[Document]:
        """
        Extract and split text from a given PDF file into chunks.

        Args:
            file_path (str | Path): Path to the PDF file.

        Returns:
            list[Document]: A list of LangChain Document objects representing text chunks.

        Raises:
            FileNotFoundError: If the specified PDF file does not exist.
        """
        p = Path(file_path)
        if not p.is_file():
            self.logger.error(f"File not found: {p}")
            raise FileNotFoundError(f"File not found: {p}")

        self.logger.info(
            f"Extracting text from {p} with model={self.model_name}, "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )
        loader = PyPDFLoader(str(p))
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)
        self.logger.info(f"Generated {len(chunks)} chunks from {p.name}")
        return chunks
