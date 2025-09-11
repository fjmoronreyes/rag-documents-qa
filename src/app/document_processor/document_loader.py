from pathlib import Path
from typing import Iterator
from config import data_storage
from logger import get_logger

class DocumentLoader:
    """
    Loader for retrieving PDF documents from a given directory.
    """

    def __init__(self, base_path: Path = Path(data_storage.documents_path)):
        """
        Initialize a DocumentLoader.

        Args:
            base_path (Path): Path to the directory containing PDF files.
                              Defaults to the documents path from config.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
        """
        self.logger = get_logger(name=self.__class__.__name__)
        self.base_path = base_path

        if not self.base_path.exists():
            self.logger.error(f"Directory not found: {self.base_path}")
            raise FileNotFoundError(f"Directory not found: {self.base_path}")
        if not self.base_path.is_dir():
            self.logger.error(f"Path is not a directory: {self.base_path}")
            raise NotADirectoryError(f"Path is not a directory: {self.base_path}")

    def load_documents(self) -> list[Path]:
        """
        Load all PDF documents from the configured directory.

        Returns:
            list[Path]: A list of paths to the PDF files found.
        """
        self.logger.info(f"Loading documents from {self.base_path}")
        pdfs = sorted(self.base_path.glob("*.pdf")) + sorted(self.base_path.glob("*.PDF"))
        if not pdfs:
            self.logger.warning(f"No PDF files found in {self.base_path}")
        else:
            self.logger.info(f"Found {len(pdfs)} PDF(s) in {self.base_path}")
        return pdfs

    def iter_documents(self) -> Iterator[Path]:
        """
        Iterate over all PDF documents in the directory.

        Yields:
            Path: Path object for each PDF file found.
        """
        yield from self.load_documents()
