from logger import get_logger
from document_processor.document_loader import DocumentLoader
from document_processor.text_extractor import TextExtractor
from document_processor.chunk_builder import ChunkBuilder
from vector_database.text_embedder import TextEmbedder
from vector_database.index_creator import IndexCreator
import gc


class IndexingPipeline:
    """
    Orchestrates the full indexing pipeline with batch processing:
    1. Load PDFs
    2. Extract text
    3. Build chunks
    4. Generate embeddings in batches
    5. Index into Chroma
    """

    def __init__(self, batch_size: int = 8):
        self.logger = get_logger(name=self.__class__.__name__)
        self.loader = DocumentLoader()
        self.extractor = TextExtractor()
        self.builder = ChunkBuilder()
        self.embedder = TextEmbedder()
        self.indexer = IndexCreator()
        self.batch_size = batch_size

    def run(self):
        for pdf in self.loader.iter_documents():
            self.logger.info(f"Processing PDF: {pdf}")
            docs = self.extractor.extract_text(pdf)
            chunks = self.builder.build_chunks(docs, save=True)
            self._process_in_batches(chunks)
        self.logger.info("Indexing pipeline completed successfully")

    def _process_in_batches(self, chunks):
        """
        Process chunks in batches of self.batch_size:
        generate embeddings and index each batch.
        """
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            texts = [c.content for c in batch]
            embeddings = self.embedder.embed_documents(texts)
            for c, emb in zip(batch, embeddings):
                c.embedding = emb
            self.indexer.add_chunks([c.__dict__ for c in batch])
            current_batch = i // self.batch_size + 1
            self.logger.info(f"Indexed batch {current_batch}/{total_batches}")
            del texts, embeddings, batch
            gc.collect()

if __name__ == "__main__":
    pipeline = IndexingPipeline()
    pipeline.run()