from document_processor.document_loader import DocumentLoader
from document_processor.text_extractor import TextExtractor
from document_processor.chunk_builder import ChunkBuilder

loader = DocumentLoader()
extractor = TextExtractor()
builder = ChunkBuilder()

for pdf in loader.iter_documents():
    docs = extractor.extract_text(pdf)
    chunks = builder.build_chunks(docs, save=True)