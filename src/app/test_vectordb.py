from vector_database.text_embedder import TextEmbedder
from vector_database.index_creator import IndexCreator
import shutil

#embedder = TextEmbedder()

texts = [
    "Hollow Knight is a 2017 Metroidvania game developed by Team Cherry.",
    "Beijing is the capital of China, a country in East Asia."
]
#embeddings = embedder.embed_documents(texts)

chunks = [
    {"chunk_id": "hk-1", 
    "content": texts[0], 
    #"embedding": embeddings[0],
    "embedding": [],
     "source": "data/pdfs/Hollow_Knight.pdf", 
     "page_label": "1"},
    {"chunk_id": "cn-1", 
    "content": texts[1], 
    #"embedding": embeddings[1],
    "embedding": [],
     "source": "data/pdfs/China.pdf", 
     "page_label": "1"},
]

# Indexar
indexer = IndexCreator()
indexer.add_chunks(chunks)
