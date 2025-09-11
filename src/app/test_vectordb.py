from vector_database.text_embedder import TextEmbedder

embedder = TextEmbedder()
texts = [
    "Hollow Knight is a Metroidvania action-adventure game.",
    "The capital of China is Beijing."
]

embeddings = embedder.embed_documents(texts)

print(f"Num embeddings: {len(embeddings)}")
print(f"First embedding length: {len(embeddings[0])}")
print(embeddings[0][:10])  # preview
