from vector_database.text_embedder import TextEmbedder
import chromadb
from config import vector_db_config

# --- inicializar cliente y colección ---
client = chromadb.PersistentClient(path=vector_db_config.chroma_path)
collection = client.get_or_create_collection(name=vector_db_config.collection_name)

# --- preparar query ---
embedder = TextEmbedder()
query = "What is the capital of China?"
query_emb = embedder.embed_query(query)

# --- ejecutar consulta ---
results = collection.query(query_embeddings=[query_emb], n_results=2)
import pdb; pdb.set_trace()

print("\n🔎 Query:", query)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"- {doc} (source={meta['source']}, page={meta['page_label']})")