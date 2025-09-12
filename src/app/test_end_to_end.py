from vector_database.text_embedder import TextEmbedder
from genai.answer_generator import AnswerGenerator
import chromadb
from config import vector_db_config

# --- inicializar cliente y colección ---
client = chromadb.PersistentClient(path=vector_db_config.chroma_path)
collection = client.get_or_create_collection(name=vector_db_config.collection_name)

# --- inicializar componentes ---
embedder = TextEmbedder()
generator = AnswerGenerator()

# --- preparar query ---
query = "When was Hollow Knight released?"
query_emb = embedder.embed_query(query)

# --- ejecutar consulta en Chroma ---
results = collection.query(query_embeddings=[query_emb], n_results=2)

print("\n🔎 Query:", query)

# --- construir contexto con referencias ---
chunks = []
for doc, meta, cid in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["ids"][0]
):
    print(f"- {doc[:100]}... (source={meta['source']}, page={meta['page_label']}, chunk_id={cid})")
    chunks.append(
        f"{doc}\n(Source: {meta['source']}, page={meta['page_label']}, chunk_id={cid})"
    )

context = "\n\n".join(chunks)

# --- generar respuesta final ---
answer = generator.generate(query=query, context=context)

print("\n🤖 Final Answer:\n", answer)
