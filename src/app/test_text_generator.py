from genai.answer_generator import AnswerGenerator

# --- inicializar generador ---
generator = AnswerGenerator()

# --- query simulada ---
query = "When was Hollow Knight released?"

# --- chunk recuperado simulado ---
retrieved_chunk = {
    "chunk_id": "922883d27d3bf3745decdda7b72518f97c96df11fae10b052c9527500798d89b",
    "content": (
        "Hollow Knight is a 2017 Metroidvania video game "
        "developed and published by Australian independent developer Team Cherry. "
        "It was released for Windows, Linux, and macOS in early 2017 and for the "
        "Nintendo Switch, PlayStation 4, and Xbox One in 2018. "
        "After release, Team Cherry supported the game with four free expansions."
    ),
    "source": "data/pdfs/Hollow_Knight.pdf",
    "page_label": "1",
}

# --- construir contexto con referencia completa ---
context = (
    f"{retrieved_chunk['content']}\n"
    f"(Source: {retrieved_chunk['source']}, page {retrieved_chunk['page_label']}, "
    f"chunk_id={retrieved_chunk['chunk_id']})"
)

# --- generar respuesta ---
answer = generator.generate(query=query, context=context)

print("\n🔎 Query:", query)
print("📖 Context:", context)
print("🤖 Answer:", answer)
