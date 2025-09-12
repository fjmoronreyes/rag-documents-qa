import json
import re
import chromadb

from config import data_storage, vector_db_config
from vector_database.text_embedder import TextEmbedder
from genai.answer_generator import AnswerGenerator
from logger import get_logger


def extract_chunk_ids_from_answer(answer: str):
    """Extract all chunk_ids from the generated answer text."""
    return re.findall(r"chunk_id=([a-f0-9]{64})", answer)


def main():
    """Run evaluation pipeline and save model answers with retrieved and cited passages."""
    logger = get_logger(name="Evaluation")

    with open(data_storage.eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    client = chromadb.PersistentClient(path=vector_db_config.chroma_path)
    collection = client.get_or_create_collection(name=vector_db_config.collection_name)

    embedder = TextEmbedder()
    generator = AnswerGenerator()

    answers = []

    for item in eval_data:
        query = item["query"]
        logger.info(f"Processing query: {query}")

        query_emb = embedder.embed_query(query)
        results = collection.query(query_embeddings=[query_emb], n_results=3)

        retrieved_passages = []
        for doc, meta, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0]
        ):
            retrieved_passages.append({
                "chunk_id": cid,
                "content": doc,
                "source": meta["source"],
                "page_label": meta["page_label"]
            })

        context = "\n\n".join(
            f"{p['content']}\n(Source: {p['source']}, page={p['page_label']}, chunk_id={p['chunk_id']})"
            for p in retrieved_passages
        )

        answer = generator.generate(query=query, context=context)
        logger.info(f"Generated answer: {answer}")

        chunk_ids = extract_chunk_ids_from_answer(answer)

        source_passages = []
        if chunk_ids:
            retrieved = collection.get(ids=chunk_ids)
            for i, cid in enumerate(retrieved["ids"]):
                source_passages.append({
                    "chunk_id": cid,
                    "content": retrieved["documents"][i],
                    "source": retrieved["metadatas"][i]["source"],
                    "page_label": retrieved["metadatas"][i]["page_label"]
                })

        answers.append({
            "query": query,
            "inferenced_answer": answer,
            "retrieved_passages": retrieved_passages,
            "source_passages": source_passages
        })

    with open(data_storage.answer_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation completed. Results saved in {data_storage.answer_path}")


if __name__ == "__main__":
    main()
