import json
import re
import chromadb
from typing import List
from difflib import SequenceMatcher

from config import data_storage, vector_db_config
from logger import get_logger


class RagEvaluator:
    """RAG evaluator that generates answers and computes evaluation metrics."""

    def __init__(self):
        self.logger = get_logger(name="RagEvaluator")
        self.client = chromadb.PersistentClient(path=vector_db_config.chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=vector_db_config.collection_name
        )

    def _extract_chunk_ids_from_answer(self, answer: str) -> List[str]:
        """Extract chunk_ids from the generated answer."""
        return re.findall(r"chunk_id=([a-f0-9]{64})", answer)

    def generate_answers(self):
        """Run pipeline on gold dataset and save answers.json."""
        from vector_database.text_embedder import TextEmbedder
        from genai.answer_generator import AnswerGenerator

        embedder = TextEmbedder()
        generator = AnswerGenerator()

        with open(data_storage.gold_path, "r", encoding="utf-8") as f:
            gold_data = json.load(f)

        answers = []
        for item in gold_data:
            query = item["query"]
            self.logger.info(f"Processing query: {query}")

            query_emb = embedder.embed_query(query)
            results = self.collection.query(query_embeddings=[query_emb], n_results=3)

            retrieved_passages = [
                {
                    "chunk_id": cid,
                    "content": doc,
                    "source": meta["source"],
                    "page_label": meta["page_label"],
                }
                for doc, meta, cid in zip(
                    results["documents"][0], results["metadatas"][0], results["ids"][0]
                )
            ]

            context = "\n\n".join(
                f"{p['content']}\n(Source: {p['source']}, page={p['page_label']}, chunk_id={p['chunk_id']})"
                for p in retrieved_passages
            )

            answer = generator.generate(query=query, context=context)
            self.logger.info(f"Generated answer: {answer}")

            chunk_ids = self._extract_chunk_ids_from_answer(answer)
            source_passages = []
            if chunk_ids:
                retrieved = self.collection.get(ids=chunk_ids)
                for i, cid in enumerate(retrieved["ids"]):
                    source_passages.append(
                        {
                            "chunk_id": cid,
                            "content": retrieved["documents"][i],
                            "source": retrieved["metadatas"][i]["source"],
                            "page_label": retrieved["metadatas"][i]["page_label"],
                        }
                    )

            answers.append(
                {
                    "query": query,
                    "inferenced_answer": answer,
                    "retrieved_passages": retrieved_passages,
                    "source_passages": source_passages,
                }
            )

        with open(data_storage.answer_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Answers saved to {data_storage.answer_path}")

    def evaluate_answers(self):
        """
        Compare model answers (answers.json) against gold references (gold.json)
        and save evaluation metrics (eval.json).

        This evaluation computes three per-query metrics and aggregates them
        into overall scores.

        Per-query metrics:
        - answer_score:
            Measures how similar the model's inferenced_answer is to the gold
            expected_answer. Similarity is computed using difflib.SequenceMatcher,
            which returns a ratio between 0.0 (completely different) and 1.0
            (identical). This is a character-level similarity score, not a semantic
            similarity; it evaluates whether the model’s output matches the wording
            of the reference answer.

        - has_valid_citation:
            Evaluates whether the model correctly cited at least one relevant passage.
            A citation is considered valid if the chunk_id of any gold source_passage
            (the reference passages supporting the expected_answer) appears in the
            model’s cited source_passages.
            This metric answers: “Did the model explicitly ground its answer in the
            correct evidence?”

        - retrieval_recall:
            Evaluates whether the retriever retrieved at least one relevant passage,
            regardless of whether the model cited it. A retrieval is considered a
            hit if the chunk_id of any gold source_passage appears in the model’s
            retrieved_passages (the top-k passages returned by the vector database).
            This metric answers: “Did the retriever bring the right evidence into
            the candidate pool?”

        Aggregated metrics:
        - average_answer_score:
            The arithmetic mean of all answer_score values across queries. It
            reflects the average similarity between generated and expected answers
            over the entire evaluation set.

        - valid_citation_rate:
            The proportion of queries where has_valid_citation=True. It is computed
            as (# of queries with at least one valid citation) / (total # of queries).
            This measures how consistently the model cites correct sources.

        - retrieval_recall_rate:
            The proportion of queries where retrieval_recall=True. It is computed as
            (# of queries where at least one gold passage was retrieved) / (total #
            of queries). This measures how often the retriever provides the model
            with the necessary evidence, even if the model does not cite it.
        """
        with open(data_storage.gold_path, "r", encoding="utf-8") as f:
            gold_data = json.load(f)
        with open(data_storage.answer_path, "r", encoding="utf-8") as f:
            answers_data = json.load(f)

        results = []
        total_score, valid_citations, retrieval_hits = 0, 0, 0

        for gold, ans in zip(gold_data, answers_data):
            expected = gold["expected_answer"]
            predicted = ans["inferenced_answer"]

            score = SequenceMatcher(None, expected.lower(), predicted.lower()).ratio()
            total_score += score

            gold_ids = {p["chunk_id"] for p in gold["source_passages"]}
            cited_ids = {p["chunk_id"] for p in ans["source_passages"]}
            retrieved_ids = {p["chunk_id"] for p in ans["retrieved_passages"]}

            has_valid_citation = bool(gold_ids & cited_ids)
            recall_hit = bool(gold_ids & retrieved_ids)

            if has_valid_citation:
                valid_citations += 1
            if recall_hit:
                retrieval_hits += 1

            results.append(
                {
                    "query": gold["query"],
                    "expected_answer": expected,
                    "inferenced_answer": predicted,
                    "answer_score": score,
                    "has_valid_citation": has_valid_citation,
                    "retrieval_recall": recall_hit,
                }
            )

        metrics = {
            "average_answer_score": total_score / len(gold_data),
            "valid_citation_rate": valid_citations / len(gold_data),
            "retrieval_recall_rate": retrieval_hits / len(gold_data),
        }

        output = {"results": results, "metrics": metrics}
        eval_path = data_storage.eval_path
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Evaluation metrics saved to {eval_path}")


def main():
    """Entry point: generate answers and then evaluate them."""
    evaluator = RagEvaluator()
    evaluator.generate_answers()
    evaluator.evaluate_answers()


if __name__ == "__main__":
    main()
