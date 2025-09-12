from transformers import AutoModelForCausalLM, AutoTokenizer
from config import model_config
from logger import get_logger
import torch


class AnswerGenerator:
    """
    Generates answers from a query and retrieved context using a causal LM (e.g., Qwen3-0.6B).
    Ensures answers are grounded with citations to the provided context.
    """

    def __init__(self, model_name: str = model_config.generative_model):
        self.logger = get_logger(name=self.__class__.__name__)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            #low_cpu_mem_usage=True
        )
        self.logger.info(f"Loaded generative model: {self.model_name}")

    def generate(self, query: str, context: str, max_new_tokens: int = 512) -> str:
        """
        Generate an answer grounded in the retrieved context, with citations.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    "You are a helpful assistant answering questions based strictly on the provided context. "
                    "Do not use external knowledge. "
                    "When you provide an answer, you must ALWAYS include explicit citations from the context. "
                    "Each citation must clearly reference:\n"
                    "- The exact passage used (quote or paraphrase).\n"
                    "- The page number (from 'page_label').\n"
                    "- The chunk_id.\n\n"
                    "If the answer cannot be derived from the context, reply with: 'The context does not provide this information.'\n\n"
                    f"Question:\n{query}\n\n"
                    f"Context:\n{context}\n\n"
                    "Answer (with citations):"
                ),
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        return response
