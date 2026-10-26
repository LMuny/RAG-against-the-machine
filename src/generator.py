"""Generate grounded answers with Qwen3-0.6B through vLLM."""

import re
from typing import Any
from src.models import MinimalSource

SYSTEM_PROMPT: str = (
    "You answer questions about the vLLM repository. "
    "Use only the provided context. "
    "If the answer is not in the context, say that you don't know. "
    "Answer in 2 to 4 sentences."
)


class Generator:
    """Generate answers from retrieved contexts with a local model."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B",
                 max_model_len: int = 8192,
                 max_tokens: int = 512) -> None:
        """Load the model (vllm is imported here, it is slow to import)."""
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_name, max_model_len=max_model_len)
        self.tokenizer = self.llm.get_tokenizer()
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8,
                                              max_tokens=max_tokens)

    def build_prompt(self, question: str,
                     contexts: list[tuple[MinimalSource, str]]) -> str:
        """Build a chat prompt, adding sources while they fit the budget."""
        budget = self.max_model_len - self.max_tokens - 200
        budget -= len(self.tokenizer.encode(SYSTEM_PROMPT + question))
        context_text = ""
        for number, (source, text) in enumerate(contexts, start=1):
            block = f"[{number}] {source.file_path}\n{text}\n\n"
            block_tokens = len(self.tokenizer.encode(block))
            if block_tokens > budget:
                break
            budget -= block_tokens
            context_text += block
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{context_text}Question: {question}"},
        ]
        return str(self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False))

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate one answer per prompt, all in a single batch."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        answers: list[str] = []
        for output in outputs:
            text = output.outputs[0].text
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            answers.append(text.strip())
        return answers
