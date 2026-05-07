"""Registry of math-reasoning eval datasets used by math_verify_eval.

Each entry pins a specific HuggingFace dataset ID and the sampling defaults
(n_samples / temperature / top_p) that match published evaluation conventions
(greedy@1 for MATH-500/AMC23/Minerva, avg@8 with T=0.6 for AIME).
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DatasetSpec:
    hf_id: str
    split: str
    problem_field: str
    answer_field: str
    n_samples: int
    temperature: float
    top_p: float


REGISTRY: dict[str, DatasetSpec] = {
    "MATH-500": DatasetSpec(
        hf_id="HuggingFaceH4/MATH-500",
        split="test",
        problem_field="problem",
        answer_field="answer",
        n_samples=1,
        temperature=0.0,
        top_p=1.0,
    ),
    "AIME-24": DatasetSpec(
        hf_id="HuggingFaceH4/aime_2024",
        split="train",
        problem_field="problem",
        answer_field="answer",
        n_samples=8,
        temperature=0.6,
        top_p=0.95,
    ),
    "AIME-25": DatasetSpec(
        hf_id="yentinglin/aime_2025",
        split="train",
        problem_field="problem",
        answer_field="answer",
        n_samples=8,
        temperature=0.6,
        top_p=0.95,
    ),
    "AMC23": DatasetSpec(
        hf_id="math-ai/amc23",
        split="test",
        problem_field="question",
        answer_field="answer",
        n_samples=1,
        temperature=0.0,
        top_p=1.0,
    ),
    "Minerva": DatasetSpec(
        hf_id="math-ai/minervamath",
        split="test",
        problem_field="question",
        answer_field="answer",
        n_samples=1,
        temperature=0.0,
        top_p=1.0,
    ),
}


def known_dataset_names() -> list[str]:
    """Return the canonical ordering of dataset names for CLI defaults."""
    return ["MATH-500", "AIME-24", "AIME-25", "AMC23", "Minerva"]


def load_eval_dataset(
    name: str,
    tokenizer,
    prompt_template: str,
) -> list[dict]:
    """Load a registered eval dataset and wrap each problem in
    `boxed.prompt` + chat template (same wrapping as run_rl.py training).

    Returns a list of dicts: {"prompt": str, "gold_answer": str,
                               "n_samples": int, "temperature": float,
                               "top_p": float}.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown eval dataset: {name!r}. Known: {sorted(REGISTRY.keys())}"
        )

    from datasets import load_dataset

    spec = REGISTRY[name]
    raw = load_dataset(spec.hf_id, split=spec.split)

    out = []
    for example in raw:
        problem_text = example[spec.problem_field]
        gold_answer = example[spec.answer_field]
        with_template = prompt_template.replace("{question}", problem_text)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": with_template}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out.append(
            {
                "prompt": prompt,
                "gold_answer": str(gold_answer),
                "n_samples": spec.n_samples,
                "temperature": spec.temperature,
                "top_p": spec.top_p,
            }
        )
    return out
