"""Post-training math-verify evaluation driver.

Used by run_rl.py (after the GRPO loop) and by eval_rl.py (standalone).
Generates with vLLM, grades with HuggingFace math-verify.
"""

import datetime
import time
from typing import Callable, Optional

from eval_datasets import REGISTRY, load_eval_dataset


def _build_or_get_llm(model, vllm_kwargs):
    """Return an LLM-like object exposing a `.generate(prompts, sampling_params)` method.

    If `vllm_kwargs` contains "llm", reuse it (for in-loop reuse from run_rl.py
    or for tests). Otherwise construct a fresh vLLM `LLM` from `vllm_kwargs`.
    """
    if vllm_kwargs is None:
        vllm_kwargs = {}
    if "llm" in vllm_kwargs:
        return vllm_kwargs["llm"], False  # not owned (caller manages)
    import os

    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM

    return LLM(**vllm_kwargs), True  # owned (we constructed it)


def _verify_one(gold: str, model_text: str) -> tuple[bool, bool, bool]:
    """Return (is_correct, was_unparseable, grader_raised).

    Uses `math_verify.parse` + `math_verify.verify`. An empty parse counts as
    unparseable (and incorrect). A grader exception counts as `grader_raised`
    (and incorrect).
    """
    from math_verify import parse, verify

    try:
        pred = parse(model_text)
    except Exception:
        return False, True, True
    if not pred:
        return False, True, False

    try:
        gold_parsed = parse(gold)
        ok = bool(verify(gold_parsed, pred))
    except Exception:
        return False, False, True
    return ok, False, False


def math_verify_eval(
    model,                               # may be None when vllm_kwargs["llm"] is supplied
    tokenizer,
    datasets: list[str],
    *,
    n_samples_override: Optional[int] = None,
    temperature_override: Optional[float] = None,
    max_tokens: int = 2048,
    prompt_template_path: Optional[str] = "boxed.prompt",
    vllm_kwargs: Optional[dict] = None,
    _dataset_loader: Callable = load_eval_dataset,  # injected for tests
) -> dict:
    """Run math-verify eval over the requested dataset names.

    Returns:
        {
          "datasets": {name: {"accuracy", "n_correct", "n_total",
                              "n_samples_per_problem", "temperature",
                              "max_tokens", "wall_time_sec",
                              "n_unparseable", "n_grader_errors"}},
          "errors": {name: reason_str},
          "math_verify_version": str,
          "timestamp": ISO8601 str,
        }
    """
    from importlib import metadata as _metadata
    from vllm import SamplingParams

    template = ""
    if prompt_template_path is not None:
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            template = f.read().strip()

    llm, owned = _build_or_get_llm(model, vllm_kwargs)
    try:
        try:
            mv_version = _metadata.version("math-verify")
        except _metadata.PackageNotFoundError:
            mv_version = "unknown"

        out = {
            "datasets": {},
            "errors": {},
            "math_verify_version": mv_version,
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

        for name in datasets:
            try:
                items = _dataset_loader(name, tokenizer, template)
            except Exception as exc:
                out["errors"][name] = f"dataset load failed: {exc!r}"
                continue

            spec = REGISTRY[name]
            n_samples = (
                n_samples_override if n_samples_override is not None else spec.n_samples
            )
            temperature = (
                temperature_override
                if temperature_override is not None
                else spec.temperature
            )
            top_p = spec.top_p

            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n_samples,
            )

            prompts = [item["prompt"] for item in items]
            gold_answers = [item["gold_answer"] for item in items]

            t0 = time.time()
            outputs = llm.generate(prompts, sampling_params)
            wall = time.time() - t0

            n_correct = 0
            n_unparseable = 0
            n_grader_errors = 0
            n_total = 0

            for problem_output, gold in zip(outputs, gold_answers):
                for completion in problem_output.outputs:
                    n_total += 1
                    ok, unparseable, raised = _verify_one(gold, completion.text)
                    if ok:
                        n_correct += 1
                    if unparseable:
                        n_unparseable += 1
                    if raised:
                        n_grader_errors += 1

            out["datasets"][name] = {
                "accuracy": n_correct / n_total if n_total else 0.0,
                "n_correct": n_correct,
                "n_total": n_total,
                "n_samples_per_problem": n_samples,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "wall_time_sec": wall,
                "n_unparseable": n_unparseable,
                "n_grader_errors": n_grader_errors,
            }

        return out
    finally:
        if owned:
            # vLLM doesn't expose a clean shutdown in 0.10.2; rely on GC.
            del llm
