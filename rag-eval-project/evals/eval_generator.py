
"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: Are the claims in the generated answer supported by
the context provided?

ISOLATION: Uses the golden context rather than the retriever's output,
so a low score points toward a generator issue.

Run:
    uv run python -m evals.eval_generator
"""

import json
from pathlib import Path
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import GeminiModel

from src.generator import generate

# Keep these for later if needed:
# from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = str(
    Path(__file__).resolve().parent.parent
    / "goldens"
    / "faithfulness_dataset.json"
)

JUDGE_MODEL = GeminiModel(
    model="gemini-3.5-flash-lite",
    temperature=0,
)

THRESHOLD = 0.7


def summarize_results(result):
    test_results = getattr(result, "test_results", None) or []
    grouped = {}

    for test_result in test_results:
        metrics = (
            getattr(test_result, "metrics_data", None)
            or getattr(test_result, "metrics", None)
            or []
        )

        for metric in metrics:
            name = getattr(metric, "name", "unknown")
            entry = grouped.setdefault(
                name, {"scores": [], "passed": 0, "total": 0}
            )

            score = getattr(metric, "score", None)
            if score is not None:
                entry["scores"].append(float(score))

            entry["total"] += 1
            if getattr(metric, "success", False):
                entry["passed"] += 1

    summary = {}

    for name, entry in grouped.items():
        scores = entry["scores"]
        total = entry["total"]

        summary[name] = {
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "pass_rate": 100 * entry["passed"] / total if total else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "n": total,
        }

    return summary


def run():
    # 1. Load the faithfulness golden dataset
    # with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    #     goldens = json.load(f)
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        goldens = json.load(f)[:2]

    # For a small initial test, uncomment the next line:
    # goldens = goldens[:2]

    # 2. Run the generator using the known-good golden context
    test_cases = []

    for g in goldens:
        context = g["ideal_context"]
        answer = generate(g["query"], context)

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,
                retrieval_context=context,
                # expected_output is not required for FaithfulnessMetric
            )
        )

    # 3. Define the evaluation metrics
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
    ]

    # 4. Run evaluation
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=1, run_async=False),
        hyperparameters={
            "generator_model": "gemini-3.5-flash-lite",
            "judge_model": "gemini-3.5-flash-lite",
            "threshold": THRESHOLD,
            "golden_set": GOLDEN_PATH,
        },
    )

    # Keep for later if you restore the evaluation harness:
    # return summarize_by_metric(result)

    # return result
    return summarize_results(result)


if __name__ == "__main__":
    run()
    # Later, if you restore the harness:
    # print_summary("generator", run())
