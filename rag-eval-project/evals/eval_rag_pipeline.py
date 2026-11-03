
# evals/eval_rag_pipeline.py

import json
from pathlib import Path
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.models import GeminiModel

from src.rag_pipeline import RagPipeline

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
    # temperature is omitted because this model uses fixed sampling defaults
)

THRESHOLD = 0.7


def run(rag):
    # 1. Load the golden dataset
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        goldens = json.load(f)

    # For a small initial test, uncomment the next line:
    goldens = goldens[:2]

    # 2. Run the complete RAG pipeline for each query
    test_cases = []

    for g in goldens:
        result = rag.invoke(g["query"])  # retrieve -> rerank -> generate

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],
                retrieval_context=result["context"],
            )
        )

    # 3. Define the three evaluation metrics
    metrics = [
        ContextualRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
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
        # hyperparameters={
        #     "pipeline": "rag_pipeline",
        #     "generator_model": "gemini-3.5-flash-lite",
        #     "judge_model": "gemini-3.5-flash-lite",
        #     "threshold": THRESHOLD,
        #     "golden_set": GOLDEN_PATH,
        # },
    )

    # Restore later if you want the custom summary:
    # return summarize_by_metric(result)

    return result


def run_local():
    """Build the pipeline and run evaluation."""
    return run(RagPipeline())


if __name__ == "__main__":
    run_local()
    # Restore later if you want the custom summary:
    # print_summary("rag_pipeline", run_local())


