
# eval_retriever.py
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import json
from pathlib import Path

from dotenv import load_dotenv
from src.reranker import RerankingRetriever
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import GeminiModel

load_dotenv()

GOLDEN_PATH = str(
    Path(__file__).resolve().parent.parent
    / "goldens"
    / "retriever_goldens.json"
)

JUDGE_MODEL = GeminiModel(
    model="gemini-3.5-flash-lite",
    temperature=0,
)

THRESHOLD = 0.7


def summarize_results(result):
    test_results = getattr(result, "test_results", None) or []

    summary = {}

    for test_result in test_results:
        metrics = (
            getattr(test_result, "metrics_data", None)
            or getattr(test_result, "metrics", None)
            or []
        )

        for metric in metrics:
            name = getattr(metric, "name", "unknown")
            entry = summary.setdefault(
                name,
                {"scores": [], "passed": 0, "total": 0},
            )

            score = getattr(metric, "score", None)
            if score is not None:
                entry["scores"].append(float(score))

            entry["total"] += 1
            if getattr(metric, "success", False):
                entry["passed"] += 1

    output = {}

    for name, entry in summary.items():
        scores = entry["scores"]
        total = entry["total"]

        output[name] = {
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "pass_rate": 100.0 * entry["passed"] / total if total else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "n": total,
        }

    return output


def run(retriever):
    # 1. Load the golden dataset (currently testing only 2 cases)
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        # goldens = json.load(f)
        goldens = json.load(f)[:2]


    retriever = RerankingRetriever()


    # 2. Retrieve documents and create test cases
    test_cases = []

    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. Define retrieval evaluation metrics
    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
            async_mode=False
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
            async_mode=False
        ),
    ]

    # 4. Run evaluation
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=1, run_async=False),
        hyperparameters={
            "retriever": "reranker",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 3,
            "judge_model": "gemini-3.5-flash-lite",
            "golden_set": GOLDEN_PATH,
        },
   )

    # return result
    return summarize_results(result)


if __name__ == "__main__":
    run(RerankingRetriever())
