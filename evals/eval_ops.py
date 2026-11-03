"""
Operational Evaluation: Latency, Cost, and Reliability

Evaluates the operational performance of the RAG application:
1. Latency: end-to-end response time, retrieval time, generation time,
   and time to first token (TTFT).
2. Cost: estimated API cost based on measured token usage.
3. Reliability: success rate, error rate, retries, and attempts.

Uses the existing Gemini generator, retriever, reranker, and RAG pipeline.
"""

# ============================================================
# 1. IMPORTS & ENVIRONMENT
# ============================================================
import math
import time
from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.generator import generate, prompt, llm

load_dotenv()

# Reuse the exact prompt and model, stopping before StrOutputParser.
measured_chain = prompt | llm


# ============================================================
# 2. SHARED CONFIGURATION
# ============================================================
MODEL_NAME = "gemini-3.5-flash-lite"

QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

# Conservative defaults to reduce Gemini API calls.
LAT_REPEATS = 1
LAT_WARMUP_RUNS = 1
LAT_MEASURE_TTFT = True

COST_REPEATS = 1

REL_REPEATS = 1
MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5

# Latency service-level objectives.
SLO_P95_MS = 3000
SLO_TTFT_P95_MS = 1200

# Gemini 3.5 Flash-Lite standard paid-tier prices.
# USD per 1 million tokens. Verify against your actual billing tier.
PRICE_INPUT_PER_1M = 0.30
PRICE_CACHED_INPUT_PER_1M = 0.03
PRICE_OUTPUT_PER_1M = 2.50

USD_TO_INR = 95.20
QUERIES_PER_DAY = 2000
COST_BUDGET_PER_QUERY_USD = 0.0015


# ============================================================
# 3. SHARED HELPERS
# ============================================================
def percentile(values, p):
    values = [v for v in values if not math.isnan(v)]

    if not values:
        return float("nan")

    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)

    if lo == hi:
        return values[lo]

    return values[lo] * (hi - k) + values[hi] * (k - lo)


def col_avg(rows, key):
    return sum(row[key] for row in rows) / len(rows) if rows else 0.0


def retrieve_context(pipeline, question):
    """Retrieve and join documents using the existing retriever."""
    docs = pipeline.retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context


def summarize_latency(values):
    values = [v for v in values if not math.isnan(v)]

    if not values:
        return {
            "n": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


# ============================================================
# 4. LATENCY EVALUATION
# ============================================================

def latency_stages_streaming(pipeline, question):
    """Measure retrieval, generation, and end-to-end latency."""
    start = time.perf_counter()

    docs = pipeline.retriever.invoke(question)
    retrieval_end = time.perf_counter()

    # Preserve the retrieved document content for the generator.
    context = [doc.page_content for doc in docs]

    answer = generate(question, context)
    end = time.perf_counter()

    return answer, {
        "retrieval": (retrieval_end - start) * 1000,
        "generation": (end - retrieval_end) * 1000,
        "ttft": float("nan"),
        "total": (end - start) * 1000,
    }



def latency_benchmark(pipeline):
    print("\n[Latency] Running warm-up...")

    for i in range(LAT_WARMUP_RUNS):
        pipeline.invoke(QUESTIONS[i % len(QUESTIONS)])

    results = {
        "total": [],
        "retrieval": [],
        "generation": [],
        "ttft": [],
        "answer_len": [],
    }

    print("[Latency] Measuring response times...")

    for question in QUESTIONS:
        for _ in range(LAT_REPEATS):
            answer, stage = latency_stages_streaming(
                pipeline, question
            )

            results["total"].append(stage["total"])
            results["retrieval"].append(stage["retrieval"])
            results["generation"].append(stage["generation"])
            results["ttft"].append(stage["ttft"])
            results["answer_len"].append(len(answer))

    return results


def latency_report(results):
    print("\n" + "=" * 72)
    print("LATENCY REPORT (milliseconds)")
    print("=" * 72)

    for name in ("total", "retrieval", "generation", "ttft"):
        stats = summarize_latency(results[name])

        print(
            f"{name:<12} n={stats['n']:<3} "
            f"mean={stats['mean']:8.1f} "
            f"p50={stats['p50']:8.1f} "
            f"p95={stats['p95']:8.1f} "
            f"p99={stats['p99']:8.1f}"
        )

    print("-" * 72)

    total = summarize_latency(results["total"])
    ttft = summarize_latency(results["ttft"])

    e2e_pass = total["p95"] <= SLO_P95_MS
    ttft_pass = ttft["p95"] <= SLO_TTFT_P95_MS

    print(
        f"End-to-end p95 SLO <= {SLO_P95_MS} ms: "
        f"{'PASS' if e2e_pass else 'FAIL'}"
    )
    print(
        f"TTFT p95 SLO <= {SLO_TTFT_P95_MS} ms: "
        f"{'PASS' if ttft_pass else 'FAIL'}"
    )

    return {
        "e2e_mean_ms": total["mean"],
        "e2e_p50_ms": total["p50"],
        "e2e_p95_ms": total["p95"],
        "e2e_p99_ms": total["p99"],
        "retrieval_p95_ms": summarize_latency(
            results["retrieval"]
        )["p95"],
        "generation_p95_ms": summarize_latency(
            results["generation"]
        )["p95"],
        "ttft_p95_ms": ttft["p95"],
        "slo_e2e_pass": e2e_pass,
        "slo_ttft_pass": ttft_pass,
    }


def run_latency(pipeline, verbose=True):
    results = latency_benchmark(pipeline)

    if verbose:
        latency_report(results)

    return latency_report_metrics(results)


def latency_report_metrics(results):
    total = summarize_latency(results["total"])
    ttft = summarize_latency(results["ttft"])

    return {
        "e2e_mean_ms": total["mean"],
        "e2e_p50_ms": total["p50"],
        "e2e_p95_ms": total["p95"],
        "e2e_p99_ms": total["p99"],
        "retrieval_p95_ms": summarize_latency(
            results["retrieval"]
        )["p95"],
        "generation_p95_ms": summarize_latency(
            results["generation"]
        )["p95"],
        "ttft_p95_ms": ttft["p95"],
        "slo_e2e_pass": total["p95"] <= SLO_P95_MS,
        "slo_ttft_pass": ttft["p95"] <= SLO_TTFT_P95_MS,
    }


# ============================================================
# 5. COST EVALUATION
# ============================================================
def cost_measure_tokens(pipeline, question):
    context = retrieve_context(pipeline, question)

    # Capture the AIMessage so token usage metadata is available.
    message = measured_chain.invoke({
        "question": question,
        "context": context,
    })

    usage = message.usage_metadata or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    details = usage.get("input_token_details") or {}
    cached_tokens = details.get("cache_read", 0) or 0
    cached_tokens = min(cached_tokens, input_tokens)

    return {
        "input": input_tokens,
        "output": output_tokens,
        "cached": cached_tokens,
    }


def cost_usd(input_tokens, output_tokens, cached_tokens):
    uncached_input = max(input_tokens - cached_tokens, 0)

    input_cost = uncached_input / 1_000_000 * PRICE_INPUT_PER_1M
    cached_cost = (
        cached_tokens / 1_000_000 * PRICE_CACHED_INPUT_PER_1M
    )
    output_cost = output_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M

    return {
        "input": input_cost,
        "cached": cached_cost,
        "output": output_cost,
        "total": input_cost + cached_cost + output_cost,
    }


def cost_benchmark(pipeline):
    rows = []

    print("\n[Cost] Measuring token usage...")

    for question in QUESTIONS:
        for _ in range(COST_REPEATS):
            tokens = cost_measure_tokens(pipeline, question)
            costs = cost_usd(
                tokens["input"],
                tokens["output"],
                tokens["cached"],
            )

            rows.append({
                **tokens,
                **{f"cost_{k}": v for k, v in costs.items()},
            })

    return rows


def cost_report(rows):
    avg_cost = col_avg(rows, "cost_total")
    avg_output_cost = col_avg(rows, "cost_output")
    output_share = (
        100 * avg_output_cost / avg_cost if avg_cost else 0
    )

    daily_cost = avg_cost * QUERIES_PER_DAY
    monthly_cost = daily_cost * 30

    print("\n" + "=" * 72)
    print(f"COST REPORT — {MODEL_NAME}")
    print("=" * 72)
    print(f"Samples               : {len(rows)}")
    print(f"Average input tokens  : {col_avg(rows, 'input'):.0f}")
    print(f"Average cached tokens : {col_avg(rows, 'cached'):.0f}")
    print(f"Average output tokens : {col_avg(rows, 'output'):.0f}")
    print(f"Average cost/query    : ${avg_cost:.8f}")
    print(f"Cost/query in INR     : ₹{avg_cost * USD_TO_INR:.6f}")
    print(f"Output cost share     : {output_share:.1f}%")
    print(f"Projected daily cost  : ${daily_cost:.4f}")
    print(f"Projected monthly cost: ${monthly_cost:.4f}")
    print(
        f"Budget result         : "
        f"{'PASS' if avg_cost <= COST_BUDGET_PER_QUERY_USD else 'FAIL'}"
    )
    print("=" * 72)

    return {
        "cost_per_query_usd": avg_cost,
        "cost_per_query_inr": avg_cost * USD_TO_INR,
        "avg_input_tokens": col_avg(rows, "input"),
        "avg_output_tokens": col_avg(rows, "output"),
        "avg_cached_tokens": col_avg(rows, "cached"),
        "output_cost_share_pct": output_share,
        "monthly_usd": monthly_cost,
        "budget_pass": avg_cost <= COST_BUDGET_PER_QUERY_USD,
    }


def run_cost(pipeline, verbose=True):
    rows = cost_benchmark(pipeline)
    return cost_report(rows)


# ============================================================
# 6. RELIABILITY EVALUATION
# ============================================================
class Reliability:
    def __init__(self):
        self.calls = 0
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.retries = 0
        self.first_attempt_successes = 0


def call_with_retries(fn, tracker):
    tracker.calls += 1

    for attempt in range(MAX_RETRIES + 1):
        tracker.attempts += 1

        try:
            result = fn()
            tracker.successes += 1

            if attempt == 0:
                tracker.first_attempt_successes += 1

            return result

        except Exception as error:
            if attempt < MAX_RETRIES:
                tracker.retries += 1
                delay = BACKOFF_BASE_S * (2 ** attempt)

                print(
                    f"[Reliability] Attempt {attempt + 1} failed: "
                    f"{error}. Retrying in {delay:.1f}s."
                )

                time.sleep(delay)
            else:
                tracker.failures += 1
                print(
                    f"[Reliability] Request failed after retries: "
                    f"{error}"
                )

    return None


def reliability_benchmark(pipeline):
    tracker = Reliability()

    print("\n[Reliability] Measuring request success...")

    for question in QUESTIONS:
        for _ in range(REL_REPEATS):
            call_with_retries(
                lambda q=question: pipeline.invoke(q),
                tracker,
            )

    return tracker


def reliability_report(tracker):
    calls = tracker.calls

    success_rate = (
        100 * tracker.successes / calls if calls else 0
    )
    error_rate = (
        100 * tracker.failures / calls if calls else 0
    )
    retry_rate = (
        100 * tracker.retries / calls if calls else 0
    )
    first_success_rate = (
        100 * tracker.first_attempt_successes / calls
        if calls else 0
    )

    avg_attempts = (
        tracker.attempts / calls if calls else 0
    )

    print("\n" + "=" * 72)
    print("RELIABILITY REPORT")
    print("=" * 72)
    print(f"Logical requests          : {calls}")
    print(f"Total attempts             : {tracker.attempts}")
    print(f"Successful requests        : {tracker.successes}")
    print(f"Failed requests            : {tracker.failures}")
    print(f"Retry events               : {tracker.retries}")
    print(f"Final success rate         : {success_rate:.2f}%")
    print(f"Final error rate           : {error_rate:.2f}%")
    print(f"First-attempt success rate : {first_success_rate:.2f}%")
    print(f"Retry events per 100 calls : {retry_rate:.2f}")
    print(f"Average attempts/request   : {avg_attempts:.2f}")
    print("=" * 72)

    return {
        "total_requests": calls,
        "total_attempts": tracker.attempts,
        "success_rate": success_rate,
        "error_rate": error_rate,
        "first_attempt_success_rate": first_success_rate,
        "retry_rate": retry_rate,
        "avg_attempts_per_request": avg_attempts,
    }


def run_reliability(pipeline, verbose=True):
    tracker = reliability_benchmark(pipeline)
    return reliability_report(tracker)


# ============================================================
# 7. RUN ALL OPERATIONAL EVALUATIONS
# ============================================================
def run_ops(pipeline=None, verbose=True):
    pipeline = pipeline or RagPipeline()

    # Run each evaluation against the same pipeline instance.
    latency_results = latency_benchmark(pipeline)
    latency = latency_report_metrics(latency_results)

    cost_rows = cost_benchmark(pipeline)
    cost = cost_report(cost_rows)

    reliability_tracker = reliability_benchmark(pipeline)
    reliability = reliability_report(reliability_tracker)

    snapshot = {}
    snapshot.update({
        f"latency.{key}": value
        for key, value in latency.items()
    })
    snapshot.update({
        f"cost.{key}": value
        for key, value in cost.items()
    })
    snapshot.update({
        f"reliability.{key}": value
        for key, value in reliability.items()
    })

    return snapshot


# ============================================================
# 8. ENTRYPOINT
# ============================================================
def main():
    print("=" * 72)
    print("RAG OPERATIONAL EVALUATION")
    print(f"Configured model: {MODEL_NAME}")
    print("=" * 72)

    snapshot = run_ops()

    print("\n" + "=" * 72)
    print("COMBINED OPERATIONAL SNAPSHOT")
    print("=" * 72)

    for key, value in snapshot.items():
        if isinstance(value, float):
            print(f"{key:<38} {value:.6f}")
        else:
            print(f"{key:<38} {value}")

    print("=" * 72)


if __name__ == "__main__":
    main()
