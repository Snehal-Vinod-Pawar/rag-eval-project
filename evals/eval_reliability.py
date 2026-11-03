"""
Operational eval: RELIABILITY

Reliability measures whether the RAG application can successfully
serve requests without failing.

We measure:
    - success rate
    - error rate
    - retry rate
    - total attempts
    - average attempts per request

Retries are important because a system may eventually succeed while
still being flaky on the first attempt.
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
import time
from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline

load_dotenv()


# ============================================================
# 2. CONFIG
# ============================================================
QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 1  # Increase to 5 after the initial test

MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5


# ============================================================
# 3. RELIABILITY TRACKER
# ============================================================
class Reliability:

    def __init__(self):
        self.calls = 0
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.retries = 0
        self.first_attempt_successes = 0


# ============================================================
# 4. RETRY WRAPPER
# ============================================================
def call_with_retries(fn, reliability):

    reliability.calls += 1

    for attempt in range(MAX_RETRIES + 1):

        reliability.attempts += 1

        try:
            result = fn()

            reliability.successes += 1

            if attempt == 0:
                reliability.first_attempt_successes += 1

            return result

        except Exception as e:

            if attempt < MAX_RETRIES:

                reliability.retries += 1

                # Exponential backoff
                delay = BACKOFF_BASE_S * (2 ** attempt)

                print(
                    f"Request failed on attempt {attempt + 1}. "
                    f"Retrying in {delay:.1f}s... Error: {e}"
                )

                time.sleep(delay)

            else:

                reliability.failures += 1

                print(
                    f"FAILED after {MAX_RETRIES} retries: {e}"
                )

                return None


# ============================================================
# 5. BENCHMARK
# ============================================================
def benchmark(pipeline):

    reliability = Reliability()

    total_requests = len(QUESTIONS) * REPEATS

    print("Measuring RAG pipeline reliability...")
    print(f"Total logical requests: {total_requests}")
    print(f"Maximum attempts per request: {MAX_RETRIES + 1}")

    for question in QUESTIONS:

        for repeat in range(REPEATS):

            print(
                f"\nRequest {reliability.calls + 1}/{total_requests}: "
                f"{question}"
            )

            call_with_retries(
                lambda q=question: pipeline.invoke(q),
                reliability
            )

    return reliability


# ============================================================
# 6. REPORT
# ============================================================
def report(rel):

    success_rate = (
        100 * rel.successes / rel.calls
        if rel.calls else 0
    )

    error_rate = (
        100 * rel.failures / rel.calls
        if rel.calls else 0
    )

    retry_rate = (
        100 * rel.retries / rel.calls
        if rel.calls else 0
    )

    first_attempt_success_rate = (
        100 * rel.first_attempt_successes / rel.calls
        if rel.calls else 0
    )

    avg_attempts = (
        rel.attempts / rel.calls
        if rel.calls else 0
    )

    print("\n" + "=" * 65)
    print("RAG PIPELINE RELIABILITY REPORT")
    print("=" * 65)

    print(f"Total logical requests       : {rel.calls}")
    print(f"Total execution attempts     : {rel.attempts}")
    print(f"Successful requests          : {rel.successes}")
    print(f"Failed requests              : {rel.failures}")
    print(f"Retry events                 : {rel.retries}")

    print("-" * 65)

    print(f"Final success rate           : {success_rate:.2f}%")
    print(f"Final error rate             : {error_rate:.2f}%")
    print(f"First-attempt success rate   : {first_attempt_success_rate:.2f}%")
    print(f"Retry events per 100 requests: {retry_rate:.2f}")
    print(f"Average attempts per request : {avg_attempts:.2f}")

    print("=" * 65)

    print(
        "Note: Reliability includes retrieval, reranking, generation, "
        "network, and API failures."
    )
    print(
        "Retry rate counts retry events per logical request, so it may "
        "exceed 100 if multiple retries occur per request."
    )


# ============================================================
# 7. ENTRYPOINT
# ============================================================
def main():

    pipeline = RagPipeline()

    reliability = benchmark(pipeline)

    report(reliability)


if __name__ == "__main__":
    main()
