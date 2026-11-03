
"""
Operational evaluation: COST for Gemini 3.5 Flash-Lite.

Measures actual token usage from the Gemini generation response and estimates
cost using published standard paid-tier prices.

Note:
- This estimates generation API cost, not the entire infrastructure cost.
- Cached input tokens are discounted only when usage metadata reports them.
- Context caching storage charges, if applicable, are not included.
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.generator import prompt, llm  # Reuse the exact prompt and Gemini model

load_dotenv()

# Stop before StrOutputParser() to retain the AIMessage and usage metadata.
measured_chain = prompt | llm


# ============================================================
# 2. CONFIG
# ============================================================
MODEL_NAME = "gemini-3.5-flash-lite"

QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 3

# Gemini 3.5 Flash-Lite standard paid-tier pricing, USD per 1M tokens.
# Source: Google Gemini API pricing, checked October 28, 2026.
PRICE_INPUT_PER_1M = 0.30
PRICE_CACHED_INPUT_PER_1M = 0.03
PRICE_OUTPUT_PER_1M = 2.50

# Approximate USD-to-INR reference rate, October 28, 2026.
USD_TO_INR = 95.20

# Business projection
QUERIES_PER_DAY = 2000

# Budget threshold: kept from your original benchmark.
COST_BUDGET_PER_QUERY_USD = 0.0015


# ============================================================
# 3. TOKEN MEASUREMENT
# ============================================================
def measure_tokens(pipeline, question):
    # Retrieve real documents from the existing retriever.
    docs = pipeline.retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in docs)

    # Use the same prompt and LLM as the production generator.
    msg = measured_chain.invoke({
        "question": question,
        "context": context_text,
    })

    usage = msg.usage_metadata or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # Cached tokens, if reported by the provider integration.
    details = usage.get("input_token_details") or {}
    cached_tokens = details.get("cache_read", 0) or 0

    # Prevent cached tokens from exceeding total input tokens.
    cached_tokens = min(cached_tokens, input_tokens)

    return {
        "input": input_tokens,
        "output": output_tokens,
        "cached": cached_tokens,
    }


# ============================================================
# 4. COST CALCULATION
# ============================================================
def cost_usd(input_tokens, output_tokens, cached_tokens):
    uncached_input = max(input_tokens - cached_tokens, 0)

    c_in = uncached_input / 1_000_000 * PRICE_INPUT_PER_1M
    c_cached = cached_tokens / 1_000_000 * PRICE_CACHED_INPUT_PER_1M
    c_out = output_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M

    return {
        "input": c_in,
        "cached": c_cached,
        "output": c_out,
        "total": c_in + c_cached + c_out,
    }


# ============================================================
# 5. BENCHMARK LOOP
# ============================================================
def benchmark(pipeline):
    rows = []
    total_runs = len(QUESTIONS) * REPEATS

    print(f"Model: {MODEL_NAME}")
    print(f"Measuring token usage across {total_runs} runs...")

    for question in QUESTIONS:
        for repeat in range(REPEATS):
            print(
                f"Question: {question[:55]}... "
                f"(run {repeat + 1}/{REPEATS})"
            )

            tok = measure_tokens(pipeline, question)
            cost = cost_usd(tok["input"], tok["output"], tok["cached"])

            rows.append({
                "question": question,
                **tok,
                **{f"cost_{k}": v for k, v in cost.items()},
            })

    return rows


# ============================================================
# 6. AGGREGATE + REPORT
# ============================================================
def avg(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def report(rows):
    if not rows:
        print("No measurements available.")
        return

    n = len(rows)

    avg_in = avg(rows, "input")
    avg_out = avg(rows, "output")
    avg_cached = avg(rows, "cached")
    avg_cost = avg(rows, "cost_total")

    min_cost = min(row["cost_total"] for row in rows)
    max_cost = max(row["cost_total"] for row in rows)

    avg_cost_in = avg(rows, "cost_input") + avg(rows, "cost_cached")
    avg_cost_out = avg(rows, "cost_output")

    out_share = 100 * avg_cost_out / avg_cost if avg_cost else 0
    input_share = 100 - out_share

    print("\n" + "=" * 72)
    print(f"COST BENCHMARK — {MODEL_NAME}")
    print("=" * 72)

    print(
        f"Pricing per 1M tokens  : "
        f"${PRICE_INPUT_PER_1M:.2f} input / "
        f"${PRICE_CACHED_INPUT_PER_1M:.2f} cached / "
        f"${PRICE_OUTPUT_PER_1M:.2f} output"
    )
    print(f"USD-to-INR rate        : ₹{USD_TO_INR:.2f}")
    print(f"Samples                : {n}")
    print(f"Average input tokens   : {avg_in:,.0f}")
    print(f"Average cached tokens  : {avg_cached:,.0f}")
    print(f"Average output tokens  : {avg_out:,.0f}")

    print("-" * 72)
    print(
        f"Average cost / query   : ${avg_cost:.8f} "
        f"(₹{avg_cost * USD_TO_INR:.6f})"
    )
    print(
        f"Minimum / maximum      : "
        f"${min_cost:.8f} / ${max_cost:.8f}"
    )
    print(
        f"Input vs output cost   : "
        f"{input_share:.1f}% input / {out_share:.1f}% output"
    )

    print("-" * 72)

    daily = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30

    print(f"Projection at {QUERIES_PER_DAY:,} queries/day:")
    print(f"  Daily cost            : ${daily:.4f} (₹{daily * USD_TO_INR:.2f})")
    print(f"  Monthly cost (30 days): ${monthly:.4f} (₹{monthly * USD_TO_INR:.2f})")

    print("-" * 72)

    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_QUERY_USD else "FAIL"

    print(
        f"BUDGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f}"
    )
    print(f"Average cost            : ${avg_cost:.8f}")
    print(f"Result                  : {verdict}")

    print("=" * 72)
    print(
        "Note: Cached-token discounts apply only when cached usage is "
        "reported. Actual charges may differ by billing tier, caching, "
        "and other applicable fees."
    )


# ============================================================
# 7. ENTRYPOINT
# ============================================================
def main():
    pipeline = RagPipeline()
    rows = benchmark(pipeline)
    report(rows)


if __name__ == "__main__":
    main()
