# LLM Evaluation Framework for RAG

An evaluation framework for assessing the retrieval quality, response quality, safety, and operational performance of a Retrieval-Augmented Generation (RAG) application using **DeepEval** and **Google Gemini**.

## Key Features

- **Retriever Evaluation:** Contextual precision and recall.
- **Generator Evaluation:** Faithfulness and answer relevancy.
- **End-to-End RAG Evaluation:** Evaluates the complete retrieval and generation pipeline.
- **Safety Evaluation:** Correctness, toxicity, prompt leakage, course-content leakage, PII leakage, and scope adherence.
- **Cost Benchmarking:** Estimates token usage and API costs.
- **Reliability Benchmarking:** Measures success rate, error rate, and retry behavior.

## Tech Stack

- Python
- DeepEval
- LangChain
- Google Gemini API
- ChromaDB
- Hugging Face Sentence Transformers

## Setup

### 1. Install Dependencies

Ensure Python 3.11 and `uv` are installed, then run the following commands from the project root:

```powershell
uv sync
```

### 2. Configure the API Key

Copy the example environment file:

```powershell
copy .env.example .env
```

Open `.env` and replace the placeholder with your Google AI Studio API key:

```env
GOOGLE_API_KEY=your-google-ai-studio-api-key-here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

Never commit your `.env` file or expose your API key publicly.

## Configuration

| Setting | Value |
|---|---|
| LLM judge | Google Gemini Flash |
| Judge model | `gemini-3.5-flash-lite` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (`chroma_store/`) |
| Retriever golden dataset | `goldens/retriever_goldens.json` |
| Retriever metrics | Contextual Recall and Contextual Precision |
| Retriever threshold | `0.7` |

The embedding model runs locally through Hugging Face. Gemini is used for LLM-based generation and evaluation where configured.

## Running Evaluations

Run commands from the project root.

### Retrieval and Generation

```powershell
uv run python -m evals.eval_retriever
uv run python -m evals.eval_generator
uv run python -m evals.eval_rag_pipeline
```

### Application and Safety

```powershell
uv run python -m evals.eval_application
uv run python -m evals.eval_toxicity
uv run python -m evals.eval_leakage
uv run python -m evals.eval_scope
uv run python -m evals.eval_scope_safety
```

### Operational Evaluations

```powershell
uv run python -m evals.eval_cost
uv run python -m evals.eval_reliability
```

Make sure each command matches the name of the corresponding Python file in the `evals/` directory.

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI Studio API key required for Gemini API access |

## Notes

- Local Hugging Face embeddings are used; no external embedding API is required.
- ChromaDB stores the indexed document chunks locally. Rebuild the index when the source documents or embedding configuration require it.
- LLM-based evaluations consume API requests and may encounter rate limits. Start with a small subset of golden test cases and increase gradually.
- The `.deepeval/` directory contains runtime cache and is not part of source control.
- Keep `.env`, API keys, and other sensitive credentials out of version control.

## Project Structure

```text
rag-eval-project/
├── evals/          # Evaluation scripts
├── goldens/        # Golden datasets
├── src/            # Retriever, generator, and RAG pipeline
├── chroma_store/   # Local vector database
├── .env.example    # Environment variable template
├── .gitignore
├── pyproject.toml
└── uv.lock
```