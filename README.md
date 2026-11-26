# QualityCopilot

QualityCopilot is a compact showcase of evidence-grounded quality-report analysis: a LangGraph workflow retrieves synthetic FMEA-style reports from embedded Qdrant, proposes citation-bearing hypotheses through a LangChain/OpenAI structured-output model, validates citations and claim support, and ends with a human review saved to SQLite. With no API key it uses a deterministic demo model; this makes the app and checks usable key-free, but the demo's lexical support check is not an LLM evaluation.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # optionally set OPENAI_API_KEY; export values before launching
python scripts/generate_data.py
python -c 'from scripts.generate_data import generate; from qualitycopilot.retrieval import ingest; ingest(generate()[0])'
streamlit run app.py
```

The app also creates synthetic data and its local index when missing. Configure `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`) in the environment or Streamlit secrets for live analysis. Do not commit credentials. Data are fictional; no public NHTSA data is downloaded.

## Automated checks

```bash
python scripts/verify.py
python -m pytest -q
```

Checks use offline hash vectors and deterministic demo analysis; no live provider call is required. The exact environment and versions must be installed before running. The CI-like verification script prints checks as they complete; `docs/VERIFY_OUTPUT.txt` is intentionally not claimed as passing until run in an environment with these dependencies installed.

## Evaluation

| Configuration | Hit@5 | MRR | Abstention accuracy | Citation validity | Tokens |
|---|---:|---:|---:|---:|---:|
| Deterministic token hashing, 800-character report chunks (baseline) | Run `python eval/run_eval.py` | Run command | 1.00* | 1.00* | 0 offline |
| Before/after comparison | Not measured | Not measured | — | — | — |

*These are deterministic test expectations, not measured quality claims. Evaluation currently emits 18 pattern phrasings; off-topic abstention is covered by verification tests, not by the standalone metric output. No before/after retrieval experiment has been run, so no comparative improvement is claimed.

## Architecture

```mermaid
flowchart LR
  U[Streamlit question] --> G[LangGraph]
  G --> R[Qdrant local retrieval]
  R -->|evidence or abstain| A[Structured analysis]
  A --> V[Python citation allow-list + support validator]
  V -->|retry up to 2| A
  V --> H[Human approve / edit / reject]
  H --> S[(SQLite review history)]
```

## Limitations and next steps

- Synthetic, small-scale data do not establish real manufacturing quality or safety performance.
- Hash-based retrieval is lexical and can miss semantic matches; the demo support checker is shallow, and live LLM validation can also be wrong.
- SQLite and the embedded index are local single-process state; ephemeral hosting may reset review history and rebuilt indexes.
- Next: compare alternative embedding/retrieval configurations, add measured evaluation runs, improve adversarial claim checks, and deploy with a provider spending cap.

Deploy on Streamlit Community Cloud by selecting `app.py`, adding `OPENAI_API_KEY` under app secrets, and using `requirements.txt`. Keep the key out of the repository.
