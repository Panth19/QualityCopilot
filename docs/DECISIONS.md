# Build decisions

- Use Python 3.11+ and pin the app/test dependencies in requirements.txt; validate the pins in the deployment environment before launch.
- Generate 180 seeded synthetic reports across six planted patterns and miscellaneous cases; omit external NHTSA downloads to keep setup deterministic and offline.
- Use Qdrant local embedded storage with deterministic signed token-hash vectors so indexing and retrieval checks work without an embedding API or model download.
- Use OpenAI's LangChain chat wrapper for live structured analysis when OPENAI_API_KEY is configured; provide a clearly identified deterministic demo model for local smoke tests and key-free showcases.
- Keep chunking at one self-contained report per point (about 800 characters) to preserve FMEA fields and citation identity.
- Enforce evidence threshold and citation allow-list in Python before accepting claims; use a structured LLM support check and cap analysis retries at two.
- Persist human decisions in a local SQLite database; the deployed single-process demo uses its local filesystem and is not durable across ephemeral host resets.
- Use Streamlit Community Cloud as the deployment target; no FastAPI wrapper, Docker, authentication, or multi-user features are included.
- Run all automated checks offline with deterministic embeddings and the demo model; live provider behavior requires an API key and is not claimed as locally verified.
