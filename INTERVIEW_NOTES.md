# Interview notes (15-minute read)

## The system in five sentences
QualityCopilot answers questions over a seeded collection of fictional quality reports. Qdrant retrieves candidate reports and LangGraph coordinates analysis, validation, and bounded retry. The model returns a Pydantic-shaped summary and hypotheses with source document IDs. Python rejects citations not present in the retrieved set, while a support checker flags unsupported claims. A person approves, edits, or rejects each hypothesis, and the decision is saved locally in SQLite.

## Design choices
- **LangGraph:** visible stateful retrieve/analyze/validate/retry flow with a hard retry limit.
- **Qdrant:** embedded local vector store; no external service or Docker required.
- **Streamlit:** quick single-process demo UI with expandable evidence and review actions.
- **Chunking:** one concise report per point (~800 characters) preserves each report's context and `doc_id`; no arbitrary mid-report splits.
- **Validator:** citation allow-list is deterministic; live support check asks the model whether cited text supports the claim. It cannot prove truth, catch every subtle inference, or guarantee an LLM is unbiased. Key-free demo mode uses simple lexical overlap instead.
- **Human review:** reviewer can approve/edit/reject and leave a comment; SQLite stores original claim, optional edited text, decision, comment, and timestamp.

## Evaluation interpretation
Run `python eval/run_eval.py` after building the index. It reports retrieval Hit@5 and MRR from ground truth IDs, without LLM-written references. Current checked-in example output is illustrative rather than measured; there is no completed before/after benchmark yet. The verify suite tests abstention, citation membership, and invalid-claim rejection offline.

## Three weaknesses and improvements
1. Token-hash retrieval is lexical: use and benchmark a proper embedding model plus a committed/rebuildable index.
2. Support checking can miss nuanced hallucinations: test independent evaluators, structured evidence spans, and human adjudication.
3. Local SQLite/Qdrant state is not durable on ephemeral hosting: use a persistent disk or managed services if the demo becomes a real product.

## Likely interview questions
1. **Why LangGraph?** It makes the state transitions and bounded validation retry explicit.
2. **Why Qdrant?** It offers vector search in embedded local mode for a small demo.
3. **How are citations enforced?** Every cited ID must belong to this run's retrieved document set.
4. **Can the model answer without evidence?** Retrieval abstains below threshold and does not call analysis.
5. **How is prompt injection handled?** Retrieved text is explicitly treated as untrusted, and output is constrained to evidence; this is mitigation, not a guarantee.
6. **What does human review do?** Saves approve/edit/reject with claim and optional comment for later inspection.
7. **What does evaluation measure?** Whether known pattern documents rank in top five and reciprocal rank, not answer correctness in production.
8. **Why synthetic data?** It avoids exposing company/customer data and gives known ground truth for repeatable tests.
9. **What are the limitations?** Synthetic scale, lexical retrieval, imperfect validation, and non-durable local hosting.
10. **How was it built?** AI-assisted implementation with automated checks; I would explain the state flow, citation guard, and measured limitations honestly.
