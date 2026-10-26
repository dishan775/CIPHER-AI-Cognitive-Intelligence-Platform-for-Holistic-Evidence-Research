# CIPHER Implementation Plan

## Stage 1: Repository Audit and Architecture
- [x] Inspect existing codebase.
- [x] Document architecture and plan.
- [x] Identify dependencies: Need to add `spacy`, `sentence-transformers`, `scikit-learn`, `python-multipart` (for uploads) to backend requirements.

## Stage 2: Shared Foundation
- [ ] Update `backend/requirements.txt` and install dependencies.
- [ ] Update `database.py` with SQLAlchemy models: Project, Document, DocumentChunk, ExtractedFact.
- [ ] Implement CRUD for Projects.
- [ ] Implement Document Ingestion (Files & URLs). Store chunks in DB.

## Stage 3: NLP Foundation
- [ ] Create `nlp_service.py` with spaCy and NLTK logic.
- [ ] Implement Preprocessing (tokenization, stop words, lemmatization).
- [ ] Implement NER and POS tagging.
- [ ] Implement classical feature extraction (TF-IDF).
- [ ] Add NLP Lab API endpoints.

## Stage 4: Research Intelligence
- [ ] Implement structured paper extraction (Authors, Objectives, Findings).
- [ ] Cross-paper comparison (Matrix of methods/results).
- [ ] Evidence graph (supporting/contradicting claims).

## Stage 5: Knowledge-to-Action
- [ ] Implement RAG (Retrieval-Augmented Generation) using SentenceTransformers embeddings.
- [ ] Source-grounded QA endpoint.
- [ ] Procedure & Action-item extraction from SOPs/manuals.

## Stage 6: Agent Workflows & Report Studio
- [ ] Task Router to map intents to sub-workflows.
- [ ] Implement draft report generation endpoint.
- [ ] Link frontend dashboard to workflow APIs.

## Stage 7 & 8: UI Updates & Verification
- [ ] Expand `dashboard.html` to include Overview, Projects, Knowledge Library, Intelligence, Copilot, NLP Lab.
- [ ] Refactor `dashboard.ts` to manage state for these sections.
- [ ] Final end-to-end acceptance tests.
