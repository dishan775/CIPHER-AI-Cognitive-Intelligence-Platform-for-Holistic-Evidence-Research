# CIPHER Architecture

## Overview
CIPHER is a local-first, NLP-powered, agentic AI research and knowledge automation platform. It operates over a shared NLP and knowledge infrastructure tailored for research intelligence and knowledge-to-action workflows.

## Stack
- **Frontend:** Vite, HTML, CSS, TypeScript. Plain DOM manipulation with responsive CSS layout, preserving the existing premium crimson/gold/dark visual identity.
- **Backend:** FastAPI (Python).
- **Database:** SQLite (via SQLAlchemy).
- **NLP & Inference:** spaCy, NLTK, scikit-learn, Ollama (Local LLM - `qwen3:1.7b`), SentenceTransformers (for embeddings).

## Data Models
1. **Project:** Groups documents, configuration, and analysis for a specific context.
2. **Document:** Represents an uploaded file or scraped webpage.
3. **DocumentChunk:** Represents chunks of text from a document.
4. **SourceMetadata:** Additional information about the document (authors, publication date, etc).
5. **ExtractedEntity / ExtractedClaim:** Extracted knowledge points linked to chunks.
6. **ComparisonResult:** Outputs from comparing documents or claims.

## Services
1. **Ingestion Service:** Handles file uploads and web scraping. Cleans text.
2. **NLP Engine:** Preprocessing, POS, NER, Text classification, semantic analysis.
3. **Research Intelligence:** Extracts metrics, compares papers, identifies research gaps.
4. **Knowledge Copilot:** RAG-based QA, procedure extraction, workflow drafting.
5. **Agent Workflow:** Routes tasks, maintains state, delegates to specific sub-agents.
