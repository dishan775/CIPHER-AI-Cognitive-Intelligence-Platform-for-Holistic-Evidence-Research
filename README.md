# CIPHER — AI Cognitive Intelligence Platform for Holistic Evidence Research

CIPHER is an end-to-end, local-first research intelligence platform combining multi-agent collaboration, natural language processing, automated knowledge graph extraction, and interactive visual analytics.

---

## Overview

CIPHER provides an integrated environment for researchers and engineers to ingest scientific literature, extract structured claims and entities, visualize dynamic knowledge networks, and run collaborative multi-agent analytical workflows.

### Key Capabilities

- **Multi-Agent Research Pipeline:** Orchestrates specialized autonomous agents (Research Analyst, Methodology Reviewer, Experiment Designer, Statistician) in a sequential reasoning workflow.
- **Automated Knowledge Graph Engine:** Automatically parses domain entities (methods, models, datasets, metrics, concepts) and extracts grounded relation triples with evidence citations.
- **Interactive Network Visualization:** Explores extracted entities and conceptual relationships via high-performance Cytoscape graph canvas with physics-based layout.
- **Document Ingestion & Chunking:** Ingests research articles, PDF documents, DOCX manuscripts, and live web sources with automated structural parsing.
- **NLP Analytics & Extraction:** Tokenization, part-of-speech tagging, named entity recognition, and semantic analysis powered by spaCy and NLTK with zero required cloud dependencies.
- **Local-First Architecture:** Complete offline capability using local NLP engines and local LLM inference via Ollama, with optional cloud acceleration.

---

## Architecture

```
CIPHER Platform
├── frontend/                     # Modern Web UI
│   ├── index.html                # Landing & Overview Gateway
│   ├── dashboard.html            # Research & Analytics Dashboard
│   ├── src/
│   │   ├── api/                  # API client communication layer
│   │   ├── styles/               # Design system & modular stylesheets
│   │   ├── crew.ts               # Multi-agent workflow visualizer
│   │   ├── dashboard.ts          # Core dashboard logic and state management
│   │   ├── kg.ts                 # Knowledge graph canvas & relation review
│   │   └── visualizer.ts         # Cytoscape interactive graph renderer
│   └── vite.config.ts            # Development server & proxy configuration
│
├── backend/                      # FastAPI Python Service
│   ├── main.py                   # Server entry point & API route orchestration
│   ├── database.py               # SQLAlchemy SQLite schemas & connection management
│   ├── nlp_service.py            # spaCy / NLTK processing pipeline
│   ├── knowledge_graph_service.py# Entity & relationship extraction engine
│   ├── crew_service.py           # Multi-agent pipeline runner
│   ├── document_pipeline.py      # Document ingestion & chunking pipeline
│   ├── web_fetch_service.py      # Web article parsing & extraction
│   ├── gemini_service.py         # Optional structured synthesis service
│   ├── routes_*.py               # Modular REST API routers
│   └── requirements.txt          # Python dependency specifications
│
└── docs/                         # Technical documentation & architecture specs
```

---

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, SQLAlchemy, SQLite, Pydantic, Uvicorn
- **NLP & Inference:** spaCy (`en_core_web_sm`), NLTK, scikit-learn, SentenceTransformers, Ollama (`qwen3:1.7b`)
- **Frontend:** TypeScript, Vite, Vanilla CSS, Cytoscape.js, Three.js
- **Document Processing:** `pdfplumber`, `python-docx`, `beautifulsoup4`

---

## Getting Started

### Prerequisites

- **Python:** 3.10 or higher
- **Node.js:** 18.0 or higher
- **npm:** 8.0 or higher
- **Ollama** *(Optional, for local LLM agent execution)*: [ollama.ai](https://ollama.ai)

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the required spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. *(Optional)* Configure environment variables:
   ```bash
   cp .env.example .env
   ```

6. Start the FastAPI server:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

The backend API will be available at `http://127.0.0.1:8000` with interactive API docs at `http://127.0.0.1:8000/docs`.

### Frontend Setup

1. In a separate terminal, navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open the application in your browser:
   - Landing Gateway: `http://localhost:5173/`
   - Research Dashboard: `http://localhost:5173/dashboard.html`

---

## Multi-Agent Pipeline Roles

| Agent | Icon | Function |
| :--- | :---: | :--- |
| **Research Analyst** | 🔬 | Conducts literature surveys, extracts key claims, and identifies research gaps. |
| **Methodology Reviewer**| ⚖️ | Audits scientific validity, identifies procedural bias, and verifies baseline benchmarks. |
| **Experiment Designer** | 🧪 | Formulates experimental protocols, variable mappings, and ablation studies. |
| **Statistician** | 📊 | Designs the statistical framework, error bounds, and power analysis. |

---

## API Documentation

When the backend is running, complete OpenAPI specifications and interactive testing consoles are available at:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## Security & Privacy

- All document ingestion and natural language processing occur locally by default.
- SQLite databases and uploaded user documents are excluded from version control.
- API keys, credentials, and environment files are strictly isolated from source control.

---

## License

This project is licensed under the Apache 2.0 License.
