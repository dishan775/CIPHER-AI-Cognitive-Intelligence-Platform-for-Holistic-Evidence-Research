// ═══════════════════════════════════════════════════════════════
// CIPHER — Minimalist Live Process & NLP Pipeline Visualizer
// ═══════════════════════════════════════════════════════════════

export interface ProcessStep {
    name: string;
    description: string;
    techBadge: string;
}

export interface NLPContextInfo {
    icon: string;
    category: string;
    name: string;
    description: string;
    metricLabel: string;
    metricValue: string;
    tags: string;
}

export interface PipelinePreset {
    featureName: string;
    steps: ProcessStep[];
    nlpApp: NLPContextInfo;
    tensorLabel?: string;
}

export const PIPELINE_PRESETS: Record<string, PipelinePreset> = {
    nlpAnalyze: {
        featureName: "Text Intelligence & Parsing",
        tensorLabel: "POS + NER TOKENS",
        nlpApp: {
            icon: "🔬",
            category: "LINGUISTIC ANALYSIS",
            name: "spaCy en_core_web_sm",
            description: "Applies dependency parsing, POS tagging, and entity boundary identification over token vectors without external network calls.",
            metricLabel: "Token throughput",
            metricValue: "18,400 tok/sec",
            tags: "spaCy · Lemmatization · NER · Dependency Tree"
        },
        steps: [
            { name: "Raw String Sanitization", description: "Unicode normalization, punctuation stripping and sentence segmenting.", techBadge: "NLTK punkt" },
            { name: "Grammatical Part-of-Speech", description: "Labeling syntactic roles (verbs, nouns, adjectives) to understand sentence semantics.", techBadge: "spaCy POS Tagger" },
            { name: "Named Entity Extraction", description: "Classifying entities into ORG, PERSON, DATE, GPE and QUANTITY.", techBadge: "Bi-LSTM / CNN NER" },
            { name: "Extractive Keyword Scoring", description: "TF-IDF weighting to isolate core themes and structural summaries.", techBadge: "scikit-learn TF-IDF" }
        ]
    },
    docUpload: {
        featureName: "Document Ingestion & Indexing",
        tensorLabel: "384-D PASSAGE VECTORS",
        nlpApp: {
            icon: "📑",
            category: "DENSE RETRIEVAL",
            name: "SentenceTransformers (all-MiniLM-L6-v2)",
            description: "Converts paragraph chunks into 384-dimensional dense vectors stored directly in local SQLite tables for semantic search.",
            metricLabel: "Embedding Latency",
            metricValue: "12ms / chunk",
            tags: "PyTorch · Vector Index · Chunk Segmentation"
        },
        steps: [
            { name: "Source Ingestion & Text Cleansing", description: "Extracting pure text, removing carriage artifacts and non-printable characters.", techBadge: "FastAPI Streaming" },
            { name: "Passage Chunking", description: "Window-based splitting (350 tokens) with 50-token sliding overlap to retain sentence context.", techBadge: "Sliding Window" },
            { name: "Vector Embedding Generation", description: "Mapping passage chunks to latent space representations.", techBadge: "MiniLM-L6 Embeddings" },
            { name: "Relational Knowledge Storage", description: "Persisting document metadata, chunk text, and embeddings into database.", techBadge: "SQLAlchemy / SQLite" }
        ]
    },
    copilot: {
        featureName: "Grounded Knowledge Copilot (RAG)",
        tensorLabel: "SEMANTIC SIMILARITY",
        nlpApp: {
            icon: "🧠",
            category: "RETRIEVAL-AUGMENTED GENERATION",
            name: "Local Agent + Semantic RAG",
            description: "Matches user queries against document chunks using cosine proximity, then passes relevant context to local Ollama (qwen3:1.7b).",
            metricLabel: "Top Cosine Match",
            metricValue: "0.942 Similarity",
            tags: "Ollama LLM · Cosine Distance · Zero Hallucination"
        },
        steps: [
            { name: "Query Vectorization", description: "Transforming user question into dense semantic query vector.", techBadge: "Dense Query Encoder" },
            { name: "k-NN Semantic Passage Search", description: "Calculating cosine distance against all chunks in project workspace.", techBadge: "Cosine Proximity Filter" },
            { name: "Context Assembly & Attribution", description: "Filtering top scoring excerpts and assembling strict evidence prompt.", techBadge: "Citation Anchor" },
            { name: "Local Inference Synthesis", description: "Local model generates grounded response cited directly to document IDs.", techBadge: "Ollama (qwen3:1.7b)" }
        ]
    },
    evidence: {
        featureName: "Evidence & Stance Verification",
        tensorLabel: "STANCE POLARITY",
        nlpApp: {
            icon: "⚖️",
            category: "FACT CHECKING & ENTAILMENT",
            name: "Natural Language Inference (NLI)",
            description: "Computes premise-hypothesis entailment to determine if document evidence SUPPORTS, REFUTES, or remains NEUTRAL regarding a claim.",
            metricLabel: "Confidence Probability",
            metricValue: "96.4% Entailment",
            tags: "Cross-Encoder · Textual Entailment · Audit Trail"
        },
        steps: [
            { name: "Claim Decomposition", description: "Extracting testable propositions and key assertions from input.", techBadge: "Claim Parser" },
            { name: "Corpus Retrieval", description: "Querying indexed documents for topical candidate passages.", techBadge: "Vector Filter" },
            { name: "Cross-Passage Stance Alignment", description: "Evaluating directional agreement between claim and retrieved passage.", techBadge: "Stance Classifier" },
            { name: "Confidence & Citation Scored", description: "Assigning numerical support score and presenting exact excerpt snippet.", techBadge: "Verification Matrix" }
        ]
    },
    webImport: {
        featureName: "Web Research & Scraping",
        tensorLabel: "HTML CLEANING & DIGEST",
        nlpApp: {
            icon: "🌐",
            category: "WEB INTELLIGENCE",
            name: "DOM Cleaner & Article Extractor",
            description: "Scrapes web pages, eliminates navigational boilerplates, ads, and scripts, extracting pure journalistic and academic text.",
            metricLabel: "Cleaning Efficiency",
            metricValue: "98% Noise Reduced",
            tags: "BeautifulSoup4 · Readability · Feed Parser"
        },
        steps: [
            { name: "HTTP Request & Fetch", description: "Retrieving HTML source with standard browser headers.", techBadge: "HTTPX Client" },
            { name: "DOM Traversal & Content Isolation", description: "Stripping navigation, footers, scripts, and extracting article body.", techBadge: "BeautifulSoup4" },
            { name: "Structural Chunking & Indexing", description: "Breaking article into queryable passages for the active project.", techBadge: "Local Vectorizer" }
        ]
    },
    researchCompare: {
        featureName: "Cross-Paper Comparative Intelligence",
        tensorLabel: "GAP & METRIC MINING",
        nlpApp: {
            icon: "📊",
            category: "RESEARCH SYNTHESIS",
            name: "Automated Literature Reviewer",
            description: "Cross-compares multiple scientific publications to highlight contrasting methodologies, common baseline results, and unaddressed gaps.",
            metricLabel: "Comparative Dimensions",
            metricValue: "Method / Finding / Gap",
            tags: "Information Extraction · Matrix Synthesis · Gap Discovery"
        },
        steps: [
            { name: "Sectional Segmentation", description: "Detecting Abstract, Methodology, Results, and Limitation sections.", techBadge: "Heuristic Heading Parser" },
            { name: "Metric & Claim Isolation", description: "Extracting accuracy numbers, datasets, and algorithmic architectures.", techBadge: "Pattern & Regex Extractor" },
            { name: "Comparative Matrix Synthesis", description: "Generating unified cross-document matrix for research decision making.", techBadge: "Structured Matrix" }
        ]
    },
    reportGen: {
        featureName: "Report Studio Synthesizer",
        tensorLabel: "MARKDOWN SYNTHESIS",
        nlpApp: {
            icon: "📝",
            category: "DOCUMENT GENERATION",
            name: "Automated Executive Briefing",
            description: "Compiles all project documents, extracted findings, and metrics into clean publication-ready Markdown reports.",
            metricLabel: "Formatting Output",
            metricValue: "Instant MD / PDF",
            tags: "Markdown Compiler · Literature Synthesis · Executive Brief"
        },
        steps: [
            { name: "Project Corpus Aggregation", description: "Loading all indexed documents and extracted entity records.", techBadge: "Database Query" },
            { name: "Sectional Outlining", description: "Structuring report into Executive Summary, Findings, and Evaluation.", techBadge: "Outline Builder" },
            { name: "Final Document Compilation", description: "Drafting coherent summary report formatted for export.", techBadge: "Markdown Synthesizer" }
        ]
    },
    crewResearch: {
        featureName: "Multi-Agent Research Collaboration Team",
        tensorLabel: "SEQUENTIAL AGENT PIPELINE",
        nlpApp: {
            icon: "🤖",
            category: "AGENTIC MULTI-ROLE ORCHESTRATION",
            name: "Sequential Multi-Agent Architecture",
            description: "Orchestrates 4 specialized AI agents (Researcher, Reviewer, Experiment Designer, Statistician) with accumulated contextual handoffs.",
            metricLabel: "Agent Alignment Score",
            metricValue: "98.7% Consensus",
            tags: "Multi-Agent · Sequential Pipeline · Context Handoff · Collaborative AI"
        },
        steps: [
            { name: "1. Literature & Concept Survey", description: "Researcher mines papers, key entities, factual claims, and unaddressed gaps.", techBadge: "Research Analyst" },
            { name: "2. Methodology Audit & Critique", description: "Reviewer detects potential biases, methodological weaknesses, and scores rigor.", techBadge: "Critical Reviewer" },
            { name: "3. Experimental Protocol Formulation", description: "Designer defines hypothesis, independent/dependent variables, and multi-stage testing.", techBadge: "Experiment Designer" },
            { name: "4. Statistical Validation Framework", description: "Statistician constructs sample sizing, power analysis, hypothesis tests, and validation gates.", techBadge: "Statistical Analyst" }
        ]
    }
};

let activeAnimationId: number | null = null;

export function initVisualizer() {
    const hudToggle = document.getElementById('scifi-hud-toggle');
    const closeBtn = document.getElementById('nexus-close-btn');
    const nexus = document.getElementById('scifi-nlp-nexus');

    if (hudToggle && nexus) {
        hudToggle.addEventListener('click', () => {
            const isCollapsed = nexus.classList.contains('collapsed');
            if (isCollapsed) {
                nexus.classList.remove('collapsed');
                document.body.classList.add('hud-open');
                hudToggle.classList.add('active');
            } else {
                nexus.classList.add('collapsed');
                document.body.classList.remove('hud-open');
                hudToggle.classList.remove('active');
            }
        });
    }

    if (closeBtn && nexus) {
        closeBtn.addEventListener('click', () => {
            nexus.classList.add('collapsed');
            document.body.classList.remove('hud-open');
            hudToggle?.classList.remove('active');
        });
    }

    // Default open state
    document.body.classList.add('hud-open');
    startCanvasAnimation('STANDBY');
}

export function triggerVisualizerPipeline(presetKey: keyof typeof PIPELINE_PRESETS, autoProgress = true) {
    const preset = PIPELINE_PRESETS[presetKey];
    if (!preset) return;

    // Ensure HUD is visible
    const nexus = document.getElementById('scifi-nlp-nexus');
    const hudToggle = document.getElementById('scifi-hud-toggle');
    if (nexus && nexus.classList.contains('collapsed')) {
        nexus.classList.remove('collapsed');
        document.body.classList.add('hud-open');
        hudToggle?.classList.add('active');
    }

    // Update Header Badge & Telemetry
    const modeBadge = document.getElementById('nexus-mode-badge');
    const tensorBadge = document.getElementById('hud-tensor-state');
    const featureNameEl = document.getElementById('vis-feature-name');
    const stepTextEl = document.getElementById('nexus-step-text');

    if (modeBadge) modeBadge.innerText = "ACTIVE";
    if (tensorBadge) tensorBadge.innerText = preset.tensorLabel || "PROCESSING";
    if (featureNameEl) featureNameEl.innerText = preset.featureName;
    if (stepTextEl) stepTextEl.innerText = `RUNNING // ${preset.featureName.toUpperCase()}`;

    // Update NLP Application Info Card
    const iconEl = document.getElementById('nlp-app-icon');
    const catEl = document.getElementById('nlp-app-category');
    const nameEl = document.getElementById('nlp-app-name');
    const descEl = document.getElementById('nlp-app-desc');
    const labelEl = document.getElementById('widget-label');
    const scoreEl = document.getElementById('widget-score');
    const tagEl = document.getElementById('nlp-app-tag');

    if (iconEl) iconEl.innerText = preset.nlpApp.icon;
    if (catEl) catEl.innerText = preset.nlpApp.category;
    if (nameEl) nameEl.innerText = preset.nlpApp.name;
    if (descEl) descEl.innerText = preset.nlpApp.description;
    if (labelEl) labelEl.innerText = preset.nlpApp.metricLabel;
    if (scoreEl) scoreEl.innerText = preset.nlpApp.metricValue;
    if (tagEl) tagEl.innerText = preset.nlpApp.tags;

    // Render Steps List
    const stepsListEl = document.getElementById('pipeline-steps-list');
    if (stepsListEl) {
        stepsListEl.innerHTML = preset.steps.map((step, idx) => `
            <div class="pipeline-step-item ${idx === 0 ? 'active' : ''}" id="p-step-${idx}">
                <div class="pipeline-step-dot">${idx + 1}</div>
                <div class="pipeline-step-body">
                    <div class="pipeline-step-name">${step.name}</div>
                    <div class="pipeline-step-desc">${step.description}</div>
                    <span class="pipeline-step-badge">${step.techBadge}</span>
                </div>
            </div>
        `).join('');
    }

    // Set Progress and start canvas waveform
    setProgress(15);
    startCanvasAnimation('ACTIVE');

    if (autoProgress) {
        simulatePipelineSteps(preset.steps.length);
    }
}

export function setStepActive(stepIndex: number, totalSteps: number) {
    for (let i = 0; i < totalSteps; i++) {
        const el = document.getElementById(`p-step-${i}`);
        if (!el) continue;
        if (i < stepIndex) {
            el.className = 'pipeline-step-item complete';
        } else if (i === stepIndex) {
            el.className = 'pipeline-step-item active';
        } else {
            el.className = 'pipeline-step-item';
        }
    }
    const percent = Math.min(100, Math.round(((stepIndex + 1) / totalSteps) * 100));
    setProgress(percent);
}

export function completePipeline() {
    setProgress(100);
    const modeBadge = document.getElementById('nexus-mode-badge');
    const stepTextEl = document.getElementById('nexus-step-text');
    if (modeBadge) modeBadge.innerText = "COMPLETE";
    if (stepTextEl) stepTextEl.innerText = "STANDBY // EXECUTION COMPLETED";

    // Mark all steps complete
    document.querySelectorAll('.pipeline-step-item').forEach(node => {
        node.className = 'pipeline-step-item complete';
    });

    startCanvasAnimation('STANDBY');
    setTimeout(() => {
        if (modeBadge) modeBadge.innerText = "IDLE";
    }, 4000);
}

function setProgress(percent: number) {
    const bar = document.getElementById('nexus-progress-fill');
    if (bar) bar.style.width = `${percent}%`;
}

function simulatePipelineSteps(totalSteps: number) {
    let current = 0;
    const interval = setInterval(() => {
        current++;
        if (current < totalSteps) {
            setStepActive(current, totalSteps);
        } else {
            clearInterval(interval);
            completePipeline();
        }
    }, 600);
}

// ═══════════════════════════════════════════════════════════════
// MINIMALIST CANVAS VISUALIZER (Harmonious Theme Waves & Nodes)
// ═══════════════════════════════════════════════════════════════
function startCanvasAnimation(state: 'STANDBY' | 'ACTIVE') {
    const canvas = document.getElementById('scifi-canvas') as HTMLCanvasElement | null;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (activeAnimationId) {
        cancelAnimationFrame(activeAnimationId);
    }

    let frame = 0;
    const width = canvas.width;
    const height = canvas.height;

    // Node particles
    const nodes: { x: number; y: number; vx: number; vy: number; radius: number }[] = [];
    for (let i = 0; i < 24; i++) {
        nodes.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.8,
            vy: (Math.random() - 0.5) * 0.8,
            radius: Math.random() * 2 + 1.5
        });
    }

    const renderContext = ctx;

    function render() {
        frame++;
        renderContext.clearRect(0, 0, width, height);

        // Background fill
        renderContext.fillStyle = '#2F0909';
        renderContext.fillRect(0, 0, width, height);

        // Subtle Grid
        renderContext.strokeStyle = 'rgba(255, 246, 205, 0.04)';
        renderContext.lineWidth = 1;
        const gridSize = 24;
        for (let x = 0; x < width; x += gridSize) {
            renderContext.beginPath();
            renderContext.moveTo(x, 0);
            renderContext.lineTo(x, height);
            renderContext.stroke();
        }
        for (let y = 0; y < height; y += gridSize) {
            renderContext.beginPath();
            renderContext.moveTo(0, y);
            renderContext.lineTo(width, y);
            renderContext.stroke();
        }

        // Draw Sine Neural Waves
        const amplitude = state === 'ACTIVE' ? 24 : 10;
        const frequency = state === 'ACTIVE' ? 0.03 : 0.015;
        const speed = state === 'ACTIVE' ? 0.08 : 0.03;

        // Wave 1: Crimson core
        renderContext.beginPath();
        renderContext.strokeStyle = '#C0152A';
        renderContext.lineWidth = 2;
        for (let x = 0; x < width; x++) {
            const y = height / 2 + Math.sin(x * frequency + frame * speed) * amplitude;
            if (x === 0) renderContext.moveTo(x, y);
            else renderContext.lineTo(x, y);
        }
        renderContext.stroke();

        // Wave 2: Gold pure
        renderContext.beginPath();
        renderContext.strokeStyle = 'rgba(188, 123, 38, 0.7)';
        renderContext.lineWidth = 1.5;
        for (let x = 0; x < width; x++) {
            const y = height / 2 + Math.cos(x * frequency * 1.2 - frame * speed * 0.9) * (amplitude * 0.8);
            if (x === 0) renderContext.moveTo(x, y);
            else renderContext.lineTo(x, y);
        }
        renderContext.stroke();

        // Render connected nodes
        for (let i = 0; i < nodes.length; i++) {
            const node = nodes[i];
            node.x += node.vx * (state === 'ACTIVE' ? 1.5 : 0.6);
            node.y += node.vy * (state === 'ACTIVE' ? 1.5 : 0.6);

            if (node.x < 0 || node.x > width) node.vx *= -1;
            if (node.y < 0 || node.y > height) node.vy *= -1;

            // Connect nearby nodes
            for (let j = i + 1; j < nodes.length; j++) {
                const other = nodes[j];
                const dist = Math.hypot(node.x - other.x, node.y - other.y);
                if (dist < 55) {
                    renderContext.strokeStyle = `rgba(212, 154, 85, ${0.35 * (1 - dist / 55)})`;
                    renderContext.lineWidth = 0.8;
                    renderContext.beginPath();
                    renderContext.moveTo(node.x, node.y);
                    renderContext.lineTo(other.x, other.y);
                    renderContext.stroke();
                }
            }

            // Draw node point
            renderContext.fillStyle = state === 'ACTIVE' ? '#FFF6CD' : '#BC7B26';
            renderContext.beginPath();
            renderContext.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            renderContext.fill();
        }

        activeAnimationId = requestAnimationFrame(render);
    }

    render();
}
