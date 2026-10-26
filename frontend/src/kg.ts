/**
 * CIPHER — Knowledge Graph Visualization & Interactive Exploration Module
 * Powered by Cytoscape.js and REST APIs.
 */

import cytoscape, { type Core, type EventObject } from 'cytoscape';

export const PRESET_PROMPTS: Record<string, { title: string; text: string }> = {
    kg_dl: {
        title: "Knowledge Graphs & Deep Learning Foundations",
        text: "Knowledge graphs structure real-world facts using entities and relationships. Deep learning models extract rich semantic representations from unstructured text. Graph Neural Networks integrate knowledge graphs with deep learning architectures to improve link prediction and relation classification on benchmark datasets."
    },
    crispr: {
        title: "CRISPR-Cas9 Genome Editing Mechanisms",
        text: "CRISPR-Cas9 enables precise genome editing by introducing targeted double-strand breaks at specific DNA sequences. Cas9 endonuclease guided by synthetic single-guide RNA binds target genomic loci. Non-homologous end joining and homology-directed repair mechanisms repair DNA lesions, advancing gene therapy and functional genomics."
    },
    quantum: {
        title: "Quantum Error Correction & Qubit Architectures",
        text: "Quantum computing utilizes superposition and entanglement across quantum bits (qubits) to perform computations exponentially faster than classical computers. Quantum error correction protects quantum information from environmental decoherence and phase noise through surface codes and fault-tolerant stabilizer architectures."
    },
    llm: {
        title: "Transformer Attention & Large Language Models",
        text: "Transformer architectures utilize multi-head self-attention mechanisms to model long-range contextual dependencies across token sequences. Large Language Models trained on massive text corpora demonstrate strong zero-shot generalization, prompt reasoning, and structured entity extraction across scientific benchmarks."
    }
};

// Type definitions matching Backend Pydantic Schemas
export interface KGNodeData {
    id: string;
    label: string;
    type: string;
    mention_count: number;
    document_count: number;
    project_id?: number;
}

export interface KGEdgeData {
    id: string;
    source: string;
    target: string;
    label: string;
    confidence: number;
    evidence_count: number;
    review_status: string;
    polarity: string;
    modality: string;
}

export interface KGGraphResponse {
    nodes: Array<{ data: KGNodeData }>;
    edges: Array<{ data: KGEdgeData }>;
    meta: {
        node_count: number;
        edge_count: number;
        truncated: boolean;
        project_id?: number;
    };
}

export interface KGEvidenceItem {
    id: number;
    document_id: number;
    document_title?: string;
    document_filename?: string;
    chunk_id?: number;
    page_num?: number;
    char_start?: number;
    char_end?: number;
    sentence_text: string;
    confidence: number;
    extraction_method: string;
    verification_status: string;
    created_at?: string;
}

export interface KGRelationDetail {
    id: number;
    source_id: number;
    source_name: string;
    source_type: string;
    target_id: number;
    target_name: string;
    target_type: string;
    predicate: string;
    confidence: number;
    polarity: string;
    modality: string;
    review_status: string;
    evidence: KGEvidenceItem[];
}

export interface KGEntityDetail {
    id: number;
    canonical_name: string;
    entity_type: string;
    normalized_key: string;
    aliases: string[];
    description?: string;
    mention_count: number;
    document_count: number;
    outgoing_relations: Array<{
        relation_id: number;
        predicate: string;
        target_id: number;
        target_name: string;
        target_type: string;
        confidence: number;
    }>;
    incoming_relations: Array<{
        relation_id: number;
        predicate: string;
        source_id: number;
        source_name: string;
        source_type: string;
        confidence: number;
    }>;
    documents: Array<{
        document_id: number;
        title: string;
        filename?: string;
    }>;
}

const API_BASE = 'http://localhost:8000';

let cy: Core | null = null;
let currentProjectId: number | null = null;
let activeSelectedRelationId: number | null = null;

// Category color mappings (harmonious palette matching CIPHER aesthetic)
const CATEGORY_COLORS: Record<string, { bg: string; border: string; text: string }> = {
    MODEL: { bg: '#8b5cf6', border: '#7c3aed', text: '#ffffff' },
    METHOD: { bg: '#3b82f6', border: '#2563eb', text: '#ffffff' },
    DATASET: { bg: '#10b981', border: '#059669', text: '#ffffff' },
    BENCHMARK: { bg: '#14b8a6', border: '#0d9488', text: '#ffffff' },
    METRIC: { bg: '#f59e0b', border: '#d97706', text: '#ffffff' },
    TECHNOLOGY: { bg: '#6366f1', border: '#4f46e5', text: '#ffffff' },
    MATERIAL: { bg: '#ec4899', border: '#db2777', text: '#ffffff' },
    CONCEPT: { bg: '#64748b', border: '#475569', text: '#ffffff' },
    PAPER: { bg: '#c0152a', border: '#991b1b', text: '#ffffff' },
    AUTHOR: { bg: '#d97706', border: '#b45309', text: '#ffffff' },
    ORGANIZATION: { bg: '#0284c7', border: '#0369a1', text: '#ffffff' },
    CHEMICAL: { bg: '#06b6d4', border: '#0891b2', text: '#ffffff' },
    GENE: { bg: '#84cc16', border: '#65a30d', text: '#ffffff' },
    DISEASE: { bg: '#ef4444', border: '#dc2626', text: '#ffffff' },
    OTHER: { bg: '#78716c', border: '#57534e', text: '#ffffff' },
};

function getCategoryColor(type: string) {
    return CATEGORY_COLORS[type.toUpperCase()] || CATEGORY_COLORS.OTHER;
}

function escapeHtml(str: string | null | undefined): string {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

/**
 * Initialize Cytoscape container and event listeners
 */
export function initKnowledgeGraph(): void {
    const container = document.getElementById('kg-cytoscape');
    if (!container) return;

    if (cy) {
        cy.destroy();
        cy = null;
    }

    cy = cytoscape({
        container,
        elements: [],
        style: [
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'color': '#ffffff',
                    'font-family': 'Inter, sans-serif',
                    'font-size': '11px',
                    'font-weight': 600,
                    'text-wrap': 'wrap',
                    'text-max-width': '110px',
                    'background-color': '#475569',
                    'border-width': 2,
                    'border-color': '#334155',
                    'width': 'mapData(mention_count, 1, 10, 36, 68)',
                    'height': 'mapData(mention_count, 1, 10, 36, 68)',
                    'transition-property': 'background-color, line-color, target-arrow-color, border-color, opacity',
                    'transition-duration': 0.25
                }
            },
            {
                selector: 'node[type = "MODEL"]',
                style: {
                    'background-color': CATEGORY_COLORS.MODEL.bg,
                    'border-color': CATEGORY_COLORS.MODEL.border
                }
            },
            {
                selector: 'node[type = "METHOD"]',
                style: {
                    'background-color': CATEGORY_COLORS.METHOD.bg,
                    'border-color': CATEGORY_COLORS.METHOD.border
                }
            },
            {
                selector: 'node[type = "DATASET"]',
                style: {
                    'background-color': CATEGORY_COLORS.DATASET.bg,
                    'border-color': CATEGORY_COLORS.DATASET.border
                }
            },
            {
                selector: 'node[type = "BENCHMARK"]',
                style: {
                    'background-color': CATEGORY_COLORS.BENCHMARK.bg,
                    'border-color': CATEGORY_COLORS.BENCHMARK.border
                }
            },
            {
                selector: 'node[type = "METRIC"]',
                style: {
                    'background-color': CATEGORY_COLORS.METRIC.bg,
                    'border-color': CATEGORY_COLORS.METRIC.border
                }
            },
            {
                selector: 'node[type = "TECHNOLOGY"]',
                style: {
                    'background-color': CATEGORY_COLORS.TECHNOLOGY.bg,
                    'border-color': CATEGORY_COLORS.TECHNOLOGY.border
                }
            },
            {
                selector: 'node[type = "MATERIAL"]',
                style: {
                    'background-color': CATEGORY_COLORS.MATERIAL.bg,
                    'border-color': CATEGORY_COLORS.MATERIAL.border
                }
            },
            {
                selector: 'node[type = "PAPER"]',
                style: {
                    'background-color': CATEGORY_COLORS.PAPER.bg,
                    'border-color': CATEGORY_COLORS.PAPER.border,
                    'shape': 'round-rectangle',
                    'width': 50,
                    'height': 34
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'border-width': 4,
                    'border-color': '#c0152a'
                }
            },
            {
                selector: 'edge',
                style: {
                    'label': 'data(label)',
                    'font-family': 'JetBrains Mono, monospace',
                    'font-size': '8px',
                    'font-weight': 600,
                    'color': '#2f0909',
                    'text-background-color': '#fffdf5',
                    'text-background-opacity': 0.92,
                    'text-background-padding': '2px',
                    'text-background-shape': 'roundrectangle',
                    'width': 'mapData(confidence, 0.5, 1.0, 1.5, 3.5)',
                    'line-color': '#a89f91',
                    'target-arrow-color': '#a89f91',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'arrow-scale': 0.9,
                    'opacity': 0.85
                }
            },
            {
                selector: 'edge[polarity = "negative"]',
                style: {
                    'line-color': '#dc2626',
                    'target-arrow-color': '#dc2626',
                    'line-style': 'dashed'
                }
            },
            {
                selector: 'edge[modality = "speculative"]',
                style: {
                    'line-style': 'dotted'
                }
            },
            {
                selector: 'edge:selected',
                style: {
                    'line-color': '#c0152a',
                    'target-arrow-color': '#c0152a',
                    'width': 4,
                    'opacity': 1.0
                }
            },
            {
                selector: '.dimmed',
                style: {
                    'opacity': 0.18
                }
            },
            {
                selector: '.highlighted',
                style: {
                    'opacity': 1.0
                }
            }
        ],
        layout: {
            name: 'cose',
            animate: false
        }
    });

    // Node click handler
    cy.on('tap', 'node', async (evt: EventObject) => {
        const node = evt.target;
        highlightNeighborhood(node);
        const entityId = parseInt(node.id(), 10);
        await inspectEntity(entityId);
    });

    // Edge click handler
    cy.on('tap', 'edge', async (evt: EventObject) => {
        const edge = evt.target;
        highlightEdge(edge);
        const relId = parseInt(edge.id(), 10);
        await inspectRelation(relId);
    });

    // Canvas click handler (clear selection)
    cy.on('tap', (evt: EventObject) => {
        if (evt.target === cy) {
            clearHighlights();
            showEmptyInspector();
        }
    });

    bindToolbarEvents();
    loadDocumentsForBuilder();
    fetchStats();
    loadGraphData();
}

/**
 * Neighborhood focus highlight
 */
function highlightNeighborhood(node: any) {
    if (!cy) return;
    cy.elements().removeClass('highlighted dimmed');
    const neighborhood = node.neighborhood().add(node);
    cy.elements().difference(neighborhood).addClass('dimmed');
    neighborhood.addClass('highlighted');
}

/**
 * Edge focus highlight
 */
function highlightEdge(edge: any) {
    if (!cy) return;
    cy.elements().removeClass('highlighted dimmed');
    const connected = edge.connectedNodes().add(edge);
    cy.elements().difference(connected).addClass('dimmed');
    connected.addClass('highlighted');
}

function clearHighlights() {
    if (!cy) return;
    cy.elements().removeClass('highlighted dimmed');
}

/**
 * Bind Filter Toolbar and Builder Buttons
 */
function bindToolbarEvents() {
    // Search input
    const searchInput = document.getElementById('kg-search-input') as HTMLInputElement | null;
    let debounceTimer: any = null;
    searchInput?.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            loadGraphData();
        }, 350);
    });

    // Type filter
    document.getElementById('kg-filter-type')?.addEventListener('change', () => {
        loadGraphData();
    });

    // Predicate filter
    document.getElementById('kg-filter-predicate')?.addEventListener('change', () => {
        loadGraphData();
    });

    // Confidence slider
    const confSlider = document.getElementById('kg-filter-conf') as HTMLInputElement | null;
    const confVal = document.getElementById('kg-filter-conf-val');
    confSlider?.addEventListener('input', () => {
        if (confVal) confVal.innerText = `${Math.round(parseFloat(confSlider.value) * 100)}%`;
    });
    confSlider?.addEventListener('change', () => {
        loadGraphData();
    });

    // Review status filter
    document.getElementById('kg-filter-review')?.addEventListener('change', () => {
        loadGraphData();
    });

    // Layout selector
    document.getElementById('kg-layout-select')?.addEventListener('change', (e: Event) => {
        const layoutName = (e.target as HTMLSelectElement).value;
        runLayout(layoutName);
    });

    // Fit view
    document.getElementById('btn-kg-fit')?.addEventListener('click', () => {
        cy?.fit(undefined, 30);
    });

    // Reset filters
    document.getElementById('btn-kg-reset')?.addEventListener('click', () => {
        if (searchInput) searchInput.value = '';
        const typeSelect = document.getElementById('kg-filter-type') as HTMLSelectElement | null;
        if (typeSelect) typeSelect.value = '';
        const predSelect = document.getElementById('kg-filter-predicate') as HTMLSelectElement | null;
        if (predSelect) predSelect.value = '';
        if (confSlider) {
            confSlider.value = '0.0';
            if (confVal) confVal.innerText = '0%';
        }
        const revSelect = document.getElementById('kg-filter-review') as HTMLSelectElement | null;
        if (revSelect) revSelect.value = '';
        loadGraphData();
    });

    // Build Knowledge Graph Button (Document Mode)
    document.getElementById('btn-kg-build')?.addEventListener('click', handleBuildGraph);

    // Generator Mode Tabs
    const tabAi = document.getElementById('kg-tab-ai-gen');
    const tabDoc = document.getElementById('kg-tab-doc-gen');
    const panelAi = document.getElementById('kg-mode-ai-panel');
    const panelDoc = document.getElementById('kg-mode-doc-panel');

    tabAi?.addEventListener('click', () => {
        tabAi.classList.add('active');
        tabDoc?.classList.remove('active');
        if (panelAi) panelAi.style.display = 'block';
        if (panelDoc) panelDoc.style.display = 'none';
    });

    tabDoc?.addEventListener('click', () => {
        tabDoc.classList.add('active');
        tabAi?.classList.remove('active');
        if (panelDoc) panelDoc.style.display = 'block';
        if (panelAi) panelAi.style.display = 'none';
    });

    // Preset Pills
    const promptInput = document.getElementById('kg-ai-prompt-input') as HTMLTextAreaElement | null;
    const titleInput = document.getElementById('kg-ai-title-input') as HTMLInputElement | null;
    document.querySelectorAll('.kg-prompt-preset-pill').forEach(pill => {
        pill.addEventListener('click', () => {
            const presetKey = pill.getAttribute('data-preset');
            if (presetKey && PRESET_PROMPTS[presetKey]) {
                if (promptInput) promptInput.value = PRESET_PROMPTS[presetKey].text;
                if (titleInput) titleInput.value = PRESET_PROMPTS[presetKey].title;
                promptInput?.focus();
            }
        });
    });

    // AI Generate Button
    document.getElementById('btn-kg-ai-generate')?.addEventListener('click', handleAiGenerateGraph);

    // Toolbar Controls (Labels, PNG, JSON)
    document.getElementById('btn-kg-toggle-labels')?.addEventListener('click', handleToggleEdgeLabels);
    document.getElementById('btn-kg-export-png')?.addEventListener('click', handleExportGraphPng);
    document.getElementById('btn-kg-export-json')?.addEventListener('click', handleExportGraphJson);

    // Review buttons inside inspector
    document.getElementById('btn-kg-approve-rel')?.addEventListener('click', () => handleRelationReview('approve'));
    document.getElementById('btn-kg-reject-rel')?.addEventListener('click', () => handleRelationReview('reject'));
    document.getElementById('btn-kg-correct-rel')?.addEventListener('click', openRelationCorrectModal);

    // Modal buttons
    document.getElementById('btn-kg-open-merge')?.addEventListener('click', openEntityMergeModal);
    document.getElementById('btn-kg-confirm-merge')?.addEventListener('click', handleConfirmMerge);
    document.getElementById('btn-kg-cancel-merge')?.addEventListener('click', closeEntityMergeModal);
    document.getElementById('btn-kg-confirm-correct')?.addEventListener('click', handleConfirmCorrection);
    document.getElementById('btn-kg-cancel-correct')?.addEventListener('click', closeRelationCorrectModal);
}

/**
 * Fetch and Render Knowledge Graph from API
 */
export async function loadGraphData(projectId?: number): Promise<void> {
    if (projectId !== undefined) {
        currentProjectId = projectId;
    }

    const searchInput = (document.getElementById('kg-search-input') as HTMLInputElement)?.value.trim() || '';
    const entityType = (document.getElementById('kg-filter-type') as HTMLSelectElement)?.value || '';
    const predicate = (document.getElementById('kg-filter-predicate') as HTMLSelectElement)?.value || '';
    const minConf = (document.getElementById('kg-filter-conf') as HTMLInputElement)?.value || '0.0';
    const reviewStatus = (document.getElementById('kg-filter-review') as HTMLSelectElement)?.value || '';

    const params = new URLSearchParams();
    if (currentProjectId) params.append('project_id', String(currentProjectId));
    if (searchInput) params.append('search', searchInput);
    if (entityType) params.append('entity_type', entityType);
    if (predicate) params.append('predicate', predicate);
    if (minConf && parseFloat(minConf) > 0) params.append('min_confidence', minConf);
    if (reviewStatus) params.append('review_status', reviewStatus);
    params.append('limit', '180');

    const loader = document.getElementById('kg-loading-overlay');
    const emptyState = document.getElementById('kg-empty-state');
    if (loader) loader.style.display = 'flex';
    if (emptyState) emptyState.style.display = 'none';

    try {
        const res = await fetch(`${API_BASE}/api/knowledge-graph/graph?${params.toString()}`);
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const data: KGGraphResponse = await res.json();

        if (!cy) return;

        cy.elements().remove();

        const elements: any[] = [];

        // Add nodes
        for (const n of data.nodes) {
            elements.push({
                group: 'nodes',
                data: {
                    id: String(n.data.id),
                    label: n.data.label,
                    type: n.data.type,
                    mention_count: n.data.mention_count || 1,
                    document_count: n.data.document_count || 1
                }
            });
        }

        // Add edges
        for (const e of data.edges) {
            elements.push({
                group: 'edges',
                data: {
                    id: String(e.data.id),
                    source: String(e.data.source),
                    target: String(e.data.target),
                    label: e.data.label,
                    confidence: e.data.confidence,
                    evidence_count: e.data.evidence_count,
                    review_status: e.data.review_status,
                    polarity: e.data.polarity,
                    modality: e.data.modality
                }
            });
        }

        cy.add(elements);

        if (elements.length === 0) {
            if (emptyState) emptyState.style.display = 'flex';
        } else {
            const layoutSelect = (document.getElementById('kg-layout-select') as HTMLSelectElement)?.value || 'cose';
            runLayout(layoutSelect);
        }

        // Update node/edge counter badge
        const badge = document.getElementById('kg-graph-count-badge');
        if (badge) {
            badge.innerText = `${data.meta.node_count} nodes · ${data.meta.edge_count} edges`;
        }

    } catch (err: any) {
        console.error('Failed to load knowledge graph:', err);
        const errToast = document.getElementById('kg-error-toast');
        if (errToast) {
            errToast.innerText = `Graph load error: ${err.message}`;
            errToast.style.display = 'block';
            setTimeout(() => { errToast.style.display = 'none'; }, 4000);
        }
    } finally {
        if (loader) loader.style.display = 'none';
    }
}

/**
 * Run chosen layout algorithm
 */
function runLayout(layoutName: string) {
    if (!cy) return;
    cy.resize();
    const count = cy.nodes().length;
    let options: any = { name: layoutName, animate: true, animationDuration: 400, fit: true, padding: 35 };

    if (layoutName === 'cose') {
        options = {
            name: 'cose',
            animate: count < 100,
            animationDuration: 500,
            fit: true,
            padding: 35,
            nodeRepulsion: () => 4500,
            idealEdgeLength: () => 70,
            edgeElasticity: () => 100,
            gravity: 0.25,
            numIter: 300
        };
    } else if (layoutName === 'breadthfirst') {
        options = {
            name: 'breadthfirst',
            directed: true,
            fit: true,
            padding: 35,
            spacingFactor: 1.25
        };
    } else if (layoutName === 'circle') {
        options = {
            name: 'circle',
            fit: true,
            padding: 35
        };
    } else if (layoutName === 'concentric') {
        options = {
            name: 'concentric',
            fit: true,
            padding: 35,
            concentric: (node: any) => node.data('mention_count') || 1,
            levelWidth: () => 2
        };
    }

    const layout = cy.layout(options);
    layout.on('layoutstop', () => {
        cy?.fit(undefined, 35);
    });
    layout.run();
}

/**
 * Force resize and center graph view (useful when tab becomes visible)
 */
export function resizeKnowledgeGraph(): void {
    if (cy) {
        cy.resize();
        cy.fit(undefined, 35);
    }
}

/**
 * Fetch and update overall Knowledge Graph Statistics
 */
export async function fetchStats(projectId?: number | null): Promise<void> {
    if (projectId !== undefined) {
        currentProjectId = projectId;
    }
    try {
        const url = currentProjectId
            ? `${API_BASE}/api/knowledge-graph/stats?project_id=${currentProjectId}`
            : `${API_BASE}/api/knowledge-graph/stats`;
        const res = await fetch(url);
        if (!res.ok) return;
        const data = await res.json();

        const entEl = document.getElementById('kg-stat-entities');
        const relEl = document.getElementById('kg-stat-relations');
        const paperEl = document.getElementById('kg-stat-papers');
        const evEl = document.getElementById('kg-stat-evidence');
        const revEl = document.getElementById('kg-stat-reviews');

        if (entEl) entEl.innerText = String(data.total_entities || 0);
        if (relEl) relEl.innerText = String(data.total_relations || 0);
        if (paperEl) paperEl.innerText = String(data.papers_represented || 0);
        if (evEl) evEl.innerText = String(data.total_evidence_records || 0);
        if (revEl) revEl.innerText = String(data.pending_reviews || 0);
    } catch (err) {
        console.warn('Failed to fetch KG stats:', err);
    }
}

/**
 * Load documents into the graph builder select dropdown
 */
async function loadDocumentsForBuilder() {
    try {
        const select = document.getElementById('kg-build-doc-select') as HTMLSelectElement | null;
        if (!select) return;

        let docsUrl = `${API_BASE}/api/documents`;
        if (currentProjectId) {
            docsUrl = `${API_BASE}/api/projects/${currentProjectId}/documents`;
        }

        const res = await fetch(docsUrl);
        if (!res.ok) return;
        const docs = await res.json();

        select.innerHTML = '<option value="">Select Ingested Document...</option>' +
            docs.map((d: any) => `<option value="${d.id}">${escapeHtml(d.title || d.filename)} (ID: ${d.id})</option>`).join('');
    } catch {
        // Fallback gracefully
    }
}

/**
 * Trigger Graph Construction on a selected document
 */
async function handleBuildGraph() {
    const docSelect = document.getElementById('kg-build-doc-select') as HTMLSelectElement | null;
    const rebuildCheck = document.getElementById('kg-build-rebuild') as HTMLInputElement | null;
    const buildBtn = document.getElementById('btn-kg-build') as HTMLButtonElement | null;
    const statusMsg = document.getElementById('kg-build-status');

    const docId = docSelect?.value;
    if (!docId) {
        if (statusMsg) {
            statusMsg.innerText = 'Please select a document first.';
            statusMsg.className = 'status-badge status-warning';
        }
        return;
    }

    if (buildBtn) buildBtn.disabled = true;
    if (statusMsg) {
        statusMsg.innerText = 'Extracting entities, relationships & sentence citations...';
        statusMsg.className = 'status-badge status-loading';
    }

    try {
        const res = await fetch(`${API_BASE}/api/knowledge-graph/build`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                document_id: parseInt(docId, 10),
                project_id: currentProjectId || null,
                rebuild: rebuildCheck ? rebuildCheck.checked : true
            })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Build failed');
        }

        const data = await res.json();
        if (statusMsg) {
            statusMsg.innerText = `Build complete! Extracted ${data.entities_extracted} entities, ${data.relations_extracted} relations, ${data.evidence_recorded} evidence citations.`;
            statusMsg.className = 'status-badge status-success';
        }

        // Refresh stats and graph visualization
        await fetchStats();
        await loadGraphData();

    } catch (err: any) {
        if (statusMsg) {
            statusMsg.innerText = `Build error: ${err.message}`;
            statusMsg.className = 'status-badge status-error';
        }
    } finally {
        if (buildBtn) buildBtn.disabled = false;
    }
}

let edgeLabelsVisible = true;

function handleToggleEdgeLabels() {
    if (!cy) return;
    edgeLabelsVisible = !edgeLabelsVisible;
    if (edgeLabelsVisible) {
        cy.style().selector('edge').style('label', 'data(label)').update();
    } else {
        cy.style().selector('edge').style('label', '').update();
    }
    const btn = document.getElementById('btn-kg-toggle-labels');
    if (btn) btn.innerText = edgeLabelsVisible ? '🏷️ Labels: ON' : '🏷️ Labels: OFF';
}

function handleExportGraphPng() {
    if (!cy) return;
    const png64 = cy.png({ bg: '#FFF6CD', full: true, scale: 2 });
    const a = document.createElement('a');
    a.href = png64;
    a.download = `CIPHER_Knowledge_Graph_${Date.now()}.png`;
    a.click();
}

function handleExportGraphJson() {
    if (!cy) return;
    const jsonStr = JSON.stringify(cy.json().elements, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `CIPHER_Knowledge_Graph_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

/**
 * Handle AI Prompt / Free-form Text Knowledge Graph Generation
 */
async function handleAiGenerateGraph() {
    const promptInput = document.getElementById('kg-ai-prompt-input') as HTMLTextAreaElement | null;
    const titleInput = document.getElementById('kg-ai-title-input') as HTMLInputElement | null;
    const generateBtn = document.getElementById('btn-kg-ai-generate') as HTMLButtonElement | null;
    const statusMsg = document.getElementById('kg-ai-status');

    const text = promptInput?.value.trim();
    if (!text || text.length < 10) {
        if (statusMsg) {
            statusMsg.innerText = 'Please enter at least 10 characters to generate a graph.';
            statusMsg.className = 'status-badge status-warning';
            statusMsg.style.display = 'inline-block';
        }
        promptInput?.focus();
        return;
    }

    const title = titleInput?.value.trim() || undefined;

    if (generateBtn) {
        generateBtn.disabled = true;
        generateBtn.innerHTML = `
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 1s linear infinite;">
                <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
                <path d="M12 2a10 10 0 0 1 10 10"></path>
            </svg>
            Extracting Graph...
        `;
    }

    if (statusMsg) {
        statusMsg.innerText = 'AI extracting entities & SPO triplets...';
        statusMsg.className = 'status-badge status-info';
        statusMsg.style.display = 'inline-block';
    }

    try {
        const payload: any = { text, title };
        if (currentProjectId) payload.project_id = currentProjectId;

        const res = await fetch(`${API_BASE}/api/knowledge-graph/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to generate graph');
        }

        const data = await res.json();

        if (statusMsg) {
            const nodeCount = data.nodes ? data.nodes.length : 0;
            const edgeCount = data.edges ? data.edges.length : 0;
            statusMsg.innerText = `Graph Generated! ${nodeCount} Entities · ${edgeCount} SPO Triplets Extracted`;
            statusMsg.className = 'status-badge status-success';
            statusMsg.style.display = 'inline-block';
        }

        // Render nodes and edges into Cytoscape
        if (cy && data.nodes) {
            cy.elements().remove();
            const elements: any[] = [];
            for (const n of data.nodes) {
                elements.push({
                    group: 'nodes',
                    data: {
                        id: String(n.id),
                        label: n.label,
                        type: n.type,
                        mention_count: n.mention_count || 1,
                        document_count: n.document_count || 1
                    }
                });
            }
            for (const e of data.edges) {
                elements.push({
                    group: 'edges',
                    data: {
                        id: String(e.id),
                        source: String(e.source),
                        target: String(e.target),
                        label: e.label,
                        confidence: e.confidence,
                        evidence_count: e.evidence_count,
                        review_status: e.review_status,
                        polarity: e.polarity,
                        modality: e.modality
                    }
                });
            }
            cy.add(elements);
            runLayout('cose');

            const badge = document.getElementById('kg-graph-count-badge');
            if (badge) {
                badge.innerText = `${data.nodes.length} nodes · ${data.edges.length} edges`;
            }
        }

        // Refresh stats & documents list
        await fetchStats(currentProjectId);
        await loadDocumentsForBuilder();

    } catch (err: any) {
        console.error('AI Knowledge Graph generation failed:', err);
        if (statusMsg) {
            statusMsg.innerText = `Error: ${err.message}`;
            statusMsg.className = 'status-badge status-error';
            statusMsg.style.display = 'inline-block';
        }
    } finally {
        if (generateBtn) {
            generateBtn.disabled = false;
            generateBtn.innerHTML = `
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
                Generate Knowledge Graph
            `;
        }
    }
}


/**
 * Inspect selected Entity details
 */
async function inspectEntity(entityId: number) {
    const inspector = document.getElementById('kg-inspector-content');
    if (!inspector) return;

    inspector.innerHTML = `<div class="inspector-loading"><span class="loader-spinner"></span> Loading entity details...</div>`;

    try {
        const res = await fetch(`${API_BASE}/api/knowledge-graph/entities/${entityId}`);
        if (!res.ok) throw new Error('Entity not found');
        const ent: KGEntityDetail = await res.json();

        const color = getCategoryColor(ent.entity_type);
        const aliases = (ent.aliases || []).map(a => `<span class="alias-pill">${escapeHtml(a)}</span>`).join('') || '<span class="text-muted">None</span>';

        const outgoingList = (ent.outgoing_relations || []).map(r => `
            <div class="relation-item-row" data-rel-id="${r.relation_id}">
                <span class="pred-tag">${escapeHtml(r.predicate)}</span>
                <span class="target-name">→ ${escapeHtml(r.target_name)}</span>
                <span class="conf-badge">${Math.round(r.confidence * 100)}%</span>
            </div>
        `).join('') || '<p class="text-muted">No outgoing relations</p>';

        const incomingList = (ent.incoming_relations || []).map(r => `
            <div class="relation-item-row" data-rel-id="${r.relation_id}">
                <span class="target-name">${escapeHtml(r.source_name)} →</span>
                <span class="pred-tag">${escapeHtml(r.predicate)}</span>
                <span class="conf-badge">${Math.round(r.confidence * 100)}%</span>
            </div>
        `).join('') || '<p class="text-muted">No incoming relations</p>';

        const docList = (ent.documents || []).map(d => `
            <div class="doc-badge-chip">📄 ${escapeHtml(d.title || d.filename || 'Document #' + d.document_id)}</div>
        `).join('') || '<p class="text-muted">No linked papers</p>';

        inspector.innerHTML = `
            <div class="entity-inspector-view">
                <div class="entity-header">
                    <span class="entity-type-badge" style="background:${color.bg}; border-color:${color.border};">${escapeHtml(ent.entity_type)}</span>
                    <h3 class="entity-canonical-title">${escapeHtml(ent.canonical_name)}</h3>
                    <p class="entity-normalized-key">Normalized Key: <code>${escapeHtml(ent.normalized_key)}</code></p>
                </div>

                <div class="inspector-metrics-grid">
                    <div class="inspector-metric-card">
                        <span class="metric-num">${ent.mention_count}</span>
                        <span class="metric-lbl">Total Mentions</span>
                    </div>
                    <div class="inspector-metric-card">
                        <span class="metric-num">${ent.document_count}</span>
                        <span class="metric-lbl">Source Papers</span>
                    </div>
                </div>

                <div class="inspector-section">
                    <label class="inspector-label">Aliases & Surface Forms</label>
                    <div class="aliases-container">${aliases}</div>
                </div>

                <div class="inspector-section">
                    <label class="inspector-label">Outgoing Claims (${ent.outgoing_relations.length})</label>
                    <div class="relations-scroll">${outgoingList}</div>
                </div>

                <div class="inspector-section">
                    <label class="inspector-label">Incoming Claims (${ent.incoming_relations.length})</label>
                    <div class="relations-scroll">${incomingList}</div>
                </div>

                <div class="inspector-section">
                    <label class="inspector-label">Supporting Papers</label>
                    <div class="docs-container">${docList}</div>
                </div>
            </div>
        `;

        // Bind clicks on relation rows to inspect that edge
        inspector.querySelectorAll('.relation-item-row').forEach(row => {
            row.addEventListener('click', () => {
                const rId = parseInt(row.getAttribute('data-rel-id') || '0', 10);
                if (rId) inspectRelation(rId);
            });
        });

    } catch (err: any) {
        inspector.innerHTML = `<div class="inspector-error"><p>${escapeHtml(err.message)}</p></div>`;
    }
}

/**
 * Inspect selected Relationship and its verifiable source sentence evidence
 */
async function inspectRelation(relationId: number) {
    const inspector = document.getElementById('kg-inspector-content');
    if (!inspector) return;

    activeSelectedRelationId = relationId;
    inspector.innerHTML = `<div class="inspector-loading"><span class="loader-spinner"></span> Loading relation evidence...</div>`;

    try {
        const res = await fetch(`${API_BASE}/api/knowledge-graph/relations/${relationId}`);
        if (!res.ok) throw new Error('Relationship not found');
        const rel: KGRelationDetail = await res.json();

        const evidenceCards = (rel.evidence || []).map((ev) => `
            <div class="evidence-citation-card">
                <div class="evidence-header-row">
                    <span class="evidence-doc-tag">📄 ${escapeHtml(ev.document_title || ev.document_filename || 'Document #' + ev.document_id)}</span>
                    ${ev.page_num ? `<span class="evidence-loc-tag">Page ${ev.page_num}</span>` : ''}
                    ${ev.chunk_id ? `<span class="evidence-loc-tag">Chunk #${ev.chunk_id}</span>` : ''}
                </div>
                <blockquote class="evidence-quote">
                    “${escapeHtml(ev.sentence_text)}”
                </blockquote>
                <div class="evidence-footer-row">
                    <span class="evidence-method">Method: ${escapeHtml(ev.extraction_method)}</span>
                    <span class="evidence-conf">Provenance Score: ${Math.round((ev.confidence || 0.9) * 100)}%</span>
                </div>
            </div>
        `).join('') || '<p class="text-muted">No evidence records available.</p>';

        const statusClass = rel.review_status === 'approved' ? 'status-approved' :
                            rel.review_status === 'rejected' ? 'status-rejected' : 'status-unreviewed';

        inspector.innerHTML = `
            <div class="relation-inspector-view">
                <div class="relation-header">
                    <span class="review-status-pill ${statusClass}">${escapeHtml(rel.review_status.toUpperCase())}</span>
                    <div class="relation-triplet">
                        <span class="rel-entity source">${escapeHtml(rel.source_name)}</span>
                        <span class="rel-arrow">─[ ${escapeHtml(rel.predicate)} ]─▶</span>
                        <span class="rel-entity target">${escapeHtml(rel.target_name)}</span>
                    </div>
                </div>

                <div class="inspector-metrics-grid">
                    <div class="inspector-metric-card">
                        <span class="metric-num">${Math.round(rel.confidence * 100)}%</span>
                        <span class="metric-lbl">Extraction Confidence</span>
                    </div>
                    <div class="inspector-metric-card">
                        <span class="metric-num">${rel.evidence.length}</span>
                        <span class="metric-lbl">Evidence Citations</span>
                    </div>
                    <div class="inspector-metric-card">
                        <span class="metric-num">${escapeHtml(rel.polarity)}</span>
                        <span class="metric-lbl">Polarity</span>
                    </div>
                    <div class="inspector-metric-card">
                        <span class="metric-num">${escapeHtml(rel.modality)}</span>
                        <span class="metric-lbl">Modality</span>
                    </div>
                </div>

                <!-- Review Actions Bar -->
                <div class="relation-review-actions">
                    <button class="btn-review btn-approve" id="btn-kg-approve-rel" title="Approve this claim">✓ Approve</button>
                    <button class="btn-review btn-reject" id="btn-kg-reject-rel" title="Reject invalid claim">✗ Reject</button>
                    <button class="btn-review btn-correct" id="btn-kg-correct-rel" title="Change predicate">✎ Correct</button>
                </div>

                <!-- Traceable Evidence Citations -->
                <div class="inspector-section">
                    <label class="inspector-label">Traceable Source Evidence (${rel.evidence.length})</label>
                    <div class="evidence-citations-container">
                        ${evidenceCards}
                    </div>
                </div>
            </div>
        `;

        // Rebind review buttons
        document.getElementById('btn-kg-approve-rel')?.addEventListener('click', () => handleRelationReview('approve'));
        document.getElementById('btn-kg-reject-rel')?.addEventListener('click', () => handleRelationReview('reject'));
        document.getElementById('btn-kg-correct-rel')?.addEventListener('click', openRelationCorrectModal);

    } catch (err: any) {
        inspector.innerHTML = `<div class="inspector-error"><p>${escapeHtml(err.message)}</p></div>`;
    }
}

function showEmptyInspector() {
    const inspector = document.getElementById('kg-inspector-content');
    if (!inspector) return;
    activeSelectedRelationId = null;
    inspector.innerHTML = `
        <div class="inspector-empty-state">
            <div class="inspector-empty-icon">🔍</div>
            <h4>Evidence & Entity Inspector</h4>
            <p>Click any node or directed relationship edge in the graph to inspect canonical properties, related concepts, and exact source sentence citations.</p>
        </div>
    `;
}

/**
 * Handle Relationship Review (Approve / Reject)
 */
async function handleRelationReview(action: 'approve' | 'reject') {
    if (!activeSelectedRelationId) return;

    try {
        const res = await fetch(`${API_BASE}/api/knowledge-graph/relations/${activeSelectedRelationId}/review`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action,
                reviewer_notes: `Marked as ${action} from CIPHER Dashboard`
            })
        });

        if (!res.ok) throw new Error('Failed to update review status');
        await inspectRelation(activeSelectedRelationId);
        await fetchStats();
        await loadGraphData();
    } catch (err: any) {
        alert(`Review action failed: ${err.message}`);
    }
}

/**
 * Open Relation Correction Modal
 */
function openRelationCorrectModal() {
    const modal = document.getElementById('kg-correct-modal');
    if (modal) modal.style.display = 'flex';
}

function closeRelationCorrectModal() {
    const modal = document.getElementById('kg-correct-modal');
    if (modal) modal.style.display = 'none';
}

async function handleConfirmCorrection() {
    if (!activeSelectedRelationId) return;
    const predSelect = document.getElementById('kg-correct-predicate-select') as HTMLSelectElement | null;
    const newPred = predSelect?.value;
    if (!newPred) return;

    try {
        const res = await fetch(`${API_BASE}/api/knowledge-graph/relations/${activeSelectedRelationId}/review`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: 'correct',
                new_predicate: newPred,
                reviewer_notes: `Corrected predicate to ${newPred}`
            })
        });

        if (!res.ok) throw new Error('Correction failed');
        closeRelationCorrectModal();
        await inspectRelation(activeSelectedRelationId);
        await loadGraphData();
    } catch (err: any) {
        alert(`Failed to correct relation: ${err.message}`);
    }
}

/**
 * Open Entity Merge Modal
 */
async function openEntityMergeModal() {
    const modal = document.getElementById('kg-merge-modal');
    const srcSelect = document.getElementById('kg-merge-source-select') as HTMLSelectElement | null;
    const tgtSelect = document.getElementById('kg-merge-target-select') as HTMLSelectElement | null;
    if (!modal || !srcSelect || !tgtSelect) return;

    // Populate entity dropdowns with current nodes in graph
    if (cy) {
        const nodes = cy.nodes().map(n => ({
            id: n.id(),
            name: n.data('label'),
            type: n.data('type')
        }));

        const options = '<option value="">Select Entity...</option>' +
            nodes.map(n => `<option value="${n.id}">${escapeHtml(n.name)} [${n.type}]</option>`).join('');

        srcSelect.innerHTML = options;
        tgtSelect.innerHTML = options;
    }

    modal.style.display = 'flex';
}

function closeEntityMergeModal() {
    const modal = document.getElementById('kg-merge-modal');
    if (modal) modal.style.display = 'none';
}

async function handleConfirmMerge() {
    const srcSelect = document.getElementById('kg-merge-source-select') as HTMLSelectElement | null;
    const tgtSelect = document.getElementById('kg-merge-target-select') as HTMLSelectElement | null;
    const confirmBtn = document.getElementById('btn-kg-confirm-merge') as HTMLButtonElement | null;

    const srcId = parseInt(srcSelect?.value || '0', 10);
    const tgtId = parseInt(tgtSelect?.value || '0', 10);

    if (!srcId || !tgtId || srcId === tgtId) {
        alert('Please select two distinct entities to merge.');
        return;
    }

    if (!confirm('Are you sure you want to merge these entities? All relations and mentions from the source entity will be permanently rerouted to the target entity.')) {
        return;
    }

    if (confirmBtn) confirmBtn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/knowledge-graph/entities/merge`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_entity_id: srcId,
                target_entity_id: tgtId
            })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Merge failed');
        }

        closeEntityMergeModal();
        await fetchStats();
        await loadGraphData();
        showEmptyInspector();
        alert('Entities successfully merged and relations rerouted.');
    } catch (err: any) {
        alert(`Merge failed: ${err.message}`);
    } finally {
        if (confirmBtn) confirmBtn.disabled = false;
    }
}
