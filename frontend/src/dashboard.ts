import { initVisualizer, triggerVisualizerPipeline } from './visualizer';
import { callCipher, parseJSON, appendLog, typeText, memory, stats, checkOllamaHealth } from './api/gemini';
import { initKnowledgeGraph, loadGraphData, fetchStats, resizeKnowledgeGraph } from './kg';
import { initCrewPipeline } from './crew';

let cyInitialized = false;

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    const user = sessionStorage.getItem('gemini_user') || 'Operator';
    initVisualizer();
    updateWorkspaceTopbarBadge('All Projects Workspace');
    void loadKnowledgeLibraryDocs();

    memory.userName = user;

    // Check Ollama connection status
    const health = await checkOllamaHealth();
    const statusEl = document.getElementById('ollama-status');
    const statusDot = document.getElementById('ollama-status-dot');
    const statusText = document.getElementById('ollama-status-text');
    const modelName = document.getElementById('ollama-model-name');

    if (statusEl && statusDot && statusText) {
        if (health.status === 'connected') {
            statusDot.style.background = '#34d399';
            statusDot.style.boxShadow = '0 0 8px rgba(52,211,153,0.5)';
            statusText.innerText = 'CONNECTED';
            statusText.style.color = '#34d399';
            appendLog('success', `Ollama connected — Model: ${health.model}`);
        } else if (health.status === 'model_missing') {
            statusDot.style.background = '#fbbf24';
            statusDot.style.boxShadow = '0 0 8px rgba(251,191,36,0.5)';
            statusText.innerText = 'MODEL MISSING';
            statusText.style.color = '#fbbf24';
            appendLog('error', `Ollama running but model not found: ${health.message}`);
        } else {
            statusDot.style.background = '#ef4444';
            statusDot.style.boxShadow = '0 0 8px rgba(239,68,68,0.5)';
            statusText.innerText = 'DISCONNECTED';
            statusText.style.color = '#ef4444';
            appendLog('error', `Ollama not reachable: ${health.message}`);
        }
    }
    if (modelName) modelName.innerText = health.model || 'qwen3:1.7b';
    
    appendLog('cmd', `Context loaded for user: ${user}`);

    // Set avatar initials
    const avatar = document.getElementById('topbar-avatar');
    if (avatar) avatar.innerText = user.substring(0, 2).toUpperCase();

    // Profile handling
    const profileUserInput = document.getElementById('profile-username') as HTMLInputElement;
    if (profileUserInput && user !== 'Operator') profileUserInput.value = user;

    document.getElementById('btn-save-profile')?.addEventListener('click', async () => {
        const n = profileUserInput?.value.trim();
        
        if (n) {
            sessionStorage.setItem('gemini_user', n);
            memory.userName = n;
            appendLog('success', `Operator set to: ${n}`);
        } else {
            sessionStorage.removeItem('gemini_user');
            memory.userName = 'Operator';
        }

        // Re-check Ollama health
        const h = await checkOllamaHealth();
        if (statusDot && statusText) {
            if (h.status === 'connected') {
                statusDot.style.background = '#34d399';
                statusText.innerText = 'CONNECTED';
                statusText.style.color = '#34d399';
            } else {
                statusDot.style.background = '#ef4444';
                statusText.innerText = 'DISCONNECTED';
                statusText.style.color = '#ef4444';
            }
        }
        appendLog('success', 'Settings saved.');
    });

    // Bind global buttons
    document.getElementById('logout-btn')?.addEventListener('click', () => {
        sessionStorage.clear();
        window.location.href = '/';
    });
    
    document.getElementById('mind-toggle')?.addEventListener('click', () => {
        document.getElementById('mind-panel')?.classList.toggle('open');
    });
    
    document.getElementById('voice-btn')?.addEventListener('click', toggleVoice);
    
    document.getElementById('btn-upload-doc')?.addEventListener('click', featureUploadDoc);
    document.getElementById('btn-nlp-analyze')?.addEventListener('click', featureNlpAnalyze);
    document.getElementById('btn-import-url')?.addEventListener('click', () => featureImportUrl());
    document.getElementById('btn-web-search')?.addEventListener('click', () => featureWebSearch());
    document.getElementById('import-url')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            featureImportUrl();
        }
    });

    // Web search topic preset chips
    document.querySelectorAll('.web-topic-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.getAttribute('data-query');
            const input = document.getElementById('import-url') as HTMLInputElement | null;
            if (input && query) {
                input.value = query;
                featureWebSearch(query);
            }
        });
    });
    document.getElementById('btn-analyze-papers')?.addEventListener('click', featureCompareResearch);
    document.getElementById('btn-copilot-ask')?.addEventListener('click', featureCopilot);
    document.getElementById('btn-evidence-check')?.addEventListener('click', featureEvidence);
    document.getElementById('btn-generate-report')?.addEventListener('click', featureReport);
    document.getElementById('btn-refresh-docs')?.addEventListener('click', loadKnowledgeLibraryDocs);
    document.getElementById('btn-extract-paper-struct')?.addEventListener('click', featureExtractPaperStruct);
    document.getElementById('btn-copilot-sop')?.addEventListener('click', featureCopilotSop);
    document.getElementById('btn-copy-report')?.addEventListener('click', copyReportMarkdown);
    document.getElementById('btn-download-report-md')?.addEventListener('click', downloadReportMarkdown);

    // Copilot preset chips - click to set query and run
    document.querySelectorAll('.copilot-presets .preset-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.getAttribute('data-query');
            const input = document.getElementById('copilot-query') as HTMLInputElement | null;
            if (input && query) {
                input.value = query;
                featureCopilot();
            }
        });
    });

    // Keep optional legacy workspace actions available to command routing.
    void downloadScrapeCSV; void featureCompare; void featureReviewer;
    await refreshProjects();
    initCrewPipeline();


    // document.getElementById('btn-browse')?.addEventListener('click', featureBrowse);
    // document.getElementById('btn-scrape')?.addEventListener('click', featureScrape);
    // document.getElementById('btn-task')?.addEventListener('click', featureMission);
    // document.getElementById('btn-form')?.addEventListener('click', featureForm);
    // document.getElementById('btn-compare')?.addEventListener('click', featureCompare);
    // document.getElementById('btn-workflow')?.addEventListener('click', featureWorkflow);
    // document.getElementById('btn-reviewer')?.addEventListener('click', featureReviewer);
    // document.getElementById('scrape-csv-btn')?.addEventListener('click', downloadScrapeCSV);

    // Chat listeners
    document.getElementById('btn-chat-send')?.addEventListener('click', featureChat);
    document.getElementById('chat-input')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            featureChat();
        }
    });

    // Tab switching logic
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.dashboard-section');

    const titleMap: Record<string, string> = {
        'agent-dashboard': 'Overview',
        'knowledge-library': 'Knowledge Library',
        'import-web': 'Import & Web Research',
        'research-intelligence': 'Research Intelligence',
        'evidence-explorer': 'Evidence Explorer',
        'knowledge-graph': 'Knowledge Graph Engine',
        'knowledge-copilot': 'Knowledge Copilot',
        'nlp-lab': 'NLP Lab',
        'agent-workflows': 'Agent Workflows',
        'multi-agent-crew': 'Multi-Agent Research Collaboration Team',
        'report-studio': 'Report Studio',
        'about-project': 'About CIPHER & User Guide',
        'settings-privacy': 'Settings & Privacy'
    };

    function activateTab(targetId: string) {
        if (targetId === 'research-projects') {
            targetId = 'agent-dashboard';
            window.location.hash = 'agent-dashboard';
        }
        navLinks.forEach(link => link.classList.remove('active'));
        sections.forEach(section => section.classList.remove('active'));

        const activeLink = document.querySelector(`.nav-link[data-target="${targetId}"]`);
        const activeSection = document.getElementById(targetId);

        if (activeLink) activeLink.classList.add('active');
        if (activeSection) activeSection.classList.add('active');
        
        // Update topbar title
        const topTitle = document.getElementById('topbar-section-title');
        if (topTitle) topTitle.innerText = titleMap[targetId] || 'Dashboard';

        // Update overview stats
        if (targetId === 'agent-dashboard') {
            const ovCalls = document.getElementById('ov-calls');
            const ovTasks = document.getElementById('ov-tasks');
            const ovResp = document.getElementById('ov-resp');
            if (ovCalls) ovCalls.innerText = String(stats.calls);
            if (ovTasks) ovTasks.innerText = String(stats.tasks);
            if (ovResp) ovResp.innerText = stats.calls > 0 ? `${(stats.timeTotal / stats.calls).toFixed(1)}s` : '0.0s';
        }

        // Update visualizer context based on active section
        if (targetId === 'knowledge-graph') {
            triggerVisualizerPipeline('evidence', false);
            if (!cyInitialized) {
                setTimeout(() => {
                    initKnowledgeGraph();
                    cyInitialized = true;
                }, 60);
            } else {
                setTimeout(() => {
                    resizeKnowledgeGraph();
                }, 60);
                const sidebarProj = document.getElementById('sidebar-active-project') as HTMLSelectElement | null;
                const pId = sidebarProj?.value ? Number(sidebarProj.value) : undefined;
                loadGraphData(pId);
                fetchStats();
            }
        }
        else if (targetId === 'nlp-lab') triggerVisualizerPipeline('nlpAnalyze', false);
        else if (targetId === 'knowledge-library') { triggerVisualizerPipeline('docUpload', false); loadKnowledgeLibraryDocs(); }
        else if (targetId === 'knowledge-copilot') triggerVisualizerPipeline('copilot', false);
        else if (targetId === 'evidence-explorer') {
            triggerVisualizerPipeline('evidence', false);
            const evDiv = document.getElementById('evidence-results');
            if (evDiv && !evDiv.children.length) featureEvidence();
        }
        else if (targetId === 'import-web') triggerVisualizerPipeline('webImport', false);
        else if (targetId === 'research-intelligence') {
            triggerVisualizerPipeline('researchCompare', false);
            const intelDiv = document.getElementById('intelligence-results');
            if (intelDiv && !intelDiv.children.length) featureCompareResearch();
        }
        else if (targetId === 'report-studio') triggerVisualizerPipeline('reportGen', false);
        else if (targetId === 'multi-agent-crew') triggerVisualizerPipeline('crewResearch', false);
        else if (targetId === 'about-project') triggerVisualizerPipeline('docUpload', false);

        window.location.hash = targetId;
    }

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = (e.currentTarget as HTMLElement).getAttribute('data-target');
            if (target) activateTab(target);
        });
    });

    // Handle hash on load
    const hash = window.location.hash.substring(1);
    if (hash && document.getElementById(hash)) {
        activateTab(hash);
    } else {
        activateTab('agent-dashboard'); // Default to overview
    }

    // Agent row clicks (overview panel)
    document.querySelectorAll('.agent-row[data-target]').forEach(row => {
        row.addEventListener('click', () => {
            const target = row.getAttribute('data-target');
            if (target) activateTab(target);
        });
    });

    // New Task button -> go to Tasks
    document.getElementById('topbar-new-task')?.addEventListener('click', () => {
        activateTab('agent-tasks');
    });

    // Handle hash changes (e.g. from voice routing)
    window.addEventListener('hashchange', () => {
        const h = window.location.hash.substring(1);
        if (h && document.getElementById(h)) activateTab(h);
    });
});

// --- UI Helpers ---
function setBadge(id: string, status: 'thinking' | 'done' | 'error' | 'idle', text="WORKING") {
    const b = document.getElementById(id);
    if (!b) return;
    if (status === 'idle') { 
        b.style.display = 'none'; 
        return; 
    }
    b.style.display = 'flex';
    b.setAttribute('data-status', status);
    const textSpan = b.querySelector('.badge-text') as HTMLElement;
    if (textSpan) textSpan.innerText = text;
}

// --- Features ---
async function featureUploadDoc() {
    triggerVisualizerPipeline('docUpload', true);
    const fileInput = document.getElementById('document-upload') as HTMLInputElement;
    if (!fileInput.files || fileInput.files.length === 0) {
        appendLog('error', 'Please select a file first.');
        return;
    }
    const file = fileInput.files[0];
    const btn = document.getElementById('btn-upload-doc') as HTMLButtonElement;
    const list = document.getElementById('documents-list');
    const progContainer = document.getElementById('upload-progress-container');
    const progBar = document.getElementById('upload-progress-bar');
    const statusText = document.getElementById('upload-status-text');
    
    btn.disabled = true;
    btn.innerText = 'Processing...';
    setPipelineStage('upload', 'RECEIVING SECURE SOURCE');
    if (progContainer) progContainer.style.display = 'block';
    if (progBar) progBar.style.width = '10%';
    if (statusText) statusText.innerText = 'Checking Workspace...';
    
    try {
        // Fetch projects or default to 1
        let res = await fetch('http://localhost:8000/api/projects');
        let projects = await res.json();
        let projectId = 1;
        
        if (!projects || projects.length === 0) {
            // Create a default project if none exist
            const createRes = await fetch('http://localhost:8000/api/projects', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: 'Default Workspace', description: 'Auto-created'})
            });
            const newProj = await createRes.json();
            projectId = newProj.id;
        } else {
            projectId = projects[0].id;
        }

        if (progBar) progBar.style.width = '30%';
        if (statusText) statusText.innerText = 'Uploading Document...';
        
        const formData = new FormData();
        formData.append("file", file);

        res = await fetch(`http://localhost:8000/api/projects/${projectId}/documents`, {
            method: 'POST',
            body: formData
        });
        
        if (!res.ok) throw new Error(await res.text());
        const docData = await res.json();
        setPipelineStage('parse', 'PARSING DOCUMENT STRUCTURE');
        
        if (progBar) progBar.style.width = '60%';
        if (statusText) statusText.innerText = 'Extracting Intelligence & NLP...';
        setPipelineStage('entities', 'EXTRACTING ENTITIES & LANGUAGE SIGNALS');
        
        const procRes = await fetch(`http://localhost:8000/api/nlp/process_document/${docData.id}`, {
            method: 'POST'
        });
        
        if (!procRes.ok) throw new Error(await procRes.text());
        const procData = await procRes.json();
        setPipelineStage('index', `INDEXED ${procData.chunks_created || 0} SEARCHABLE PASSAGES`, true);
        let intelligence: any = null;
        try { intelligence = await api(`/api/nlp/documents/${docData.id}/intelligence`); } catch (_) { /* count view remains available */ }
        
        if (progBar) progBar.style.width = '100%';
        if (statusText) {
            statusText.innerText = 'Ingestion Complete!';
            statusText.style.color = '#34d399';
        }
        
        if (list) {
            document.getElementById('library-empty')?.remove();
            const el = document.createElement('div');
            el.className = 'content-text';
            el.style.marginTop = '1rem';
            el.style.padding = '1rem';
            el.style.border = '1px solid var(--dash-border)';
            el.style.borderRadius = '8px';
            el.style.background = 'rgba(52, 211, 153, 0.05)';
            el.style.animation = 'hero-fade 0.5s ease';
            el.innerHTML = `
                <div style="color: #34d399; font-weight: 600; margin-bottom: 0.5rem;">✅ Ingestion Complete</div>
                <div><b>File:</b> ${docData.filename}</div>
                <div><b>Entities Found:</b> ${procData.entities_found || 0}</div>
                <div><b>Status:</b> ${procData.status}</div>
            `;
            el.insertAdjacentHTML('beforeend', `<div class="nlp-metrics"><span class="tag">${procData.word_count || 0} words</span><span class="tag">${procData.chunks_created || 0} passages</span><span class="tag">${procData.entities_found || 0} entities</span></div><p class="content-text">Ready for Copilot, Evidence Explorer, and Report Studio.</p>`);
            if (intelligence) {
                const keywordTags = intelligence.keywords.map((item: any) => `<span class="tag">${esc(item.term)}</span>`).join('');
                const entityTags = intelligence.entities.slice(0, 12).map((item: any) => `<span class="tag">${esc(item.text)} · ${esc(item.label)}</span>`).join('');
                el.insertAdjacentHTML('beforeend', `<div class="intelligence-detail"><p><strong>Executive extraction</strong><br>${intelligence.summary.map(esc).join(' ')}</p><p><strong>Keywords</strong><br>${keywordTags || 'No dominant keywords'}</p><p><strong>Named entities</strong><br>${entityTags || 'No entities detected'}</p></div>`);
            }
            list.prepend(el);
        }
        appendLog('success', `Document ${docData.filename} ingested successfully.`);
        
        // Hide progress bar after a short delay
        setTimeout(() => {
            if (progContainer) progContainer.style.display = 'none';
            if (progBar) progBar.style.width = '0%';
            if (statusText) {
                statusText.innerText = 'Preparing...';
                statusText.style.color = 'var(--text-secondary)';
            }
        }, 3000);
        
    } catch(e: any) {
        console.error(e);
        appendLog('error', `Ingestion failed: ${e.message}`);
        setPipelineStage('', 'PROCESSING FAILED · READY TO RETRY');
        
        if (progBar) {
            progBar.style.background = '#ef4444';
            progBar.style.width = '100%';
        }
        if (statusText) {
            statusText.innerText = 'Failed!';
            statusText.style.color = '#ef4444';
        }
        
        if (list) {
            const el = document.createElement('div');
            el.className = 'content-text';
            el.style.marginTop = '1rem';
            el.style.color = '#ef4444';
            el.innerText = `❌ Error: ${e.message}`;
            list.prepend(el);
        }
        
        setTimeout(() => {
            if (progContainer) progContainer.style.display = 'none';
            if (progBar) {
                progBar.style.width = '0%';
                progBar.style.background = 'var(--accent)';
            }
            if (statusText) {
                statusText.innerText = 'Preparing...';
                statusText.style.color = 'var(--text-secondary)';
            }
        }, 4000);
    } finally {
        btn.disabled = false;
        btn.innerText = 'Upload & Ingest';
        fileInput.value = '';
    }
}

function setPipelineStage(stage: string, readout: string, complete = false) {
    const stages = ['upload', 'parse', 'entities', 'index'];
    const stageIndex = stages.indexOf(stage);
    document.querySelectorAll<HTMLElement>('.pipeline-stage').forEach(node => {
        const index = stages.indexOf(node.dataset.stage || '');
        node.classList.toggle('active', node.dataset.stage === stage && !complete);
        node.classList.toggle('complete', (stageIndex >= 0 && index < stageIndex) || (complete && node.dataset.stage === stage));
    });
    const readoutEl = document.getElementById('pipeline-readout');
    if (readoutEl) readoutEl.innerText = readout;
}

async function featureNlpAnalyze() {
    triggerVisualizerPipeline('nlpAnalyze', true);
    const text = (document.getElementById('nlp-text') as HTMLTextAreaElement).value;
    if (!text) return;
    try {
        const res = await fetch('http://localhost:8000/api/nlp/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: text})
        });
        const data = await res.json();
        const resDiv = document.getElementById('nlp-results');
        if (!res.ok) throw new Error(await res.text());
        if (resDiv) renderNlpResults(resDiv, data);
    } catch(e) {
        console.error(e);
        appendLog('error', 'NLP analysis failed');
    }
}

const API = 'http://localhost:8000';
const esc = (value: unknown) => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char] || char));
async function api(path: string, options: RequestInit = {}) {
    const response = await fetch(`${API}${path}`, options);
    if (!response.ok) { const detail = await response.text(); throw new Error(detail || `Request failed (${response.status})`); }
    return response.json();
}
function updateWorkspaceTopbarBadge(projectName?: string) {
    const tag = document.getElementById('topbar-active-proj-tag');
    if (tag) {
        tag.innerText = `Workspace: ${projectName || 'All Projects'}`;
        tag.style.opacity = '1';
        tag.style.color = '#34d399';
    }
}

async function loadKnowledgeLibraryDocs() {
    const list = document.getElementById('documents-list');
    const countBadge = document.getElementById('library-doc-count');
    if (!list) return;

    try {
        const sidebarProj = document.getElementById('sidebar-active-project') as HTMLSelectElement | null;
        const pId = sidebarProj?.value ? `?project_id=${sidebarProj.value}` : '';
        const docs = await api(`/api/documents${pId}`);
        
        if (countBadge) countBadge.innerText = String(docs.length || 0);

        // Also populate #intel-doc-select for Research Intelligence!
        const intelSelect = document.getElementById('intel-doc-select') as HTMLSelectElement | null;
        if (intelSelect) {
            intelSelect.innerHTML = docs.length
                ? docs.map((d: any) => `<option value="${d.id}">${esc(d.title || d.filename)} (ID: ${d.id})</option>`).join('')
                : '<option value="">No indexed documents available</option>';
        }

        if (!docs || docs.length === 0) {
            list.innerHTML = `
                <div class="empty-state-card" style="grid-column: 1 / -1; padding: 3rem 1.5rem; text-align: center; background: rgba(0,0,0,0.02); border: 1px dashed var(--dash-border); border-radius: 12px;">
                    <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📚</div>
                    <h3 style="margin: 0 0 0.5rem 0; font-weight: 700; color:#2F0909;">No Documents Ingested Yet</h3>
                    <p style="color: rgba(47,9,9,0.6); max-width: 450px; margin: 0 auto 1.2rem auto; font-size: 0.88rem;">
                        Upload research papers (PDF, DOCX, TXT) or import web URLs to populate your knowledge library with semantic embeddings and structured intelligence.
                    </p>
                </div>
            `;
            return;
        }

        list.innerHTML = docs.map((d: any) => {
            const format = (d.filename?.split('.').pop() || d.source_type || 'TXT').toUpperCase();
            const words = d.word_count || 0;
            const chunks = d.chunk_count || 0;
            const statusClass = d.processing_status === 'processed' ? 'status-pill-success' : 'status-pill-pending';
            
            return `
                <div class="document-card" data-doc-id="${d.id}">
                    <div class="doc-card-header">
                        <div class="doc-format-badge format-${format.toLowerCase()}">${esc(format)}</div>
                        <span class="status-pill ${statusClass}">${esc(d.processing_status || 'indexed')}</span>
                    </div>
                    <h4 class="doc-title" title="${esc(d.title || d.filename)}">${esc(d.title || d.filename)}</h4>
                    <div class="doc-meta-row">
                        <span>📊 ${words.toLocaleString()} words</span>
                        <span>🧩 ${chunks} passages</span>
                    </div>
                    <div class="doc-actions-row">
                        <button class="btn-ghost" style="padding:0.35rem 0.75rem; font-size:0.75rem;" onclick="inspectDocument(${d.id})" title="Inspect extracted intelligence">🔍 Inspect</button>
                        <button class="btn-ghost" style="padding:0.35rem 0.5rem; font-size:0.75rem; color:#ef4444;" onclick="deleteDocumentItem(${d.id})" title="Delete document">🗑️</button>
                    </div>
                </div>
            `;
        }).join('');

    } catch (err) {
        console.warn('Failed to load documents list:', err);
    }
}

// Global inspect and delete helpers
(window as any).inspectDocument = async (id: number) => {
    try {
        const intel = await api(`/api/nlp/documents/${id}/intelligence`);
        const target = document.getElementById('documents-list');
        if (!target) return;
        const modal = document.createElement('div');
        modal.className = 'research-card';
        modal.style.marginTop = '1rem';
        modal.style.border = '2px solid var(--crimson-core)';
        modal.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0;">📄 Document #${id} Extracted Intelligence</h3>
                <button class="btn-ghost" onclick="this.parentElement.parentElement.remove()" style="font-size:1.1rem; cursor:pointer;">✖</button>
            </div>
            <p><strong>Executive Summary:</strong><br>${(intel.summary || []).map(esc).join(' ')}</p>
            <p><strong>Dominant Keywords:</strong><br>${(intel.keywords || []).map((k: any) => `<span class="tag">${esc(k.term)}</span>`).join(' ') || 'None'}</p>
            <p><strong>Entities:</strong><br>${(intel.entities || []).slice(0, 15).map((e: any) => `<span class="tag">${esc(e.text)} (${esc(e.label)})</span>`).join(' ') || 'None'}</p>
        `;
        target.prepend(modal);
        modal.scrollIntoView({ behavior: 'smooth' });
    } catch (err: any) {
        alert(`Failed to inspect document #${id}: ${err.message}`);
    }
};

(window as any).deleteDocumentItem = async (id: number) => {
    if (!confirm(`Are you sure you want to delete document #${id}?`)) return;
    try {
        await api(`/api/documents/${id}`, { method: 'DELETE' });
        await loadKnowledgeLibraryDocs();
        fetchStats();
    } catch (err: any) {
        alert(`Failed to delete document: ${err.message}`);
    }
};

async function refreshProjects() {
    try {
        const projects = await api('/api/projects');
        
        // Populate sidebar active project selector
        const sidebarProj = document.getElementById('sidebar-active-project') as HTMLSelectElement | null;
        if (sidebarProj) {
            const currVal = sidebarProj.value;
            sidebarProj.innerHTML = `<option value="">All Projects Workspace</option>` + projects.map((p: any) => `<option value="${p.id}">${esc(p.name)}</option>`).join('');
            if (currVal) sidebarProj.value = currVal;
            
            const selectedText = sidebarProj.value && sidebarProj.selectedIndex > 0 ? sidebarProj.options[sidebarProj.selectedIndex].text : 'All Projects';
            updateWorkspaceTopbarBadge(selectedText);

            sidebarProj.onchange = () => {
                const pId = sidebarProj.value ? Number(sidebarProj.value) : undefined;
                const name = sidebarProj.value && sidebarProj.selectedIndex > 0 ? sidebarProj.options[sidebarProj.selectedIndex].text : 'All Projects';
                updateWorkspaceTopbarBadge(name);
                loadGraphData(pId);
                fetchStats();
                loadKnowledgeLibraryDocs();
            };
        } else {
            updateWorkspaceTopbarBadge('Active');
        }

        const metrics = document.getElementById('evaluation-results');
        if (metrics) metrics.innerHTML = `<div class="research-card"><h3>Workspace metrics</h3><p>${projects.length} active research workspace${projects.length === 1 ? '' : 's'}. Processed documents expose entities, chunks, retrieval, and evidence checks.</p></div>`;

        // Load documents into library and dropdowns
        await loadKnowledgeLibraryDocs();
    } catch {
        updateWorkspaceTopbarBadge('Active');
    }
}

function renderNlpResults(target: HTMLElement, data: any) {
    const keywords = (data.keywords || []).map((item: any) => `<span class="tag">${esc(item.term)} · ${item.count}</span>`).join('') || '<small>No keywords found</small>';
    const entities = (data.entities || []).slice(0, 18).map((item: any) => `<span class="tag">${esc(item.text)} · ${esc(item.label)}</span>`).join('') || '<small>No named entities found</small>';
    target.innerHTML = `<div class="research-card"><h3>Text intelligence</h3><p>${data.preprocessing?.word_count || 0} words · ${data.preprocessing?.sentence_count || 0} sentences</p><p><strong>Summary</strong><br>${(data.summary || []).map(esc).join(' ') || 'No summary available.'}</p><p><strong>Keywords</strong><br>${keywords}</p><p><strong>Entities</strong><br>${entities}</p></div>`;
}

function isProbablyUrl(text: string): boolean {
    const trimmed = text.trim();
    return /^https?:\/\//i.test(trimmed) || /^www\./i.test(trimmed) || /^[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(\/.*)?$/i.test(trimmed);
}

async function featureWebSearch(searchQuery?: string) {
    const input = document.getElementById('import-url') as HTMLInputElement | null;
    const query = (searchQuery || input?.value || '').trim();
    const searchTarget = document.getElementById('web-search-results');
    if (!query || !searchTarget) return;

    searchTarget.style.display = 'block';
    searchTarget.innerHTML = `
        <div class="research-card">
            <p>🔍 Searching open research knowledge & science preprints for "<strong>${esc(query)}</strong>"...</p>
        </div>
    `;

    try {
        const resp = await api(`/api/web/search?q=${encodeURIComponent(query)}&limit=6`);
        const results = resp.results || [];
        if (!results.length) {
            searchTarget.innerHTML = `
                <div class="research-card">
                    <h3>No Web Articles Found</h3>
                    <p>No direct matches found for "${esc(query)}". You can paste an exact article URL above to fetch and analyze directly.</p>
                </div>
            `;
            return;
        }

        searchTarget.innerHTML = `
            <div class="research-card" style="margin-bottom:0.75rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                    <h3 style="margin:0; font-size:1rem; color:#2F0909;">
                        Web Search Results (${results.length}) for "${esc(query)}"
                    </h3>
                    <button class="btn-secondary btn-sm" id="btn-close-web-search" style="padding:2px 8px; font-size:0.75rem;">Close</button>
                </div>
                <p style="margin:0.25rem 0 0.75rem 0; font-size:0.8rem; color:rgba(47,9,9,0.7);">
                    Click "Import & Analyze" on any paper or article below to ingest full content, extract summary, key news, and relevant visuals into your active workspace.
                </p>
                <div class="web-search-results-grid">
                    ${results.map((item: any) => `
                        <div class="web-search-item">
                            ${item.imageUrl ? `<img src="${esc(item.imageUrl)}" class="web-search-thumb" alt="${esc(item.title)}" onerror="this.style.display='none'">` : ''}
                            <div class="web-search-info">
                                <h4 class="web-search-title">${esc(item.title)}</h4>
                                <div style="display:flex; gap:0.5rem; align-items:center; margin-bottom:0.25rem;">
                                    <span class="web-search-badge">${esc(item.source || 'Open Web')}</span>
                                    <span style="font-size:0.72rem; color:rgba(47,9,9,0.5);">${esc(item.type === 'journal_paper' ? 'Research Paper' : 'Knowledge Base')}</span>
                                </div>
                                <p class="web-search-snippet">${esc(item.snippet || '')}</p>
                            </div>
                            <button class="btn-primary btn-sm btn-search-import" data-url="${esc(item.url)}" data-title="${esc(item.title)}" style="flex-shrink:0; padding:6px 14px; font-size:0.8rem;">
                                Import & Analyze
                            </button>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        document.getElementById('btn-close-web-search')?.addEventListener('click', () => {
            searchTarget.style.display = 'none';
        });

        document.querySelectorAll('.btn-search-import').forEach((btn) => {
            btn.addEventListener('click', (e) => {
                const targetBtn = e.currentTarget as HTMLElement;
                const importUrl = targetBtn.getAttribute('data-url');
                if (importUrl) {
                    if (input) input.value = importUrl;
                    featureImportUrl(importUrl);
                    searchTarget.style.display = 'none';
                }
            });
        });
    } catch (err: any) {
        searchTarget.innerHTML = `<div class="research-card status-error"><p>${esc(err.message)}</p></div>`;
    }
}

async function featureImportUrl(targetUrl?: string) {
    triggerVisualizerPipeline('webImport', true);
    const input = document.getElementById('import-url') as HTMLInputElement | null;
    let url = (targetUrl || input?.value || '').trim();
    const output = document.getElementById('import-results');
    if (!url || !output) return;

    // Smart detection: If user typed a search term instead of a URL, route to web search!
    if (!isProbablyUrl(url)) {
        await featureWebSearch(url);
        return;
    }

    // Auto-prefix https if missing scheme
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
        url = 'https://' + url;
    }

    try {
        const projects = await api('/api/projects');
        const projId = projects.length ? projects[0].id : 1;
        output.innerHTML = `
            <div class="research-card">
                <p>🌐 Fetching web document, extracting high-resolution images, full content, and synthesizing key intelligence...</p>
            </div>
        `;

        const doc = await api(`/api/projects/${projId}/import-url`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({url})
        });

        const processed = await api(`/api/nlp/process_document/${doc.id}`, {method: 'POST'});
        const words = doc.word_count || processed.word_count || 0;
        const passages = processed.chunks_created || processed.chunks_indexed || 1;
        const readingTime = doc.reading_time_min || Math.max(1, Math.round(words / 200));

        // Format summary
        const summaryText = typeof doc.summary === 'string' 
            ? doc.summary 
            : (Array.isArray(doc.summary) ? doc.summary.join(' ') : (doc.content_preview || ''));

        // Format key news
        const keyNewsList: string[] = Array.isArray(doc.key_news) && doc.key_news.length > 0 
            ? doc.key_news 
            : (Array.isArray(doc.summary) && doc.summary.length > 1 ? doc.summary : ['Key findings extracted and indexed into knowledge base.']);

        const keyNewsHtml = keyNewsList.map((item: string, idx: number) => `
            <div class="key-news-item">
                <span class="key-news-badge">KEY POINT #${idx + 1}</span>
                <p class="key-news-text">${esc(item)}</p>
            </div>
        `).join('');

        // Format images
        const images: any[] = Array.isArray(doc.images) ? doc.images : [];
        const imagesHtml = images.map((img: any, idx: number) => {
            const imgUrl = typeof img === 'string' ? img : img.url;
            const imgAlt = typeof img === 'string' ? `Visual #${idx + 1}` : (img.alt || `Visual #${idx + 1}`);
            const isCover = typeof img === 'object' && img.is_cover;
            return `
                <div class="article-image-card">
                    <div class="article-image-wrap">
                        ${isCover ? '<span class="cover-badge">Primary Asset</span>' : ''}
                        <img src="${esc(imgUrl)}" alt="${esc(imgAlt)}" loading="lazy" onerror="this.closest('.article-image-card').style.display='none'">
                    </div>
                    <div class="article-image-caption">
                        ${esc(imgAlt)}
                    </div>
                </div>
            `;
        }).join('');

        // Format full readable content
        const fullContent = doc.full_content || doc.content_text || doc.content_preview || '';
        const paragraphs = fullContent.split(/\n\s*\n/).filter((p: string) => p.trim().length > 0);
        const formattedParagraphsHtml = paragraphs.length > 0 
            ? paragraphs.map((p: string) => `<p class="full-content-paragraph">${esc(p.trim())}</p>`).join('')
            : `<p class="full-content-paragraph">${esc(fullContent)}</p>`;

        const engineLabel = doc.structured_by === 'gemini' ? 'AI Structured' : 'Local NLP Backend';

        output.innerHTML = `
            <div class="web-research-result-container">
                <!-- Header Ingestion Badge -->
                <div class="research-card ingestion-header-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.75rem;">
                        <div style="display:flex; gap:0.4rem; align-items:center; flex-wrap:wrap;">
                            <span class="badge-tag status-success">Source Ingested & Vector Indexed</span>
                            <span class="badge-tag domain-tag">${esc(doc.domain || 'Web Document')}</span>
                        </div>
                        <a href="${esc(doc.url)}" target="_blank" rel="noopener noreferrer" class="source-link-btn">
                            Visit Original Source ↗
                        </a>
                    </div>
                    <h2 class="article-title">${esc(doc.title)}</h2>
                    <div class="article-meta-row">
                        <span><strong>${words.toLocaleString()}</strong> words</span>
                        <span>·</span>
                        <span><strong>${readingTime} min</strong> read</span>
                        <span>·</span>
                        <span><strong>${passages}</strong> dense vector passages</span>
                        <span>·</span>
                        <span class="tech-pill">${engineLabel}</span>
                    </div>
                </div>

                <!-- Executive Summary -->
                <div class="research-card summary-card">
                    <div class="card-subtitle-row">
                        <span class="section-indicator-dot"></span>
                        <h3 style="margin:0; font-size:1rem; font-weight:700; color:#2F0909;">Executive Summary</h3>
                    </div>
                    <p class="summary-paragraph">
                        ${esc(summaryText || 'Executive summary synthesized from document content.')}
                    </p>
                </div>

                <!-- Key News & Highlights -->
                <div class="research-card key-news-card">
                    <div class="card-subtitle-row">
                        <span class="section-indicator-dot" style="background:#e11d48; box-shadow:0 0 8px rgba(225,29,72,0.5);"></span>
                        <h3 style="margin:0; font-size:1rem; font-weight:700; color:#2F0909;">Key News & Core Takeaways</h3>
                    </div>
                    <div class="key-news-grid">
                        ${keyNewsHtml}
                    </div>
                </div>

                <!-- Relevant Images Gallery -->
                ${images.length > 0 ? `
                <div class="research-card images-gallery-card">
                    <div class="card-subtitle-row">
                        <span class="section-indicator-dot" style="background:#8b5cf6; box-shadow:0 0 8px rgba(139,92,246,0.5);"></span>
                        <h3 style="margin:0; font-size:1rem; font-weight:700; color:#2F0909;">Relevant Article Images (${images.length})</h3>
                    </div>
                    <div class="images-responsive-grid">
                        ${imagesHtml}
                    </div>
                </div>
                ` : ''}

                <!-- Full Readable Content Viewer -->
                <div class="research-card full-content-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                        <div class="card-subtitle-row" style="margin-bottom:0;">
                            <span class="section-indicator-dot" style="background:#0ea5e9; box-shadow:0 0 8px rgba(14,165,233,0.5);"></span>
                            <h3 style="margin:0; font-size:1rem; font-weight:700; color:#2F0909;">Full Readable Content (${words.toLocaleString()} words)</h3>
                        </div>
                        <button class="btn-secondary btn-sm" id="btn-toggle-full-content" style="padding:4px 12px; font-size:0.78rem;">
                            Expand Full Content
                        </button>
                    </div>
                    <div id="full-content-body" class="full-content-collapsible">
                        ${formattedParagraphsHtml}
                    </div>
                </div>
            </div>
        `;

        // Bind toggle for full content
        const toggleBtn = document.getElementById('btn-toggle-full-content');
        const contentBody = document.getElementById('full-content-body');
        if (toggleBtn && contentBody) {
            toggleBtn.addEventListener('click', () => {
                const isExpanded = contentBody.classList.toggle('expanded');
                toggleBtn.innerText = isExpanded ? 'Collapse Full Content' : 'Expand Full Content';
            });
        }

        await loadKnowledgeLibraryDocs();
        fetchStats();
    } catch (error: any) {
        output.innerHTML = `<div class="research-card status-error"><p>${esc(error.message)}</p></div>`;
    }
}

async function featureCopilot() {
    triggerVisualizerPipeline('copilot', true);
    const query = (document.getElementById('copilot-query') as HTMLInputElement)?.value.trim();
    const target = document.getElementById('copilot-answer');
    if (!target) return;
    if (!query) {
        target.innerHTML = '<div class="research-card"><p>Please enter a research question or click a suggested prompt above.</p></div>';
        return;
    }
    
    target.innerHTML = '<div class="research-card"><p>⚡ Computing dense 384-D vector query embedding and retrieving passages...</p></div>';
    
    try {
        const sidebarProj = document.getElementById('sidebar-active-project') as HTMLSelectElement | null;
        const project = sidebarProj?.value ? Number(sidebarProj.value) : null;
        
        const data = await api('/api/copilot/ask', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query, project_id: project, top_k: 4})
        });
        
        const sources = (data.sources && data.sources.length) ? data.sources : ((data.citations || []).map((c: any) => ({
            document: c.doc_title || 'Document',
            score: c.similarity || c.score || 0.85,
            excerpt: c.preview || c.excerpt || ''
        })));
        
        target.innerHTML = `
            <div class="rag-answer-container">
                <div class="rag-answer-header">
                    <div>
                        <span style="font-size:0.72rem; font-weight:800; background:var(--crimson-core); color:#fff; padding:0.2rem 0.5rem; border-radius:10px;">SOURCE-GROUNDED RAG</span>
                        <strong style="margin-left:0.5rem; color:#2F0909;">Answer Synthesis</strong>
                    </div>
                    <span style="font-size:0.75rem; color:rgba(47,9,9,0.6); font-family:var(--font-mono);">${data.latency_ms || 12}ms latency · Dense 384-D Search</span>
                </div>
                <div class="rag-answer-text">${esc(data.answer)}</div>
                
                <h4 style="margin:1rem 0 0.5rem 0; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em; color:rgba(47,9,9,0.7);">
                    📚 Exact Source Citations (${sources.length}):
                </h4>
                <div class="rag-sources-grid">
                    ${sources.map((s: any) => `
                        <div class="rag-source-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                                <div class="rag-source-title">${esc(s.document)}</div>
                                <span class="tag" style="font-size:0.65rem;">score: ${s.score}</span>
                            </div>
                            <div class="rag-source-excerpt">"${esc(s.excerpt)}"</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    } catch (error: any) {
        target.innerHTML = `<div class="research-card status-error"><p>${esc(error.message)}</p></div>`;
    }
}

async function featureCopilotSop() {
    const target = document.getElementById('copilot-answer');
    if (!target) return;
    target.innerHTML = '<div class="research-card"><p>⏳ Extracting operational Standard Operating Procedures (SOP) across indexed documents...</p></div>';
    try {
        const sidebarProj = document.getElementById('sidebar-active-project') as HTMLSelectElement | null;
        const project = sidebarProj?.value ? Number(sidebarProj.value) : 1;
        const rawChecklist = await api(`/api/copilot/extract_procedures?project_id=${project}`, {method: 'POST'});
        const checklist = Array.isArray(rawChecklist) 
            ? rawChecklist 
            : (rawChecklist.execution_steps || rawChecklist.steps || []).map((s: any, idx: number) => 
                typeof s === 'string' ? {step: idx + 1, action: s, details: ''} : s
            );
        
        target.innerHTML = `
            <div class="research-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; border-bottom:1px solid var(--dash-border); padding-bottom:0.75rem;">
                    <h3 style="margin:0; font-weight:800; color:var(--crimson-core);">📋 Extracted Standard Operating Procedures</h3>
                    <span class="tag">${checklist.length} Procedural Steps</span>
                </div>
                <div style="display:flex; flex-direction:column; gap:0.75rem;">
                    ${checklist.map((item: any) => `
                        <div style="background:rgba(47,9,9,0.02); padding:0.75rem 1rem; border-radius:8px; border:1px solid var(--dash-border);">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <strong style="color:var(--text-main);">Step ${item.step}: ${esc(item.action)}</strong>
                                <span class="status-pill status-pill-success">Mandatory</span>
                            </div>
                            <p style="margin:0.35rem 0 0 0; font-size:0.8rem; color:rgba(47,9,9,0.7);">${esc(item.details || '')}</p>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    } catch (err: any) {
        target.innerHTML = `<div class="research-card status-error"><p>Failed to extract SOP: ${esc(err.message)}</p></div>`;
    }
}

async function featureEvidence() {
    triggerVisualizerPipeline('evidence', true);
    const claim = (document.getElementById('evidence-claim') as HTMLInputElement)?.value.trim() || '';
    const target = document.getElementById('evidence-results');
    if (!target) return;
    
    target.innerHTML = '<div class="research-card"><p>⚖️ Evaluating empirical claims and computing cross-document stance metrics...</p></div>';
    
    try {
        const sidebarProj = document.getElementById('sidebar-active-project') as HTMLSelectElement | null;
        const project = sidebarProj?.value ? Number(sidebarProj.value) : 1;
        
        const data = await api('/api/research/evidence', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({claim: claim || null, query: claim || null, project_id: project})
        });
        
        if (data.mode === 'project_audit') {
            const list = data.evidence || data.evidence_matrix || [];
            target.innerHTML = `
                <div class="research-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                        <h3 style="margin:0; font-weight:800;">Cross-Document Empirical Stance Matrix</h3>
                        <span class="tag">${list.length} Claims Audited</span>
                    </div>
                    <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:1rem;">
                        ${list.map((item: any) => {
                            const stanceLower = (item.stance || 'neutral').toLowerCase();
                            const pillClass = stanceLower.includes('support') ? 'stance-pill-supports' : (stanceLower.includes('contra') ? 'stance-pill-contradicts' : 'stance-pill-neutral');
                            return `
                                <div style="background:#fff; border:1px solid var(--dash-border); border-radius:10px; padding:1rem; display:flex; flex-direction:column; gap:0.5rem;">
                                    <div style="display:flex; justify-content:space-between; align-items:center;">
                                        <span class="stance-pill ${pillClass}">${esc(item.stance || 'Neutral')}</span>
                                        <span class="tag" style="font-size:0.65rem;">conf: ${item.confidence || 0.75}</span>
                                    </div>
                                    <strong style="font-size:0.88rem; color:#2F0909;">"${esc(item.claim || item.claim_text)}"</strong>
                                    <small style="color:var(--crimson-core); font-weight:600;">Source: ${esc(item.source_doc || item.document || 'Document')}</small>
                                    <p style="margin:0; font-size:0.8rem; color:rgba(47,9,9,0.7);">${esc(item.evidence_snippet || item.excerpt || item.explanation || '')}</p>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        } else {
            const list = data.evidence || [];
            target.innerHTML = list.length ? `
                <div class="research-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                        <h3 style="margin:0; font-weight:800;">Claim Verification Results for: "${esc(data.claim)}"</h3>
                        <span class="tag">${list.length} Corroborating Passages</span>
                    </div>
                    <div style="display:flex; flex-direction:column; gap:0.85rem;">
                        ${list.map((e: any) => {
                            const stanceLower = (e.stance || 'neutral').toLowerCase();
                            const pillClass = stanceLower.includes('support') ? 'stance-pill-supports' : (stanceLower.includes('contra') ? 'stance-pill-contradicts' : 'stance-pill-neutral');
                            return `
                                <div style="background:rgba(47,9,9,0.02); border:1px solid var(--dash-border); border-radius:10px; padding:1rem;">
                                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                                        <div style="display:flex; align-items:center; gap:0.5rem;">
                                            <span class="stance-pill ${pillClass}">${esc(e.stance)}</span>
                                            <strong style="color:var(--crimson-core); font-size:0.85rem;">${esc(e.document)}</strong>
                                        </div>
                                        <span class="tag" style="font-size:0.7rem;">similarity: ${e.score}</span>
                                    </div>
                                    <p style="margin:0 0 0.5rem 0; font-size:0.85rem; color:#2F0909; line-height:1.5;">"${esc(e.excerpt)}"</p>
                                    ${e.explanation ? `<small style="color:rgba(47,9,9,0.6);">${esc(e.explanation)}</small>` : ''}
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            ` : '<div class="research-card"><p>No corroborating or conflicting evidence passages were found for this claim.</p></div>';
        }
    } catch (error: any) {
        target.innerHTML = `<div class="research-card status-error"><p>${esc(error.message)}</p></div>`;
    }
}

async function featureExtractPaperStruct() {
    const select = document.getElementById('intel-doc-select') as HTMLSelectElement | null;
    const target = document.getElementById('intelligence-results');
    if (!select || !target) return;
    const docId = select.value;
    if (!docId) {
        target.innerHTML = '<div class="research-card"><p>Please select an indexed document from the dropdown above.</p></div>';
        return;
    }
    target.innerHTML = '<div class="research-card"><p>⏳ Extracting scientific structure (methodology, datasets, metrics, findings)...</p></div>';
    try {
        const res = await api(`/api/research/extract_paper/${docId}`, { method: 'POST' });
        target.innerHTML = `
            <div class="research-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; border-bottom:1px solid var(--dash-border); padding-bottom:0.75rem;">
                    <h3 style="margin:0; font-weight:800; color:var(--crimson-core);">📄 ${esc(select.options[select.selectedIndex]?.text)}</h3>
                    <span class="tag">Year: ${esc(res.publication_year || 'N/A')}</span>
                </div>
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:1rem;">
                    <div style="background:rgba(47,9,9,0.02); padding:0.9rem; border-radius:8px; border:1px solid var(--dash-border);">
                        <strong style="color:var(--text-main);">🔬 Methodology:</strong>
                        <p style="margin:0.4rem 0 0 0; font-size:0.85rem; color:rgba(47,9,9,0.75);">${esc(res.methodology || 'Extracted empirical architecture and signal processing steps.')}</p>
                    </div>
                    <div style="background:rgba(47,9,9,0.02); padding:0.9rem; border-radius:8px; border:1px solid var(--dash-border);">
                        <strong style="color:var(--text-main);">📦 Datasets Used:</strong>
                        <p style="margin:0.4rem 0 0 0; font-size:0.85rem; color:rgba(47,9,9,0.75);">${esc(res.datasets || 'Benchmark training corpora.')}</p>
                    </div>
                    <div style="background:rgba(47,9,9,0.02); padding:0.9rem; border-radius:8px; border:1px solid var(--dash-border);">
                        <strong style="color:var(--text-main);">📈 Reported Metrics:</strong>
                        <p style="margin:0.4rem 0 0 0; font-size:0.85rem; color:rgba(47,9,9,0.75);">${esc(res.metrics || 'Accuracy, ROC-AUC, and latency values.')}</p>
                    </div>
                    <div style="background:rgba(47,9,9,0.02); padding:0.9rem; border-radius:8px; border:1px solid var(--dash-border);">
                        <strong style="color:var(--text-main);">💡 Key Findings:</strong>
                        <p style="margin:0.4rem 0 0 0; font-size:0.85rem; color:rgba(47,9,9,0.75);">${esc(res.findings || 'Primary conclusions and breakthroughs.')}</p>
                    </div>
                    <div style="background:rgba(47,9,9,0.02); padding:0.9rem; border-radius:8px; border:1px solid var(--dash-border);">
                        <strong style="color:var(--text-main);">⚠️ Limitations & Gaps:</strong>
                        <p style="margin:0.4rem 0 0 0; font-size:0.85rem; color:rgba(47,9,9,0.75);">${esc(res.limitations || 'Evaluation constraints.')}</p>
                    </div>
                    <div style="background:rgba(47,9,9,0.02); padding:0.9rem; border-radius:8px; border:1px solid var(--dash-border);">
                        <strong style="color:var(--text-main);">🚀 Future Work:</strong>
                        <p style="margin:0.4rem 0 0 0; font-size:0.85rem; color:rgba(47,9,9,0.75);">${esc(res.future_work || 'Recommended extensions.')}</p>
                    </div>
                </div>
            </div>
        `;
    } catch (err: any) {
        target.innerHTML = `<div class="research-card status-error"><p>Failed to extract paper structure: ${esc(err.message)}</p></div>`;
    }
}

async function featureCompareResearch() {
    triggerVisualizerPipeline('researchCompare', true);
    const target = document.getElementById('intelligence-results');
    if (!target) return;
    target.innerHTML = '<div class="research-card"><p>⏳ Extracting multi-paper structures and computing comparative synthesis matrix…</p></div>';
    try {
        const sidebarProj = document.getElementById('sidebar-active-project') as HTMLSelectElement | null;
        const project = sidebarProj?.value ? `?project_id=${sidebarProj.value}` : '';
        const data = await api(`/api/research/comparison${project}`);
        
        target.innerHTML = data.comparisons && data.comparisons.length ? `
            <div class="research-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; border-bottom:1px solid var(--dash-border); padding-bottom:0.75rem;">
                    <h3 style="margin:0; font-weight:800; color:#2F0909;">Cross-Paper Comparative Synthesis Matrix</h3>
                    <span class="tag">${data.comparisons.length} Papers Compared</span>
                </div>
                ${data.synthesis ? `
                    <div style="background:linear-gradient(135deg, rgba(192,21,42,0.05), rgba(59,130,246,0.05)); border:1px solid var(--dash-border); border-radius:10px; padding:1.25rem; margin-bottom:1.5rem;">
                        <h4 style="margin:0 0 0.5rem 0; font-weight:700; color:var(--crimson-core);">Executive Synthesis</h4>
                        <p style="margin:0; font-size:0.9rem; line-height:1.6; color:#2F0909;">${esc(data.synthesis)}</p>
                    </div>
                ` : ''}
                <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:1rem;">
                    ${data.comparisons.map((i: any) => `
                        <div style="background:#fff; border:1px solid var(--dash-border); border-radius:10px; padding:1rem; display:flex; flex-direction:column; gap:0.5rem;">
                            <h4 style="margin:0; font-weight:700; color:var(--crimson-core);">${esc(i.document)}</h4>
                            <p style="margin:0; font-size:0.8rem; color:rgba(47,9,9,0.7);"><strong style="color:#2F0909;">Method:</strong> ${esc(i.methodology)}</p>
                            <p style="margin:0; font-size:0.8rem; color:rgba(47,9,9,0.7);"><strong style="color:#2F0909;">Findings:</strong> ${esc(i.findings)}</p>
                            <p style="margin:0; font-size:0.8rem; color:rgba(47,9,9,0.7);"><strong style="color:#2F0909;">Limitations:</strong> ${esc(i.limitations)}</p>
                        </div>
                    `).join('')}
                </div>
            </div>
        ` : '<div class="research-card"><p>No processed research papers are available yet. Ingest documents in Knowledge Library first.</p></div>';
    } catch (error: any) {
        target.innerHTML = `<div class="research-card status-error"><p>${esc(error.message)}</p></div>`;
    }
}

async function featureReport() {
    triggerVisualizerPipeline('reportGen', true);
    const typeSelect = document.getElementById('report-type-select') as HTMLSelectElement | null;
    const titleInput = document.getElementById('report-custom-title') as HTMLInputElement | null;
    const editorCard = document.getElementById('report-editor-card');
    const editorArea = document.getElementById('report-markdown-editor') as HTMLTextAreaElement | null;
    const viewTitle = document.getElementById('report-view-title');
    
    const reportType = typeSelect?.value || 'literature_review';
    const customTitle = titleInput?.value.trim() || undefined;
    
    const sidebarProj = document.getElementById('sidebar-active-project') as HTMLSelectElement | null;
    const projectId = sidebarProj?.value ? Number(sidebarProj.value) : 1;
    
    if (editorCard) editorCard.style.display = 'block';
    if (viewTitle) viewTitle.innerText = 'Generating Draft...';
    if (editorArea) editorArea.value = '⏳ Compiling comprehensive structured report from indexed knowledge and empirical benchmarks...';
    
    try {
        const data = await api('/api/reports/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                project_id: projectId,
                report_type: reportType,
                title: customTitle
            })
        });
        
        if (viewTitle) viewTitle.innerText = data.title;
        if (editorArea) editorArea.value = data.content || data.markdown || '';
        editorCard?.scrollIntoView({ behavior: 'smooth' });
    } catch (error: any) {
        if (editorArea) editorArea.value = `Error generating report: ${error.message}`;
    }
}

function copyReportMarkdown() {
    const editorArea = document.getElementById('report-markdown-editor') as HTMLTextAreaElement | null;
    const copyBtn = document.getElementById('btn-copy-report');
    if (!editorArea) return;
    navigator.clipboard.writeText(editorArea.value);
    if (copyBtn) {
        const orig = copyBtn.innerText;
        copyBtn.innerText = '✅ Copied!';
        setTimeout(() => { copyBtn.innerText = orig; }, 2000);
    }
}

function downloadReportMarkdown() {
    const editorArea = document.getElementById('report-markdown-editor') as HTMLTextAreaElement | null;
    const viewTitle = document.getElementById('report-view-title');
    if (!editorArea || !editorArea.value) return;
    const title = (viewTitle?.innerText || 'research_report').replace(/[^a-z0-9]/gi, '_').toLowerCase();
    const blob = new Blob([editorArea.value], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${title}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

async function featureBrowse() {
    stats.tasks++;
    const url = (document.getElementById('browse-url') as HTMLInputElement).value;
    const task = (document.getElementById('browse-task') as HTMLInputElement).value;
    void task;
    
    setBadge('badge-browse', 'thinking');
    const barUi = document.getElementById('browse-bar-ui');
    const contentBox = document.getElementById('browse-content');
    const factsBox = document.getElementById('browse-facts');
    const logDiv = document.getElementById('browse-log');
    const progBar = document.getElementById('browse-progress');
    
    if (barUi) barUi.innerText = `🔒 ${url}`;
    if (contentBox) contentBox.innerText = '';
    if (factsBox) factsBox.innerHTML = '';
    if (logDiv) logDiv.innerHTML = '';
    if (progBar) progBar.style.width = '0%';
    
    const steps = [
        "Fetching page content via backend scraper...",
        "Parsing HTML with BeautifulSoup...",
        "Cross-referencing content with task parameters...",
        "Summarizing findings locally..."
    ];

    for(let i=0; i<steps.length; i++) {
        const div = document.createElement('div');
        div.className = 'reasoning-line';
        div.innerText = steps[i];
        logDiv?.appendChild(div);
        if (progBar) progBar.style.width = `${(i+1)*25}%`;
        await new Promise(r => setTimeout(r, 600));
    }

    let backendData: any = null;
    try {
        const res = await fetch('http://localhost:8000/api/scrape', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });
        if (!res.ok) throw new Error("HTTP error");
        backendData = await res.json();
    } catch(e) {
        appendLog('error', 'Backend scrape failed, using fallback knowledge.');
    }

    try {
        let data;
        if (backendData) {
            data = {
                extracted_content: backendData.full_text.substring(0, 500) + '...',
                key_facts: backendData.summary,
                summary: backendData.summary.join(' ')
            };
        } else {
            throw new Error("No data returned from scraper");
        }

        if(data && contentBox && factsBox) {
            await typeText(contentBox, data.extracted_content || "No content extracted.", 10);
            (data.key_facts || []).forEach((f: string) => {
                const li = document.createElement('li'); li.innerText = f;
                factsBox.appendChild(li);
            });
            setBadge('badge-browse', 'done', 'DONE');
        } else {
            throw new Error("Invalid response format from backend.");
        }
    } catch(e: any) {
        if (logDiv) {
            const div = document.createElement('div');
            div.className = 'reasoning-line';
            div.style.color = 'var(--crimson-glow)';
            div.innerText = `Failed: ${e.message || "Unknown error"}`;
            logDiv.appendChild(div);
        }
        setBadge('badge-browse', 'error', 'ERROR');
    }
}

let lastScrapedData: any = null;
async function featureScrape() {
    stats.tasks++;
    const url = (document.getElementById('scrape-url') as HTMLInputElement).value;
    setBadge('badge-scrape', 'thinking');
    
    const rawBox = document.getElementById('scrape-raw');
    const sumBox = document.getElementById('scrape-summary');
    const csvBtn = document.getElementById('scrape-csv-btn');
    
    if (rawBox) rawBox.innerText = 'Scraping and analyzing...';
    if (sumBox) sumBox.innerHTML = '';
    if (csvBtn) csvBtn.style.display = 'none';
    lastScrapedData = null;

    try {
        const res = await fetch('http://localhost:8000/api/scrape', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });
        
        if(!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        lastScrapedData = data;
        
        if (rawBox) rawBox.innerText = data.full_text;
        
        let highlightedHtml = '';
        data.summary.forEach((sentence: string) => {
            highlightedHtml += `<span style="background:rgba(212,160,23,0.15); color:var(--gold-light); padding:2px 4px; border-radius:3px; margin-bottom:10px; display:inline-block;">${sentence}</span><br>`;
        });
        
        // Add Images
        if (data.images && data.images.length > 0) {
            highlightedHtml += `<h4 style="color:var(--gold-pure); margin-top:15px; margin-bottom:5px;">Relevant Images</h4><div style="display:flex; gap:10px; overflow-x:auto; padding-bottom:10px;">`;
            data.images.forEach((imgUrl: string) => {
                highlightedHtml += `<img src="${imgUrl}" style="height:100px; border-radius:5px; border:1px solid rgba(255,255,255,0.1); object-fit:cover;" onerror="this.style.display='none'" />`;
            });
            highlightedHtml += `</div>`;
        }
        
        // Add URLs
        if (data.urls && data.urls.length > 0) {
            highlightedHtml += `<h4 style="color:var(--gold-pure); margin-top:15px; margin-bottom:5px;">Relevant Links</h4><ul style="padding-left:20px; font-size:0.85rem; max-height:200px; overflow-y:auto;">`;
            data.urls.forEach((u: any) => {
                highlightedHtml += `<li style="margin-bottom:5px;"><a href="${u.url}" target="_blank" style="color:#60a5fa; text-decoration:none;">${u.text}</a></li>`;
            });
            highlightedHtml += `</ul>`;
        }

        if (sumBox) sumBox.innerHTML = highlightedHtml;
        if (csvBtn) csvBtn.style.display = 'inline-block';
        
        setBadge('badge-scrape', 'done', 'DONE');
    } catch(e: any) {
        appendLog('error', `Scrape failed: ${e.message}`);
        if (rawBox) rawBox.innerText = `Error: Please ensure the Python backend is running on localhost:8000. \nDetails: ${e.message}`;
        setBadge('badge-scrape', 'error', 'ERROR');
    }
}

function downloadScrapeCSV() {
    if(!lastScrapedData) return;
    let csvContent = "data:text/csv;charset=utf-8,Type,Content\n";
    csvContent += `Title,"${lastScrapedData.title.replace(/"/g, '""')}"\n`;
    csvContent += `URL,"${lastScrapedData.url.replace(/"/g, '""')}"\n`;
    lastScrapedData.summary.forEach((s: string, i: number) => {
        csvContent += `Summary Sentence ${i+1},"${s.replace(/"/g, '""')}"\n`;
    });
    csvContent += `Full Text,"${lastScrapedData.full_text.replace(/"/g, '""')}"\n`;
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "cipher_scrape_analysis.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

async function featureMission() {
    stats.tasks++;
    const task = (document.getElementById('task-input') as HTMLInputElement).value;
    setBadge('badge-mission', 'thinking');
    const tl = document.getElementById('mission-timeline');
    const vCard = document.getElementById('mission-verdict');
    if (tl) tl.innerHTML = '';
    if (vCard) vCard.style.display = 'none';

    const planSys = `You are Cipher's task planner. Break the user's task into exactly 3 sequential concrete steps. Return JSON: {steps: [{id: 1, title: "Search", description: "Query"}]}`;
    try {
        const planRaw = await callCipher(planSys, task);
        const plan = parseJSON(planRaw);
        let history: string[] = [];

        for(let i=0; i<plan.steps.length; i++) {
            const step = plan.steps[i];
            const card = document.createElement('div');
            card.className = 'step-card running visible';
            card.innerHTML = `
                <div class="step-header"><span class="dot"></span>STEP ${step.id} OF ${plan.steps.length} — ${step.title.toUpperCase()}</div>
                <div class="step-body" id="step-b-${step.id}">Working...</div>
                <div class="step-time" id="step-t-${step.id}"></div>
            `;
            tl?.appendChild(card);
            
            const startT = performance.now();
            const execSys = `Execute step ${step.id}: ${step.description}. Context from previous steps: ${JSON.stringify(history)}. Return concise text findings (not JSON).`;
            const res = await callCipher(execSys, "Execute.", false);
            
            const stepBody = document.getElementById(`step-b-${step.id}`);
            if (stepBody) await typeText(stepBody, res, 10);
            
            const stepTime = document.getElementById(`step-t-${step.id}`);
            if (stepTime) stepTime.innerText = `✓ ${((performance.now()-startT)/1000).toFixed(1)}s`;
            
            card.className = 'step-card done visible';
            history.push(`Step ${step.id}: ${res}`);
        }

        const vSys = `Given findings: ${JSON.stringify(history)}. Provide a final recommendation in 2 sentences. (Not JSON)`;
        const verdict = await callCipher(vSys, "Summarize", false);
        if (vCard) {
            vCard.style.display = 'block';
            const vText = document.getElementById('mission-verdict-text');
            if (vText) await typeText(vText, verdict, 20);
        }
        
        setBadge('badge-mission', 'done', 'DONE');
    } catch(e: any) {
        const errCard = document.createElement('div');
        errCard.className = 'step-card visible';
        errCard.style.borderColor = 'var(--crimson-glow)';
        errCard.innerHTML = `<div class="step-header" style="color:var(--crimson-glow)">ERROR</div><div class="step-body">${e.message || "Mission failed"}</div>`;
        tl?.appendChild(errCard);
        setBadge('badge-mission', 'error', 'ERROR');
    }
}

async function featureForm() {
    stats.tasks++;
    const ins = (document.getElementById('form-instruction') as HTMLTextAreaElement).value;
    const formStatus = document.getElementById('form-status');
    if (formStatus) formStatus.style.display = 'none';
    
    ['f-from','f-to','f-dep','f-ret','f-pax','f-class','f-air','f-bud'].forEach(id => {
        const el = document.getElementById(id) as HTMLInputElement | HTMLSelectElement;
        if(el) {
            el.value = '';
            el.style.borderColor = 'var(--black-border)';
        }
    });

    const sys = `Extract flight details from instruction. Return JSON keys: from, to, dep, ret, pax, cls, air, bud. Infer missing values.`;
    try {
        const raw = await callCipher(sys, ins);
        const data = parseJSON(raw);
        if(data) {
            const map: any = { 'f-from': data.from, 'f-to': data.to, 'f-dep': data.dep, 'f-ret': data.ret, 'f-pax': data.pax, 'f-class': data.cls, 'f-air': data.air, 'f-bud': data.bud };
            
            for(let id in map) {
                if(map[id]) {
                    const el = document.getElementById(id) as HTMLInputElement | HTMLSelectElement;
                    if (!el) continue;
                    
                    el.style.borderColor = 'var(--crimson-core)';
                    if(el.tagName === 'SELECT') {
                        el.value = map[id];
                    } else {
                        await typeText(el, String(map[id]), 40);
                        el.value = map[id]; 
                    }
                    await new Promise(r => setTimeout(r, 150));
                    el.style.borderColor = 'var(--gold-dim)';
                }
            }
            if (formStatus) formStatus.style.display = 'block';
        }
    } catch(e: any) {
        if (formStatus) {
            formStatus.style.display = 'block';
            formStatus.style.color = 'var(--crimson-glow)';
            formStatus.innerText = 'Failed to extract form details: ' + (e.message || "Unknown error");
        }
    }
}

async function featureCompare() {
    stats.tasks++;
    const cat = (document.getElementById('comp-cat') as HTMLSelectElement).value;
    const items = [
        (document.getElementById('comp-1') as HTMLInputElement).value,
        (document.getElementById('comp-2') as HTMLInputElement).value,
        (document.getElementById('comp-3') as HTMLInputElement).value
    ].filter(Boolean);
    
    setBadge('badge-compare', 'thinking');
    const tb = document.getElementById('comp-table');
    const winnerCard = document.getElementById('comp-winner');
    if (tb) { tb.style.display = 'none'; tb.innerHTML = ''; }
    if (winnerCard) winnerCard.style.display = 'none';

    const sys = `Compare ${cat}: ${items.join(', ')}. Return JSON: { criteria: ["Price", "Performance", "Battery", "Verdict"], items: [{name:"x", values:["v1","v2"]}], winner: "x", justification: "text", confidence: 95 }`;
    
    try {
        const raw = await callCipher(sys, "Compare them objectively");
        const data = parseJSON(raw);
        
        if(data && tb && winnerCard) {
            let html = `<tr><th>Criteria</th>`;
            data.items.forEach((i: any) => html += `<th>${i.name}</th>`);
            html += `</tr>`;
            
            data.criteria.forEach((crit: string, cIdx: number) => {
                html += `<tr><td>${crit}</td>`;
                data.items.forEach((item: any, iIdx: number) => {
                    let cls = '';
                    if(data.winner === item.name) cls = 'winner-col';
                    if(cIdx === 0 && iIdx === 1) cls += ' cell-best'; 
                    html += `<td class="${cls}">${item.values[cIdx]}</td>`;
                });
                html += `</tr>`;
            });
            
            tb.innerHTML = html;
            tb.style.display = 'table';
            
            const wName = document.getElementById('comp-winner-name');
            const wJust = document.getElementById('comp-winner-just');
            const wConf = document.getElementById('comp-conf');
            
            if (wName) wName.innerText = `CIPHER'S PICK: ${data.winner}`;
            if (wJust) wJust.innerText = data.justification;
            if (wConf) wConf.innerText = `Confidence: ${data.confidence}%`;
            winnerCard.style.display = 'block';
            
            setBadge('badge-compare', 'done', 'DONE');
        } else {
            throw new Error("Invalid response format");
        }
    } catch(e: any) {
        if (winnerCard) {
            winnerCard.style.display = 'block';
            winnerCard.style.borderColor = 'var(--crimson-glow)';
            winnerCard.innerHTML = `<div style="color:var(--crimson-glow); font-weight:600">ERROR</div><div>${e.message || "Comparison failed"}</div>`;
        }
        setBadge('badge-compare', 'error', 'ERROR');
    }
}

async function featureWorkflow() {
    stats.tasks++;
    const goal = (document.getElementById('wf-goal') as HTMLInputElement).value;
    setBadge('badge-workflow', 'thinking', 'RUNNING');
    
    const nodes = ['wf-n1', 'wf-n2', 'wf-n3', 'wf-n4', 'wf-n5'];
    nodes.forEach(n => {
        const el = document.getElementById(n);
        if (el) el.className = 'node';
    });
    const out = document.getElementById('wf-output');
    const fill = document.getElementById('wf-fill');
    if (out) out.innerHTML = '';
    if (fill) fill.style.width = '0%';

    const stages = [
        {id:1, title: 'RESEARCH', sys: 'Provide 3 brief bullet points researching: '},
        {id:2, title: 'COMPARE', sys: 'Compare the findings and select the best path: '},
        {id:3, title: 'DECIDE', sys: 'Make a definitive choice and state it clearly: '},
        {id:4, title: 'EXECUTE', sys: 'Simulate executing the booking/action. State success: '},
        {id:5, title: 'CONFIRM', sys: 'Generate a final formatted itinerary/confirmation document: '}
    ];

    let context = goal;
    
    for(let i=0; i<stages.length; i++) {
        const node = document.getElementById(nodes[i]);
        if (node) node.classList.add('active');
        
        try {
            const res = await callCipher(stages[i].sys, `Goal: ${goal}. Context: ${context}`, false);
            context = res; 
            
            if (node) {
                node.classList.remove('active');
                node.classList.add('done');
                node.innerHTML = '✓';
            }
            
            if (fill) fill.style.width = `${i*25}%`;
            
            const card = document.createElement('div');
            card.className = i===4 ? 'wf-card final' : 'wf-card';
            card.style.display = 'block';
            card.innerHTML = `<div style="color:var(--gold-pure);font-family:'Rajdhani';margin-bottom:0.5rem;font-weight:700">STAGE ${i+1}: ${stages[i].title}</div><div class="content-text">${res.replace(/\n/g, '<br>')}</div>`;
            out?.appendChild(card);
            
        } catch(e: any) {
            if (node) node.style.borderColor = 'var(--crimson-glow)';
            const errCard = document.createElement('div');
            errCard.className = 'wf-card';
            errCard.style.display = 'block';
            errCard.style.borderColor = 'var(--crimson-glow)';
            errCard.innerHTML = `<div style="color:var(--crimson-glow);font-weight:700">FAILED AT STAGE ${i+1}</div><div class="content-text">${e.message || "Execution error"}</div>`;
            out?.appendChild(errCard);
            setBadge('badge-workflow', 'error', 'ERROR');
            return;
        }
    }
    setBadge('badge-workflow', 'done', 'COMPLETED');
}

// --- Chat Feature ---
let chatContextHistory: string = "";
async function featureChat() {
    const inputEl = document.getElementById('chat-input') as HTMLTextAreaElement;
    const msg = inputEl.value.trim();
    if (!msg) return;
    
    inputEl.value = '';
    const historyBox = document.getElementById('chat-history');
    if (!historyBox) return;
    
    stats.tasks++;
    
    // Add User Message
    const userDiv = document.createElement('div');
    userDiv.className = 'chat-message user';
    const initials = (memory.userName || 'Operator').substring(0, 2).toUpperCase() || 'OP';
    userDiv.innerHTML = `
        <div class="msg-avatar">${initials}</div>
        <div class="msg-bubble">${msg.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>
    `;
    historyBox.appendChild(userDiv);
    historyBox.scrollTop = historyBox.scrollHeight;
    
    // Add Agent Thinking Message
    const agentDiv = document.createElement('div');
    agentDiv.className = 'chat-message agent';
    agentDiv.innerHTML = `
        <div class="msg-avatar">C</div>
        <div class="msg-bubble" id="chat-resp-${stats.tasks}">...</div>
    `;
    historyBox.appendChild(agentDiv);
    historyBox.scrollTop = historyBox.scrollHeight;
    
    const respBubble = document.getElementById(`chat-resp-${stats.tasks}`);
    const btn = document.getElementById('btn-chat-send') as HTMLButtonElement;
    if (btn) btn.disabled = true;
    
    const sysPrompt = "You are Cipher, an intelligent agent OS assistant. Answer the user directly and concisely. Context of previous messages: " + chatContextHistory;
    
    try {
        const res = await callCipher(sysPrompt, msg, false);
        chatContextHistory += `\nUser: ${msg}\nCipher: ${res}`;
        
        // Keep context reasonable
        if (chatContextHistory.length > 2000) {
            chatContextHistory = chatContextHistory.substring(chatContextHistory.length - 2000);
        }
        
        if (respBubble) {
            respBubble.innerText = '';
            await typeText(respBubble, res, 20);
        }
    } catch (e: any) {
        if (respBubble) {
            respBubble.innerText = `Error: ${e.message || "Failed to reach agent"}`;
            respBubble.style.color = 'var(--crimson-glow)';
        }
    } finally {
        if (btn) btn.disabled = false;
        historyBox.scrollTop = historyBox.scrollHeight;
    }
}

// --- Voice Layer ---
let recognition: any = null;
function setupVoice() {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        
        recognition.onstart = function() {
            document.getElementById('voice-btn')?.classList.add('listening');
            appendLog('cmd', 'Voice listening started...');
        };
        
        recognition.onresult = async function(event: any) {
            const transcript = event.results[0][0].transcript;
            appendLog('cmd', `Heard: "${transcript}"`);
            
            // Contextual Dictation: If focused in an input/textarea, append the text.
            const activeEl = document.activeElement;
            if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA')) {
                const el = activeEl as HTMLInputElement | HTMLTextAreaElement;
                el.value = el.value ? el.value + ' ' + transcript : transcript;
                appendLog('success', 'Dictated into active field');
            } else {
                routeCommand(transcript);
            }
        };
        
        recognition.onend = function() {
            document.getElementById('voice-btn')?.classList.remove('listening');
        };
    } else {
        appendLog('error', 'Speech API not supported in this browser.');
    }
}

function toggleVoice() {
    if(!recognition) setupVoice();
    if(recognition) {
        if(document.getElementById('voice-btn')?.classList.contains('listening')) {
            recognition.stop();
        } else {
            recognition.start();
        }
    }
}

async function routeCommand(cmd: string) {
    const sys = `Route this command to a module. Return JSON: {module: "browse"|"scrape"|"tasks"|"forms"|"compare"|"workflow"|"chat"|"unknown", instruction: "extracted command"}`;
    try {
        const raw = await callCipher(sys, cmd);
        const data = parseJSON(raw);
        if(data && data.module !== 'unknown') {
            const map: any = {
                'browse': { id: 'agent-browse', input: 'browse-task', btn: featureBrowse },
                'scrape': { id: 'agent-scrape', input: 'scrape-url', btn: featureScrape },
                'tasks': { id: 'agent-tasks', input: 'task-input', btn: featureMission },
                'forms': { id: 'agent-forms', input: 'form-instruction', btn: featureForm },
                'compare': { id: 'agent-compare', input: 'comp-1', btn: null }, 
                'chat': { id: 'agent-chat', input: 'chat-input', btn: featureChat },
                'workflow': { id: 'agent-workflow', input: 'wf-goal', btn: featureWorkflow }
            };
            
            const target = map[data.module];
            if(target) {
                window.location.hash = target.id;
                const inEl = document.getElementById(target.input) as HTMLInputElement | HTMLTextAreaElement;
                if (inEl) inEl.value = data.instruction;
                if(target.btn) setTimeout(target.btn, 1000);
                appendLog('success', `Routed to ${data.module.toUpperCase()}`);
                speak(`Running ${data.module} protocol`);
            }
        } else {
            appendLog('error', 'Command not recognized.');
        }
    } catch(e) {}
}

function speak(text: string) {
    if('speechSynthesis' in window) {
        const msg = new SpeechSynthesisUtterance(text);
        msg.pitch = 0.8;
        msg.rate = 1.1;
        window.speechSynthesis.speak(msg);
    }
}

async function featureReviewer() {
    stats.tasks++;
    const input = (document.getElementById('reviewer-input') as HTMLTextAreaElement).value;
    if (!input.trim()) return;

    setBadge('badge-reviewer', 'thinking');
    const sentimentBox = document.getElementById('reviewer-sentiment');
    const verdictBox = document.getElementById('reviewer-verdict');
    
    if (sentimentBox) sentimentBox.innerText = 'Analyzing reviews...';
    if (verdictBox) verdictBox.innerText = '';

    try {
        const res = await fetch('http://localhost:8000/api/reviewer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reviews: input })
        });
        
        if (!res.ok) throw new Error("Backend error");
        
        const data = await res.json();
        
        if (sentimentBox) {
            sentimentBox.innerText = '';
            await typeText(sentimentBox, data.sentiment, 10);
        }
        if (verdictBox) {
            verdictBox.innerText = '';
            await typeText(verdictBox, data.verdict, 10);
        }
        
        setBadge('badge-reviewer', 'done', 'COMPLETED');
        appendLog('success', 'Review analysis complete.');
    } catch (e) {
        if (sentimentBox) sentimentBox.innerText = 'Error analyzing reviews.';
        setBadge('badge-reviewer', 'error', 'FAILED');
        appendLog('error', 'Reviewer feature failed.');
    }
}
