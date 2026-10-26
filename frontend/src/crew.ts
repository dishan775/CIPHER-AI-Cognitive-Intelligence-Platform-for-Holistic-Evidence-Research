/**
 * CIPHER — Multi-Agent Research Collaboration Team
 * ===============================================
 * Interactive orchestrator for Researcher, Reviewer, Experiment Designer,
 * and Statistician agent collaboration.
 */

import { triggerVisualizerPipeline } from './visualizer';
import { appendLog } from './api/gemini';

const API_BASE = 'http://localhost:8000';

export interface AgentStepResult {
    role: string;
    title: string;
    icon: string;
    color: string;
    status: string;
    output: any;
    confidence: number;
    duration_ms: number;
}

export interface CrewRunData {
    id: number;
    topic: string;
    status: string;
    agents: AgentStepResult[];
    final_report: string;
    created_at: string;
}

export function initCrewPipeline() {
    const runBtn = document.getElementById('btn-launch-crew') as HTMLButtonElement | null;
    const topicInput = document.getElementById('crew-topic-input') as HTMLInputElement | null;
    const presets = document.querySelectorAll('.crew-preset-pill');
    const copyReportBtn = document.getElementById('btn-copy-crew-report');
    const downloadReportBtn = document.getElementById('btn-download-crew-report');

    // Preset click listeners
    presets.forEach(preset => {
        preset.addEventListener('click', () => {
            const topic = preset.getAttribute('data-topic');
            if (topic && topicInput) {
                topicInput.value = topic;
                topicInput.focus();
            }
        });
    });

    // Launch button click listener
    if (runBtn && topicInput) {
        runBtn.addEventListener('click', async () => {
            const topic = topicInput.value.trim();
            if (!topic || topic.length < 3) {
                alert('Please enter a research topic (at least 3 characters).');
                topicInput.focus();
                return;
            }

            const sidebarProj = document.getElementById('sidebar-active-project') as HTMLSelectElement | null;
            const projectId = sidebarProj?.value ? Number(sidebarProj.value) : undefined;

            await executeCrewPipeline(topic, projectId);
        });

        topicInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                runBtn.click();
            }
        });
    }

    // Report action buttons
    if (copyReportBtn) {
        copyReportBtn.addEventListener('click', () => {
            const reportEl = document.getElementById('crew-final-report-text');
            if (reportEl && reportEl.innerText) {
                navigator.clipboard.writeText(reportEl.innerText);
                const orig = copyReportBtn.innerHTML;
                copyReportBtn.innerHTML = '✅ Copied!';
                setTimeout(() => {
                    copyReportBtn.innerHTML = orig;
                }, 2000);
            }
        });
    }

    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', () => {
            const reportEl = document.getElementById('crew-final-report-text');
            const topicInput = document.getElementById('crew-topic-input') as HTMLInputElement | null;
            if (reportEl && reportEl.innerText) {
                const topicSlug = (topicInput?.value || 'research_report')
                    .toLowerCase()
                    .replace(/[^a-z0-9]+/g, '_')
                    .slice(0, 30);
                const blob = new Blob([reportEl.innerText], { type: 'text/markdown;charset=utf-8' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `CIPHER_Crew_${topicSlug}.md`;
                a.click();
                URL.revokeObjectURL(url);
            }
        });
    }

    // Load initial run history
    loadCrewRunHistory();
}

/**
 * Execute the multi-agent pipeline sequentially
 */
export async function executeCrewPipeline(topic: string, projectId?: number) {
    const runBtn = document.getElementById('btn-launch-crew') as HTMLButtonElement | null;
    const stepperContainer = document.getElementById('crew-stepper-container');
    const stepperStatus = document.getElementById('crew-stepper-status');
    const trackProgress = document.getElementById('crew-track-progress');
    const resultsContainer = document.getElementById('crew-results-container');
    const reportContainer = document.getElementById('crew-report-container');

    // UI state: Running
    if (runBtn) {
        runBtn.disabled = true;
        runBtn.innerHTML = `
            <svg class="spinner-svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="animation: spin 1s linear infinite;">
                <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
                <path d="M12 2a10 10 0 0 1 10 10"></path>
            </svg>
            Agents Collaborating...
        `;
    }

    // Trigger HUD visualizer preset
    try {
        triggerVisualizerPipeline('crewResearch', true);
    } catch (e) {
        console.warn('Visualizer trigger error:', e);
    }

    // Reveal stepper
    if (stepperContainer) stepperContainer.style.display = 'block';
    if (stepperStatus) {
        stepperStatus.className = 'crew-stepper-status running';
        stepperStatus.innerText = 'PIPELINE ACTIVE';
    }

    // Reset nodes
    const roles = ['researcher', 'reviewer', 'experiment_designer', 'statistician'];
    roles.forEach(role => {
        const node = document.getElementById(`crew-step-${role}`);
        if (node) {
            node.className = 'crew-step-node';
        }
    });

    // Start with Researcher active
    const rNode = document.getElementById('crew-step-researcher');
    if (rNode) rNode.className = 'crew-step-node active';
    if (trackProgress) trackProgress.style.width = '12%';

    // Clear previous results while loading
    if (resultsContainer) resultsContainer.innerHTML = '';
    if (reportContainer) reportContainer.style.display = 'none';

    appendLog('cmd', `[Multi-Agent] Initializing collaborative research on: "${topic}"`);

    // Simulated stepper progression while awaiting API
    let stepIdx = 0;
    const stepInterval = setInterval(() => {
        stepIdx = (stepIdx + 1) % roles.length;
        roles.forEach((r, idx) => {
            const node = document.getElementById(`crew-step-${r}`);
            if (!node) return;
            if (idx < stepIdx) {
                node.className = 'crew-step-node completed';
            } else if (idx === stepIdx) {
                node.className = 'crew-step-node active';
            } else {
                node.className = 'crew-step-node';
            }
        });
        if (trackProgress) {
            trackProgress.style.width = `${Math.min(25 + stepIdx * 25, 90)}%`;
        }
    }, 4000);

    try {
        const payload: any = { topic };
        if (projectId) payload.project_id = projectId;

        const resp = await fetch(`${API_BASE}/api/crew/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        clearInterval(stepInterval);

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Pipeline execution failed');
        }

        const data: CrewRunData = await resp.json();

        // Mark all steps completed
        roles.forEach(role => {
            const node = document.getElementById(`crew-step-${role}`);
            if (node) node.className = 'crew-step-node completed';
        });
        if (trackProgress) trackProgress.style.width = '100%';
        if (stepperStatus) {
            stepperStatus.className = 'crew-stepper-status completed';
            stepperStatus.innerText = 'CONSENSUS REACHED';
        }

        appendLog('success', `[Multi-Agent] All 4 agents completed analysis for run #${data.id}`);

        // Render findings
        renderCrewResults(data);
        loadCrewRunHistory();

    } catch (err: any) {
        clearInterval(stepInterval);
        console.error('Crew pipeline error:', err);
        appendLog('error', `[Multi-Agent] Error: ${err.message}`);
        if (stepperStatus) {
            stepperStatus.className = 'crew-stepper-status';
            stepperStatus.innerText = 'EXECUTION FAILED';
        }
        alert(`Multi-Agent Pipeline failed: ${err.message}`);
    } finally {
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
                Launch Multi-Agent Pipeline
            `;
        }
    }
}

/**
 * Render all agent results and final synthesized report
 */
export function renderCrewResults(data: CrewRunData) {
    const resultsContainer = document.getElementById('crew-results-container');
    const reportContainer = document.getElementById('crew-report-container');
    const reportText = document.getElementById('crew-final-report-text');

    if (!resultsContainer) return;
    resultsContainer.innerHTML = '';

    // Render each agent card
    data.agents.forEach(agent => {
        const card = document.createElement('div');
        card.className = 'crew-result-card';
        card.innerHTML = buildAgentCardHTML(agent);
        resultsContainer.appendChild(card);
    });

    // Render final consolidated report
    if (reportContainer && reportText && data.final_report) {
        reportText.innerText = data.final_report;
        reportContainer.style.display = 'block';
    }
}

function buildAgentCardHTML(agent: AgentStepResult): string {
    const role = agent.role;
    const out = agent.output || {};

    let contentHTML = '';

    if (role === 'researcher') {
        const summary = out.summary || 'Literature survey completed.';
        const papers = out.key_papers || [];
        const entities = out.entities || [];
        const claims = out.claims || [];
        const gaps = out.gaps || [];

        contentHTML = `
            <div class="crew-summary-box">${escapeHtml(summary)}</div>

            ${papers.length > 0 ? `
                <h4 style="font-size:0.85rem; font-weight:700; color:#2F0909; margin-bottom:0.6rem;">📚 Key Identified Papers</h4>
                <div class="crew-papers-list">
                    ${papers.map((p: any) => `
                        <div class="crew-paper-item">
                            <div class="crew-paper-title">
                                <span>${escapeHtml(p.title || 'Untitled')}</span>
                                <span class="crew-paper-year">${p.year || '2024'}</span>
                            </div>
                            <div class="crew-paper-contrib">${escapeHtml(p.contribution || '')}</div>
                        </div>
                    `).join('')}
                </div>
            ` : ''}

            ${entities.length > 0 ? `
                <h4 style="font-size:0.85rem; font-weight:700; color:#2F0909; margin-bottom:0.5rem;">🏷️ Core Conceptual Entities</h4>
                <div class="crew-tag-cloud">
                    ${entities.map((e: any) => `
                        <span class="crew-entity-tag">${escapeHtml(typeof e === 'string' ? e : e.text || '')}</span>
                    `).join('')}
                </div>
            ` : ''}

            ${claims.length > 0 ? `
                <h4 style="font-size:0.85rem; font-weight:700; color:#2F0909; margin-bottom:0.5rem;">💡 Extracted Empirical Claims</h4>
                <div class="crew-claims-list">
                    ${claims.map((c: string) => `
                        <div class="crew-claim-item">
                            <span class="crew-claim-bullet">▸</span>
                            <span>${escapeHtml(c)}</span>
                        </div>
                    `).join('')}
                </div>
            ` : ''}

            ${gaps.length > 0 ? `
                <h4 style="font-size:0.85rem; font-weight:700; color:#991B1B; margin-bottom:0.5rem;">⚠️ Unaddressed Research Gaps</h4>
                <div class="crew-gaps-list">
                    ${gaps.map((g: string) => `
                        <div class="crew-gap-item">
                            <span>⚡</span>
                            <span>${escapeHtml(g)}</span>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        `;
    } else if (role === 'reviewer') {
        const assessment = out.overall_assessment || 'Review completed.';
        const score = out.methodology_score !== undefined ? Math.round(out.methodology_score * 100) : 75;
        const strengths = out.strengths || [];
        const weaknesses = out.weaknesses || [];
        const biasFlags = out.bias_flags || [];
        const recs = out.recommendations || [];

        contentHTML = `
            <div class="crew-score-row">
                <div class="crew-score-card">
                    <div class="crew-score-val">${score}%</div>
                    <div class="crew-score-lbl">Methodology Rigor Score</div>
                </div>
                <div style="flex:1; font-size:0.88rem; color:#2F0909; line-height:1.5;">
                    ${escapeHtml(assessment)}
                </div>
            </div>

            <div class="crew-strengths-weaknesses">
                <div class="crew-sw-box strengths">
                    <div class="crew-sw-title" style="color:#059669;">✅ Identified Strengths</div>
                    <ul class="crew-sw-list">
                        ${strengths.map((s: string) => `<li>• ${escapeHtml(s)}</li>`).join('')}
                    </ul>
                </div>
                <div class="crew-sw-box weaknesses">
                    <div class="crew-sw-title" style="color:#D97706;">⚠️ Methodological Concerns</div>
                    <ul class="crew-sw-list">
                        ${weaknesses.map((w: string) => `<li>• ${escapeHtml(w)}</li>`).join('')}
                    </ul>
                </div>
            </div>

            ${biasFlags.length > 0 ? `
                <h4 style="font-size:0.85rem; font-weight:700; color:#991B1B; margin-bottom:0.5rem;">🚩 Detected Potential Biases</h4>
                <div class="crew-claims-list">
                    ${biasFlags.map((b: string) => `
                        <div class="crew-claim-item" style="border-left:3px solid #F59E0B;">
                            <span>⚠️</span>
                            <span>${escapeHtml(b)}</span>
                        </div>
                    `).join('')}
                </div>
            ` : ''}

            ${recs.length > 0 ? `
                <h4 style="font-size:0.85rem; font-weight:700; color:#2F0909; margin-bottom:0.5rem;">🎯 Peer Review Recommendations</h4>
                <div class="crew-claims-list">
                    ${recs.map((r: string) => `
                        <div class="crew-claim-item">
                            <span style="color:#F59E0B;">★</span>
                            <span>${escapeHtml(r)}</span>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        `;
    } else if (role === 'experiment_designer') {
        const title = out.experiment_title || 'Proposed Experimental Protocol';
        const hypothesis = out.hypothesis || 'Hypothesis under formulation.';
        const methodology = out.methodology || '';
        const variables = out.variables || {};
        const phases = out.phases || [];
        const outcomes = out.expected_outcomes || [];

        contentHTML = `
            <div style="margin-bottom:1rem;">
                <div style="font-size:1rem; font-weight:800; color:#059669; margin-bottom:0.3rem;">⚗️ ${escapeHtml(title)}</div>
                <div style="font-size:0.88rem; line-height:1.5; color:#2F0909;"><strong>Hypothesis:</strong> ${escapeHtml(hypothesis)}</div>
            </div>

            ${methodology ? `
                <div class="crew-summary-box" style="border-left-color:#10B981; margin-bottom:1.2rem;">
                    ${escapeHtml(methodology)}
                </div>
            ` : ''}

            <h4 style="font-size:0.85rem; font-weight:700; color:#2F0909; margin-bottom:0.6rem;">🔬 Variable Taxonomy</h4>
            <div class="crew-variables-grid">
                <div class="crew-var-card">
                    <div class="crew-var-title">Independent Variables</div>
                    <ul class="crew-sw-list">
                        ${(variables.independent || []).map((v: string) => `<li>• ${escapeHtml(v)}</li>`).join('')}
                    </ul>
                </div>
                <div class="crew-var-card">
                    <div class="crew-var-title">Dependent Variables</div>
                    <ul class="crew-sw-list">
                        ${(variables.dependent || []).map((v: string) => `<li>• ${escapeHtml(v)}</li>`).join('')}
                    </ul>
                </div>
                <div class="crew-var-card">
                    <div class="crew-var-title">Control Variables</div>
                    <ul class="crew-sw-list">
                        ${(variables.control || []).map((v: string) => `<li>• ${escapeHtml(v)}</li>`).join('')}
                    </ul>
                </div>
            </div>

            ${phases.length > 0 ? `
                <h4 style="font-size:0.85rem; font-weight:700; color:#2F0909; margin-bottom:0.6rem;">📅 Execution Phases</h4>
                <div class="crew-phases-timeline">
                    ${phases.map((ph: any) => `
                        <div class="crew-phase-item">
                            <div>
                                <div class="crew-phase-name">${escapeHtml(ph.name || 'Phase')}</div>
                                <div class="crew-phase-desc">${escapeHtml(ph.description || '')}</div>
                            </div>
                            <span class="crew-phase-duration">${escapeHtml(ph.duration || '2 weeks')}</span>
                        </div>
                    `).join('')}
                </div>
            ` : ''}

            ${outcomes.length > 0 ? `
                <h4 style="font-size:0.85rem; font-weight:700; color:#059669; margin-bottom:0.5rem;">🎯 Expected Empirical Outcomes</h4>
                <div class="crew-claims-list">
                    ${outcomes.map((o: string) => `
                        <div class="crew-claim-item">
                            <span style="color:#10B981;">✓</span>
                            <span>${escapeHtml(o)}</span>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        `;
    } else if (role === 'statistician') {
        const framework = out.statistical_framework || 'Statistical analysis framework.';
        const sampleSize = out.sample_size || {};
        const tests = out.tests || [];
        const metrics = out.metrics || [];
        const validation = out.validation_strategy || '';
        const risks = out.risk_assessment || [];

        contentHTML = `
            <div class="crew-summary-box" style="border-left-color:#8B5CF6; margin-bottom:1.2rem;">
                ${escapeHtml(framework)}
            </div>

            <h4 style="font-size:0.85rem; font-weight:700; color:#2F0909; margin-bottom:0.6rem;">📊 Power Analysis & Sample Sizing</h4>
            <div class="crew-stat-params">
                <div class="crew-stat-card">
                    <div class="crew-stat-num">N = ${sampleSize.recommended || 300}</div>
                    <div class="crew-stat-lbl">Recommended Size</div>
                </div>
                <div class="crew-stat-card">
                    <div class="crew-stat-num">${Math.round((sampleSize.power || 0.8) * 100)}%</div>
                    <div class="crew-stat-lbl">Statistical Power (1-β)</div>
                </div>
                <div class="crew-stat-card">
                    <div class="crew-stat-num">α = ${sampleSize.alpha || 0.05}</div>
                    <div class="crew-stat-lbl">Significance Level</div>
                </div>
                <div class="crew-stat-card">
                    <div class="crew-stat-num" style="font-size:0.95rem; line-height:2rem;">${sampleSize.effect_size || "Cohen's d ≥ 0.5"}</div>
                    <div class="crew-stat-lbl">Target Effect Size</div>
                </div>
            </div>

            ${tests.length > 0 ? `
                <h4 style="font-size:0.85rem; font-weight:700; color:#2F0909; margin-bottom:0.6rem;">🔬 Recommended Statistical Tests</h4>
                <table class="crew-tests-table">
                    <thead>
                        <tr>
                            <th>Test / Procedure</th>
                            <th>Experimental Purpose</th>
                            <th>Key Assumptions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${tests.map((t: any) => `
                            <tr>
                                <td><strong>${escapeHtml(t.name || '')}</strong></td>
                                <td>${escapeHtml(t.purpose || '')}</td>
                                <td>${escapeHtml(t.assumptions || '')}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            ` : ''}

            ${metrics.length > 0 ? `
                <h4 style="font-size:0.85rem; font-weight:700; color:#2F0909; margin-bottom:0.6rem;">🎯 Evaluated Statistical Metrics</h4>
                <div class="crew-claims-list">
                    ${metrics.map((m: any) => `
                        <div class="crew-claim-item" style="border-left:3px solid #8B5CF6;">
                            <span>📈</span>
                            <span><strong>${escapeHtml(m.name || 'Metric')}</strong> (${escapeHtml(m.type || 'evaluation')}) — Threshold: <code>${escapeHtml(m.threshold || 'N/A')}</code></span>
                        </div>
                    `).join('')}
                </div>
            ` : ''}

            ${validation ? `
                <div style="font-size:0.85rem; background:rgba(139,92,246,0.05); border:1px solid rgba(139,92,246,0.2); border-radius:8px; padding:0.8rem 1rem; margin-bottom:1rem; color:#2F0909;">
                    <strong>Validation Protocol:</strong> ${escapeHtml(validation)}
                </div>
            ` : ''}

            ${risks.length > 0 ? `
                <h4 style="font-size:0.85rem; font-weight:700; color:#5B21B6; margin-bottom:0.5rem;">🛡️ Statistical Risk Mitigation</h4>
                <div class="crew-claims-list">
                    ${risks.map((r: any) => `
                        <div class="crew-claim-item">
                            <span style="color:#7C3AED;">●</span>
                            <span><strong>${escapeHtml(r.risk || 'Risk')}:</strong> ${escapeHtml(r.mitigation || '')}</span>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        `;
    }

    return `
        <div class="crew-result-header">
            <div class="crew-result-agent-info">
                <span class="crew-result-agent-icon">${agent.icon || '🤖'}</span>
                <div>
                    <span class="crew-result-agent-title">${escapeHtml(agent.title || agent.role)}</span>
                    <span style="font-size:0.75rem; color:rgba(47,9,9,0.5); margin-left:0.5rem; text-transform:uppercase;">Role: ${escapeHtml(agent.role)}</span>
                </div>
            </div>
            <div class="crew-result-badge-list">
                <span class="crew-confidence-pill">Confidence: ${Math.round((agent.confidence || 0.8) * 100)}%</span>
                <span class="crew-duration-pill">${agent.duration_ms ? `${agent.duration_ms}ms` : 'Instant'}</span>
            </div>
        </div>
        <div class="crew-result-body">
            ${contentHTML}
        </div>
    `;
}

/**
 * Load and render recent crew runs
 */
export async function loadCrewRunHistory() {
    const listEl = document.getElementById('crew-history-list');
    if (!listEl) return;

    try {
        const resp = await fetch(`${API_BASE}/api/crew/runs?limit=8`);
        if (!resp.ok) return;
        const runs = await resp.json();

        if (runs.length === 0) {
            listEl.innerHTML = `<div style="font-size:0.82rem; color:rgba(47,9,9,0.5); padding:0.5rem 0;">No collaborative runs recorded yet. Launch a topic above!</div>`;
            return;
        }

        listEl.innerHTML = runs.map((run: any) => `
            <div class="crew-history-item" data-run-id="${run.id}">
                <div>
                    <div class="crew-history-topic">${escapeHtml(run.topic)}</div>
                    <div class="crew-history-meta">Run #${run.id} · ${run.agents ? run.agents.length : 4} Agents · ${formatDate(run.created_at)}</div>
                </div>
                <div style="display:flex; align-items:center; gap:0.5rem;">
                    <button class="btn-ghost" style="padding:0.4rem 0.8rem; font-size:0.72rem;" onclick="window.viewCrewRun(${run.id})">Inspect</button>
                </div>
            </div>
        `).join('');

    } catch (e) {
        console.warn('Failed to load crew history:', e);
    }
}

// Global inspect handler
(window as any).viewCrewRun = async (runId: number) => {
    try {
        const resp = await fetch(`${API_BASE}/api/crew/runs/${runId}`);
        if (!resp.ok) throw new Error('Run not found');
        const data: CrewRunData = await resp.json();

        // Populate topic
        const topicInput = document.getElementById('crew-topic-input') as HTMLInputElement | null;
        if (topicInput) topicInput.value = data.topic;

        // Show stepper
        const stepperContainer = document.getElementById('crew-stepper-container');
        const stepperStatus = document.getElementById('crew-stepper-status');
        const trackProgress = document.getElementById('crew-track-progress');
        if (stepperContainer) stepperContainer.style.display = 'block';
        if (stepperStatus) {
            stepperStatus.className = 'crew-stepper-status completed';
            stepperStatus.innerText = `RUN #${data.id} (ARCHIVED)`;
        }
        if (trackProgress) trackProgress.style.width = '100%';

        const roles = ['researcher', 'reviewer', 'experiment_designer', 'statistician'];
        roles.forEach(role => {
            const node = document.getElementById(`crew-step-${role}`);
            if (node) node.className = 'crew-step-node completed';
        });

        renderCrewResults(data);

        // Smooth scroll to results
        document.getElementById('crew-stepper-container')?.scrollIntoView({ behavior: 'smooth' });

    } catch (err: any) {
        alert(`Failed to load run: ${err.message}`);
    }
};

function formatDate(isoStr?: string): string {
    if (!isoStr) return 'Recently';
    try {
        const d = new Date(isoStr);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch {
        return isoStr;
    }
}

function escapeHtml(str: string): string {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
