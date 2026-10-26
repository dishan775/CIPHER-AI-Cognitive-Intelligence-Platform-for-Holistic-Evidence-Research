// --- Ollama-powered Cipher Agent Client ---
// Calls the backend proxy at /api/agent/chat which forwards to local Ollama

const BACKEND_URL = 'http://localhost:8000';

export const memory = {
    userName: null as string | null,
    preferences: {},
    sessionHistory: [] as string[]
};

export const stats = { calls: 0, tasks: 0, timeTotal: 0 };

export function updateStats(timeMs: number) {
    stats.calls++;
    stats.timeTotal += (timeMs / 1000);
    const callsEl = document.getElementById('stat-calls');
    const tasksEl = document.getElementById('stat-tasks');
    const respEl = document.getElementById('stat-resp');
    if(callsEl) callsEl.innerText = `Calls: ${stats.calls}`;
    if(tasksEl) tasksEl.innerText = `Tasks: ${stats.tasks}`;
    if(respEl) respEl.innerText = `Avg: ${(stats.timeTotal / stats.calls).toFixed(1)}s`;
}

export function appendLog(type: 'api' | 'success' | 'error' | 'cmd', msg: string) {
    const logBox = document.getElementById('mind-log');
    if (!logBox) return;
    
    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    const time = new Date().toISOString().substring(11, 19);
    entry.innerText = `[${time}] ${msg}`;
    logBox.prepend(entry);
    if (logBox.children.length > 200) {
        logBox.lastChild?.remove();
    }
}

export function parseJSON(raw: string): any {
    // Strip markdown code fences if present
    const clean = raw.replace(/```json|```/g, "").trim();
    try { 
        return JSON.parse(clean); 
    } catch(e) { 
        // Try to extract JSON from surrounding text
        const jsonMatch = clean.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            try {
                return JSON.parse(jsonMatch[0]);
            } catch(e2) {
                console.error("JSON parse error:", raw);
                return null;
            }
        }
        console.error("JSON parse error:", raw); 
        return null; 
    }
}

export async function typeText(el: HTMLElement, text: string, speed = 20) {
    el.innerHTML = '';
    for(let i=0; i<text.length; i++) {
        el.innerHTML += text.charAt(i);
        await new Promise(r => setTimeout(r, speed));
    }
}

export async function checkOllamaHealth(): Promise<{status: string, model: string, message: string}> {
    try {
        const res = await fetch(`${BACKEND_URL}/api/agent/health`);
        if (!res.ok) throw new Error('Backend not reachable');
        return await res.json();
    } catch(e) {
        return { status: 'disconnected', model: 'unknown', message: 'Backend server not reachable' };
    }
}

export async function callCipher(systemPrompt: string, userMessage: string, jsonMode = true): Promise<string> {
    appendLog('api', `Calling Ollama (local agent)...`);
    const start = performance.now();
    const memCtx = `User context: ${JSON.stringify(memory)}. Today: ${new Date().toDateString()}. `;
    
    const payload = {
        system_prompt: memCtx + systemPrompt,
        user_message: userMessage,
        json_mode: jsonMode
    };

    try {
        const res = await fetch(`${BACKEND_URL}/api/agent/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) {
            const errData = await res.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errData.detail || `HTTP ${res.status}`);
        }
        
        const data = await res.json();
        const end = performance.now();
        updateStats(end - start);
        
        const text = data.response;
        const modelUsed = data.model || 'qwen3:1.7b';
        const durationSec = ((end - start) / 1000).toFixed(1);
        
        appendLog('success', `${modelUsed} responded (${durationSec}s)`);
        return text;
    } catch(e: any) {
        appendLog('error', `Agent Error: ${e.message}`);
        throw e;
    }
}
