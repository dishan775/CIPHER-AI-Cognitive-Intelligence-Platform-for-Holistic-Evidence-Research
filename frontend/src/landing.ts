function initHeroArt() {
    const svg = document.getElementById('hero-svg');
    if (!svg) return;
    
    let html = '';
    
    // Stars
    for(let i=0; i<60; i++) {
        const x = Math.random() * 100;
        const y = Math.random() * 100;
        let color, r, op;
        if(i<30) { color = '#2F0909'; r = 0.8 + Math.random()*0.4; op = 0.25 + Math.random()*0.2; }
        else if(i<50) { color = 'var(--gold-pure)'; r = 0.6 + Math.random()*0.4; op = 0.15 + Math.random()*0.15; }
        else { color = 'var(--crimson-core)'; r = 1.0 + Math.random()*0.8; op = 0.2 + Math.random()*0.15; }
        const dur = 3 + Math.random()*4;
        const del = Math.random()*5;
        
        html += `<circle cx="${x}%" cy="${y}%" r="${r}" fill="${color}" opacity="${op}">
            <animate attributeName="opacity" values="${op};${op*0.2};${op}" dur="${dur}s" begin="${del}s" repeatCount="indefinite"/>
        </circle>`;
    }
    
    // Traces
    for(let i=0; i<18; i++) {
        const x1 = Math.random()*100; const y1 = Math.random()*100;
        const x2 = x1 + (Math.random()>0.5 ? 20 : 0); const y2 = y1 + (x2===x1 ? 20 : 0);
        const color = Math.random()>0.5 ? 'rgba(212,160,23,0.12)' : 'rgba(192,21,42,0.10)';
        html += `<line x1="${x1}%" y1="${y1}%" x2="${x2}%" y2="${y2}%" stroke="${color}" stroke-width="0.5"/>`;
    }

    // Ghost Rings
    html += `<circle cx="50%" cy="50%" r="420" fill="none" stroke="var(--crimson-core)" stroke-width="0.5" stroke-dasharray="10 30" opacity="0.06">
        <animateTransform attributeName="transform" type="rotate" from="0 50% 50%" to="360 50% 50%" dur="120s" repeatCount="indefinite"/>
    </circle>`;
    html += `<circle cx="50%" cy="50%" r="320" fill="none" stroke="var(--gold-pure)" stroke-width="0.5" stroke-dasharray="50 50" opacity="0.05">
        <animateTransform attributeName="transform" type="rotate" from="360 50% 50%" to="0 50% 50%" dur="90s" repeatCount="indefinite"/>
    </circle>`;

    svg.innerHTML = html;
}

document.addEventListener('DOMContentLoaded', () => {
    initHeroArt();
});
