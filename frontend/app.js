/* ============================================
   SOURCEDOTCOM — AI ANALYST FRONTEND LOGIC
   Prepared by Sushant
   ============================================ */

// DOM refs
const dropZone     = document.getElementById('drop-zone');
const fileInput    = document.getElementById('file-input');
const chatInput    = document.getElementById('chat-input');
const sendChat     = document.getElementById('send-chat');
const chatBox      = document.getElementById('chat-box');
const statusPill   = document.getElementById('status-pill');
const statusText   = document.getElementById('status-text');
const progressWrap = document.getElementById('upload-progress');
const progressFill = document.getElementById('progress-fill');
const progressLbl  = document.getElementById('progress-label');
const dropTitle    = document.getElementById('drop-title');
const dropSub      = document.getElementById('drop-sub');
const downloadBtn  = document.getElementById('download-report-btn');

let currentSessionId = null;
let currentDomain = 'General';

// ============================================
//  1. DRAG & DROP (Emerald Theme)
// ============================================
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', (e) => {
    if (!dropZone.contains(e.relatedTarget)) {
        dropZone.classList.remove('drag-over');
    }
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
});

fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file) uploadFile(file);
});

// ============================================
//  2. FILE UPLOAD & ANALYSIS
// ============================================
async function uploadFile(file) {
    // Validate file type
    const validTypes = ['.csv', '.xlsx', '.xls', '.json'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validTypes.includes(ext)) {
        appendMessage('bot', `⚠️ Unsupported file type "${ext}". Please use CSV, Excel, or JSON.`);
        return;
    }

    setStatus('busy', 'Analyzing...');
    showProgress();
    appendMessage('bot', `🔬 Analyzing <strong>${file.name}</strong>…\nThe AI is routing your data to the right specialist.`);

    const formData = new FormData();
    formData.append('file', file);

    // Simulate progressive loading bar
    let pct = 5;
    const ticker = setInterval(() => {
        pct = Math.min(pct + Math.random() * 8, 88);
        setProgress(pct, 'AI processing…');
    }, 500);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        clearInterval(ticker);

        if (!response.ok) {
            const errData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errData.detail || `Server Busy: The analysis is taking longer than expected. Try a smaller dataset or refresh and try again.`);
        }

        const data = await response.json();

        setProgress(100, 'Complete!');

        // Slight delay to let progress bar hit 100
        await sleep(400);
        hideProgress();

        // Render results
        renderResults(data);
        setStatus('ready', 'Online');

    } catch (err) {
        clearInterval(ticker);
        hideProgress();
        setStatus('ready', 'Ready');
        console.error('[Copilot Error]', err);
        appendMessage('bot', `❌ Analysis failed: ${err.message}\n\nPlease check your file format or try again in a moment. The server may be busy processing other requests.`);

        // Reset drop zone
        resetDropZone();
    }
}

function renderResults(data) {
    // Extract Session ID for scalability
    currentSessionId = data.session_id;
    currentDomain = (data.domain || 'General').toLowerCase();

    // Update header
    const domain = (data.title || 'GENERAL ANALYST').replace(' ANALYST', '').trim();
    document.getElementById('analyst-title').innerText =
        `${data.title} · AI Strategy Prepared by Sushant`;

    const chipDomain = document.getElementById('chip-domain');
    if (chipDomain) chipDomain.innerText = domain;
    const headerStats = document.getElementById('header-stats');
    if (headerStats) headerStats.style.display = 'flex';

    // Strategy content (Render markdown directly)
    let htmlContent = data.strategy || '(No strategy generated)';
    // Basic Markdown Parser for headings and bold
    htmlContent = htmlContent.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    htmlContent = htmlContent.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    htmlContent = htmlContent.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    htmlContent = htmlContent.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    htmlContent = htmlContent.replace(/\n/gim, '<br>');
    htmlContent = htmlContent.replace(/<\/h2><br>/gim, '</h2>');
    htmlContent = htmlContent.replace(/<\/h3><br>/gim, '</h3>');

    document.getElementById('strategy-content').innerHTML = htmlContent;

    // Download Report Button Action
    const downloadBtn = document.getElementById('floating-download-btn');
    if (downloadBtn) {
        downloadBtn.style.display = 'flex';
        downloadBtn.onclick = () => {
            const domain = currentDomain.toLowerCase();
            window.open(`${window.location.origin}/download-pdf/${currentSessionId}`, '_blank');
        };
    }

    // Inject Detailed Dataset Anatomy and Metrics (Synced across both cards)
    if (data.summary) {
        // card 1: Anatomy
        document.getElementById('sum-anatomy-records').innerText = data.summary.records || '0';
        document.getElementById('sum-anatomy-cols').innerText = data.summary.columns || '0';
        document.getElementById('sum-anatomy-num').innerText = data.summary.numeric_feats || '0';
        document.getElementById('sum-anatomy-cat').innerText = data.summary.categorical_feats || '0';
        document.getElementById('sum-anatomy-dupes').innerText = data.summary.duplicates || '0';
        document.getElementById('sum-anatomy-missing').innerText = data.summary.missing_pct || '0%';
        document.getElementById('sum-anatomy-mem').innerText = data.summary.memory || '0 MB';

        // card 2: Section 1
        document.getElementById('sum-sec1-records').innerText = data.summary.records || '0';
        document.getElementById('sum-sec1-cols').innerText = data.summary.columns || '0';
        document.getElementById('sum-sec1-num').innerText = data.summary.numeric_feats || '0';
        document.getElementById('sum-sec1-cat').innerText = data.summary.categorical_feats || '0';
        document.getElementById('sum-sec1-missing').innerText = data.summary.missing_pct || '0%';
        document.getElementById('sum-sec1-mem').innerText = data.summary.memory || '0 MB';
        
        if (document.getElementById('metric-mae')) {
            document.getElementById('metric-mae').innerText = data.summary.mae || 'N/A';
            document.getElementById('metric-r2').innerText = data.summary.r2 || 'N/A';
        }
    }

    // Charts
    const ts = Date.now();
    const basePath = `/reports/${data.session_id}`;
    
    const c0 = document.getElementById('chart-0');
    const c1 = document.getElementById('chart-1');
    const c2 = document.getElementById('chart-2');
    const c3 = document.getElementById('chart-3');
    const c4 = document.getElementById('chart-4');
    const c5 = document.getElementById('chart-5');
    const c6 = document.getElementById('chart-6');
    const c7 = document.getElementById('chart-7');
    const c8 = document.getElementById('chart-8');
    const c9 = document.getElementById('chart-9');

    if (c0) c0.src = `${basePath}/dist.png?t=${ts}`;
    if (c1) c1.src = `${basePath}/box.png?t=${ts}`;
    if (c2) c2.src = `${basePath}/correlation.png?t=${ts}`;
    if (c3) c3.src = `${basePath}/trend.png?t=${ts}`;
    if (c4) c4.src = `${basePath}/scatter.png?t=${ts}`;
    if (c5) c5.src = `${basePath}/pred_vs_actual.png?t=${ts}`;
    if (c6) c6.src = `${basePath}/residual.png?t=${ts}`;
    if (c7) c7.src = `${basePath}/pie.png?t=${ts}`;
    if (c8) c8.src = `${basePath}/interaction.png?t=${ts}`;
    
    if (c9) {
        c9.onload = () => document.getElementById('chart-wrapper-9').style.display = 'flex';
        c9.onerror = () => document.getElementById('chart-wrapper-9').style.display = 'none';
        c9.src = `${basePath}/forecast.png?t=${ts}`;
    }

    // Explanations for Insight Boxes
    const explanations = {
        0: "Histograms show the frequency distribution of your primary metric. A concentrated bell shape suggests stability, whereas wide variance indicates performance inconsistency.",
        1: "Box plots identify 'noise' and outliers in your segments. The central box represents the middle 50% of your data, helping you spot anomalies rapidly.",
        2: "Correlation Heatmaps find hidden relationships. Values closer to +1 mean two factors rise together; -1 means they are inversely linked.",
        4: "Scatter analysis maps the direct 'cause-and-effect' visual between your top driver and your goal, revealing if the relationship is linear or complex.",
        7: "Composition charts reveal how much each sub-segment (e.g., Region, Department) contributes to the total metric volume.",
        3: "Historical trends show your growth trajectory over time, allowing you to identify seasonal spikes or sudden performance drops.",
        9: "The Forecasting module projections help you prepare for the next 30 steps of performance based on historical regression models.",
        5: "Prediction vs Actual validates model integrity. A diagonal alignment confirms the Neural Network has successfully 'learned' your business patterns.",
        6: "Residual plots check for 'bias.' Perfect models show random error scatter; patterns in residuals suggest the model is missing key variables.",
        8: "Feature Importance/SHAP scores quantify exactly which factors (e.g., Tenure, Region, Salary) exert the most force on your primary goal."
    };

    Object.keys(explanations).forEach(id => {
        const el = document.getElementById(`insight-${id}`);
        if (el) el.innerText = explanations[id];
    });

    // Show results area
    const resultsArea = document.getElementById('results-area');
    resultsArea.style.display = 'flex';

    // Collapse drop zone
    collapseDropZone();

    // Chat confirmation
    appendMessage('bot', `✅ Analysis Complete!\n\n💡 ${data.insight || 'Report generated successfully.'}\n\nYou can now ask me questions about your data.`);
}

// ============================================
//  3. CHAT
// ============================================
function setChatLoading(loading) {
    chatInput.disabled = loading;
    sendChat.disabled = loading;
    if (loading) {
        sendChat.innerHTML = '<span class="loading-spinner"></span>';
    } else {
        sendChat.innerText = 'Send';
        chatInput.focus();
    }
}

function scrollToBottom() {
    chatBox.scrollTo({
        top: chatBox.scrollHeight,
        behavior: 'smooth'
    });
}

async function sendMessage() {
    const question = chatInput.value.trim();
    if (!question || !currentSessionId) return;

    appendMessage('user', question);
    chatInput.value = '';
    scrollToBottom();
    
    setChatLoading(true);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId, question })
        });

        if (!response.ok) {
            const errData = await response.json();
            console.error('Chat Error Details:', errData);
            appendMessage('bot', `❌ Server Error: ${response.status} - Request could not be processed.`);
            return;
        }

        const data = await response.json();
        appendMessage('bot', data.answer);
        scrollToBottom();
    } catch (error) {
        console.error('Fetch Error:', error);
        appendMessage('bot', '❌ Connection Error: Ensure you have an active internet connection.');
    } finally {
        setChatLoading(false);
    }
}

// Attach Event Listeners
sendChat.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// ============================================
//  4. HELPERS
// ============================================
function appendMessage(sender, text) {
    const msg = document.createElement('div');
    msg.className = `message ${sender}`;

    if (sender === 'bot') {
        const icon = document.createElement('span');
        icon.className = 'msg-icon';
        icon.textContent = '◈';
        const p = document.createElement('p');
        // Support basic HTML in bot messages
        p.innerHTML = text.replace(/\n/g, '<br>');
        msg.appendChild(icon);
        msg.appendChild(p);
    } else {
        msg.innerText = text;
    }

    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function setStatus(type, label) {
    statusPill.className = 'status-pill';
    if (type === 'busy') statusPill.classList.add('busy');
    statusText.innerText = label;
}

function showProgress() {
    dropTitle.innerText = 'Uploading & Analyzing…';
    dropSub.style.display = 'none';
    document.querySelector('.upload-ring').innerHTML =
        `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#00ff88" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>`;
    document.querySelector('.upload-ring svg').style.animation = 'spin 1s linear infinite';
    progressWrap.style.display = 'block';
    setProgress(5, 'Starting…');
}

function hideProgress() {
    progressWrap.style.display = 'none';
}

function setProgress(pct, label) {
    progressFill.style.width = pct + '%';
    progressLbl.innerText = label;
}

function collapseDropZone() {
    dropZone.classList.add('collapsed');
    dropZone.style.minHeight = '70px';
    dropTitle.innerText = '↺  Drop another file to re-analyze';
    const p = dropZone.querySelector('p');
    if (p) p.style.display = 'none';
    const ring = dropZone.querySelector('.upload-ring');
    if (ring) ring.style.width = '36px';
    if (ring) ring.style.height = '36px';
    if (ring) ring.innerHTML =
        `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00ff88" stroke-width="2"><polyline points="16 16 12 12 8 16"></polyline><line x1="12" y1="12" x2="12" y2="21"></line><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path></svg>`;
}

function resetDropZone() {
    dropZone.classList.remove('collapsed');
    dropZone.style.minHeight = '';
    dropTitle.innerText = 'Drop Your Data File Here';
    if (dropSub) dropSub.style.display = '';
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
