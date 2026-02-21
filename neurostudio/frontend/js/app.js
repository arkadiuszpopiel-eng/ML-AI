/**
 * NeuroForge - Frontend Application v0.7.0
 * Handles WebSocket chat, model management, conversation history,
 * system monitor, prompt templates, RAG, file upload, sidebar tabs,
 * chat toolbar capabilities, resource panel, download cancel,
 * and multi-agent role model selection.
 */

// ──── State ────
const state = {
    ws: null,
    monitorWs: null,
    sessionId: null,
    connected: false,
    modelLoaded: false,
    sending: false,
    monitorOpen: false,
    attachedFile: null,
    activeDownloads: {},  // filename -> AbortController
    capabilities: {
        internet: true,
        code_exec: true,
        filesystem: true,
        shell: true,
        rag: true,
    },
};

// ──── DOM Elements ────
const $ = (id) => document.getElementById(id);
const chatMessages = $('chat-messages');
const chatInput = $('chat-input');
const btnSend = $('btn-send');
const btnNewChat = $('btn-new-chat');
const modelSelect = $('model-select');
const btnLoadModel = $('btn-load-model');
const btnUnloadModel = $('btn-unload-model');
const statusDot = $('status-dot');
const statusText = $('status-text');
const typingIndicator = $('typing-indicator');
const sidebarToggle = $('sidebar-toggle');
const sidebar = $('sidebar');
const tempSlider = $('temperature');
const tempValue = $('temp-value');

// ──── Initialize ────
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    loadModels();
    loadRecommendedModels();
    loadStatus();
    checkEngineSetup();
    loadConversationHistory();
    loadDocumentsList();
    loadTemplates();
    loadRouterConfig();
    loadProviders();
    loadAgentRoles();
    setupEventListeners();
    setupSidebarTabs();
    setupChatToolbar();
    setupResourcePanel();
    startMiniMonitor();
});

// ──── WebSocket ────
function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/chat`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        state.connected = true;
        console.log('WebSocket connected');
    };

    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWSMessage(data);
    };

    state.ws.onclose = () => {
        state.connected = false;
        console.log('WebSocket disconnected, reconnecting in 3s...');
        setTimeout(connectWebSocket, 3000);
    };

    state.ws.onerror = (err) => {
        console.error('WebSocket error:', err);
    };
}

function handleWSMessage(data) {
    switch (data.type) {
        case 'session':
            state.sessionId = data.session_id;
            break;

        case 'text':
            appendAssistantText(data.content);
            break;

        case 'tool_call':
            appendToolCall(data.tool, data.args);
            break;

        case 'tool_result':
            appendToolResult(data.tool, data.result);
            break;

        case 'model_switch':
            appendModelSwitch(data.from, data.to);
            break;

        case 'provider_info':
            appendProviderInfo(data.provider, data.model, data.routed);
            break;

        case 'fallback':
            appendFallbackNotice(data.from, data.to);
            break;

        case 'usage_update':
            updateProviderBarUsage(data.session_tokens, data.session_cost);
            break;

        case 'error':
            appendError(data.message);
            break;

        case 'done':
            state.sending = false;
            btnSend.disabled = false;
            typingIndicator.classList.add('hidden');
            scrollToBottom();
            // Refresh conversation list (auto-save happened on server)
            loadConversationHistory();
            break;

        case 'cleared':
            clearChat();
            break;
    }
}

function sendMessage(text) {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
        appendError('Brak polaczenia z serwerem. Odswież strone.');
        return;
    }
    if (!text.trim() && !state.attachedFile) return;

    state.sending = true;
    btnSend.disabled = true;
    typingIndicator.classList.remove('hidden');

    // Remove welcome message
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    // Handle file attachment
    let finalText = text;
    if (state.attachedFile) {
        finalText = `[Zalaczony plik: ${state.attachedFile.name}, sciezka: ${state.attachedFile.path}]\n\n${text}`;
        removeAttachment();
    }

    // Add user message to UI
    appendUserMessage(finalText);

    // Send via WebSocket (include temperature from slider)
    const temperature = parseFloat(tempSlider.value) || 0.7;
    state.ws.send(JSON.stringify({ type: 'message', content: finalText, temperature }));

    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';
}

// ──── Message Rendering ────

let currentAssistantEl = null;

function appendUserMessage(text) {
    currentAssistantEl = null;
    const el = document.createElement('div');
    el.className = 'message user';
    el.innerHTML = `
        <div class="message-avatar">U</div>
        <div class="message-content"><p>${escapeHtml(text)}</p></div>
    `;
    chatMessages.appendChild(el);
    scrollToBottom();
}

function appendAssistantText(text) {
    if (!currentAssistantEl) {
        const el = document.createElement('div');
        el.className = 'message assistant';
        el.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="message-content"></div>
        `;
        chatMessages.appendChild(el);
        currentAssistantEl = el.querySelector('.message-content');
    }

    // Render markdown-like content
    const rendered = renderMarkdown(text);
    const p = document.createElement('div');
    p.innerHTML = rendered;
    currentAssistantEl.appendChild(p);
    scrollToBottom();
}

function appendToolCall(toolName, args) {
    if (!currentAssistantEl) {
        appendAssistantText('');
    }

    const el = document.createElement('div');
    el.className = 'tool-call';

    const argsStr = Object.entries(args || {})
        .map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)
        .join('\n');

    el.innerHTML = `
        <div class="tool-call-header" onclick="this.nextElementSibling.classList.toggle('hidden')">
            &#9881; ${escapeHtml(toolName)}
        </div>
        <div class="tool-call-args">${escapeHtml(argsStr)}</div>
    `;
    currentAssistantEl.appendChild(el);
    scrollToBottom();
}

function appendToolResult(toolName, result) {
    if (!currentAssistantEl) return;

    const el = document.createElement('div');
    el.className = 'tool-result' + (result.success === false ? ' error' : '');

    let content;
    if (result.error) {
        content = `Error: ${result.error}`;
    } else if (typeof result.result === 'string') {
        content = result.result.substring(0, 2000);
    } else {
        content = JSON.stringify(result.result, null, 2).substring(0, 2000);
    }

    el.innerHTML = `<pre>${escapeHtml(content)}</pre>`;
    currentAssistantEl.appendChild(el);
    scrollToBottom();
}

function appendModelSwitch(fromModel, toModel) {
    const el = document.createElement('div');
    el.className = 'model-switch';
    el.textContent = `Model: ${fromModel || 'brak'} \u2192 ${toModel || '?'}`;
    chatMessages.appendChild(el);
    scrollToBottom();
}

function appendProviderInfo(providerName, model, routed) {
    const el = document.createElement('div');
    el.className = 'model-switch';
    const routedTag = routed ? ' <small style="opacity:0.6">(smart routing)</small>' : '';
    el.innerHTML = `<span class="provider-badge cloud">${escapeHtml(providerName)}</span> ${model ? escapeHtml(model) : ''}${routedTag}`;
    chatMessages.appendChild(el);
    scrollToBottom();
}

function appendFallbackNotice(fromProvider, toProvider) {
    const el = document.createElement('div');
    el.className = 'model-switch';
    el.innerHTML = `<span class="provider-badge" style="background:var(--warning);color:#333">Fallback</span> ${escapeHtml(fromProvider)} &rarr; ${escapeHtml(toProvider)}`;
    chatMessages.appendChild(el);
    scrollToBottom();
}

function appendError(message) {
    if (!currentAssistantEl) {
        appendAssistantText('');
    }
    const el = document.createElement('div');
    el.className = 'tool-result error';
    el.innerHTML = `<pre>${escapeHtml(message)}</pre>`;
    currentAssistantEl.appendChild(el);
    state.sending = false;
    btnSend.disabled = false;
    typingIndicator.classList.add('hidden');
    scrollToBottom();
}

function clearChat() {
    currentAssistantEl = null;
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <h1>&#9889; NeuroForge</h1>
            <p>Lokalne studio AI z pelnym dostepem do Twojego komputera.</p>
            <p class="hint">Zaladuj model i zacznij rozmowe!</p>
        </div>
    `;
}

// ──── Conversation History ────

async function loadConversationHistory() {
    try {
        const resp = await fetch('/api/conversations');
        const data = await resp.json();
        const container = $('conversation-list');

        if (!data.conversations || data.conversations.length === 0) {
            container.innerHTML = '<div class="conversation-list-empty">Brak zapisanych rozmow</div>';
            return;
        }

        container.innerHTML = '';
        for (const conv of data.conversations.slice(0, 20)) {
            const el = document.createElement('div');
            el.className = 'conversation-item';
            if (conv.session_id === state.sessionId) {
                el.classList.add('active');
            }

            const date = new Date(conv.updated_at * 1000);
            const dateStr = date.toLocaleDateString('pl-PL') + ' ' + date.toLocaleTimeString('pl-PL', {hour: '2-digit', minute: '2-digit'});

            el.innerHTML = `
                <div class="conv-title">${escapeHtml(conv.title || 'Nowa rozmowa')}</div>
                <div class="conv-meta">
                    ${dateStr} &middot; ${conv.message_count || 0} wiad.
                    <span class="conv-actions">
                        <button class="conv-btn" onclick="event.stopPropagation(); exportConversation('${conv.session_id}', 'markdown')" title="Eksport MD">&#128196;</button>
                        <button class="conv-btn" onclick="event.stopPropagation(); exportConversation('${conv.session_id}', 'json')" title="Eksport JSON">&#123;&#125;</button>
                        <button class="conv-btn conv-btn-del" onclick="event.stopPropagation(); if(confirm('Usunac?')) deleteConversation('${conv.session_id}')" title="Usun">&#128465;</button>
                    </span>
                </div>
            `;
            el.onclick = () => switchToConversation(conv.session_id);

            container.appendChild(el);
        }
    } catch (e) {
        console.error('Failed to load conversation history:', e);
    }
}

function switchToConversation(sessionId) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({ type: 'set_session', session_id: sessionId }));

        // Load conversation messages into UI
        fetch(`/api/conversations/${sessionId}`)
            .then(r => r.json())
            .then(data => {
                if (!data.messages) return;
                chatMessages.innerHTML = '';
                currentAssistantEl = null;

                for (const msg of data.messages) {
                    if (msg.role === 'user') {
                        appendUserMessage(msg.content);
                    } else if (msg.role === 'assistant') {
                        if (msg.content) {
                            appendAssistantText(msg.content);
                        }
                    }
                }
                loadConversationHistory();
            })
            .catch(e => console.error('Failed to load conversation:', e));
    }
}

async function deleteConversation(sessionId) {
    try {
        await fetch(`/api/conversations/${sessionId}`, { method: 'DELETE' });
        loadConversationHistory();
    } catch (e) {
        console.error('Failed to delete conversation:', e);
    }
}

function exportConversation(sessionId, format) {
    // Trigger file download via hidden link
    const url = `/api/conversations/${sessionId}/export?format=${format}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation-${sessionId.slice(0, 8)}.${format === 'markdown' ? 'md' : 'json'}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// ──── Model Management ────

async function loadModels() {
    try {
        const resp = await fetch('/api/models');
        const data = await resp.json();

        modelSelect.innerHTML = '<option value="">-- Wybierz model --</option>';
        for (const m of data.models) {
            const opt = document.createElement('option');
            opt.value = m.filename;
            opt.textContent = `${m.base_name} (${m.quantization}) - ${m.size_display}`;
            modelSelect.appendChild(opt);
        }
    } catch (e) {
        console.error('Failed to load models:', e);
    }
}

let _recommendedModelsData = [];
let _currentDlCategory = 'all';

async function loadRecommendedModels() {
    try {
        const resp = await fetch('/api/models/recommended');
        const data = await resp.json();
        _recommendedModelsData = data.models || [];
        renderRecommendedModels();
    } catch (e) {
        console.error('Failed to load recommended models:', e);
    }
}

function renderRecommendedModels() {
    const container = $('recommended-models');
    container.innerHTML = '';
    const cat = _currentDlCategory;

    const CATEGORY_LABELS = {
        coding: 'Kodowanie',
        general: 'Ogolne',
        creative: 'Kreatywne',
        powerhouse: 'Potezne',
        lightweight: 'Lekkie',
    };

    const filtered = cat === 'all'
        ? _recommendedModelsData
        : _recommendedModelsData.filter(m => m.category === cat);

    if (filtered.length === 0) {
        container.innerHTML = '<p style="color:var(--text-secondary);font-size:0.85em;text-align:center">Brak modeli w tej kategorii</p>';
        return;
    }

    for (const m of filtered) {
        const card = document.createElement('div');
        card.className = 'model-card';

        const catLabel = CATEGORY_LABELS[m.category] || m.category;
        const caps = (m.capabilities || []).map(c => `<span class="cap-tag">${escapeHtml(c)}</span>`).join('');

        card.innerHTML = `
            <div class="model-card-top">
                <div class="model-card-name">${escapeHtml(m.name)}</div>
                <span class="model-card-cat">${escapeHtml(catLabel)}</span>
            </div>
            <div class="model-card-desc">${escapeHtml(m.description)}</div>
            <div class="model-card-caps">${caps}</div>
            <div class="model-card-meta">
                <span class="model-card-size">${escapeHtml(m.size)}</span>
                ${m.downloaded
                    ? '<span class="downloaded">&#10003; Pobrany</span>'
                    : `<button class="btn btn-sm btn-primary btn-download" data-repo="${escapeHtml(m.repo)}" data-file="${escapeHtml(m.filename)}">Pobierz</button>`
                }
            </div>
            <div class="download-progress hidden" id="dl-progress-${m.filename.replace(/[^a-zA-Z0-9]/g, '_')}">
                <div class="dl-progress-bar"><div class="dl-progress-fill"></div></div>
                <div class="dl-progress-info">
                    <span class="dl-progress-pct">0%</span>
                    <span class="dl-progress-speed"></span>
                    <span class="dl-progress-eta"></span>
                </div>
            </div>
        `;

        const dlBtn = card.querySelector('.btn-download');
        if (dlBtn) {
            dlBtn.addEventListener('click', function() {
                downloadModelWithProgress(m.repo, m.filename, this, card);
            });
        }
        container.appendChild(card);
    }
}

async function downloadModelWithProgress(repo, filename, btn, card) {
    btn.disabled = true;
    btn.textContent = 'Pobieranie...';

    const safeId = filename.replace(/[^a-zA-Z0-9]/g, '_');
    const progressEl = card.querySelector(`#dl-progress-${safeId}`);
    if (progressEl) progressEl.classList.remove('hidden');

    // Create AbortController for cancel support
    const abortController = new AbortController();
    state.activeDownloads[filename] = abortController;

    // Add cancel button
    let cancelBtn = card.querySelector('.btn-cancel-download');
    if (!cancelBtn) {
        cancelBtn = document.createElement('button');
        cancelBtn.className = 'btn-cancel-download';
        cancelBtn.textContent = 'Anuluj pobieranie';
        cancelBtn.addEventListener('click', () => cancelDownload(filename));
        if (progressEl) progressEl.after(cancelBtn);
    }
    cancelBtn.classList.remove('hidden');

    const startTime = Date.now();
    let fill, pctEl, speedEl, etaEl;
    if (progressEl) {
        fill = progressEl.querySelector('.dl-progress-fill');
        pctEl = progressEl.querySelector('.dl-progress-pct');
        speedEl = progressEl.querySelector('.dl-progress-speed');
        etaEl = progressEl.querySelector('.dl-progress-eta');
    }

    try {
        const resp = await fetch('/api/models/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo, filename }),
            signal: abortController.signal,
        });

        if (!resp.ok) {
            const data = await resp.json();
            throw new Error(data.detail || 'Pobieranie nie powiodlo sie');
        }

        // Read SSE stream for real progress
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let success = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                let event;
                try { event = JSON.parse(line.slice(6)); } catch { continue; }

                if (event.type === 'progress' && progressEl) {
                    fill.style.width = event.pct + '%';
                    pctEl.textContent = Math.round(event.pct) + '%';

                    // Speed in MB/s
                    if (event.speed > 0) {
                        speedEl.textContent = event.speed.toFixed(1) + ' MB/s';
                    }

                    // ETA
                    if (event.eta > 0) {
                        const mins = Math.floor(event.eta / 60);
                        const secs = Math.floor(event.eta % 60);
                        etaEl.textContent = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
                    }

                    // Size info
                    if (event.total > 0) {
                        const dlMB = (event.downloaded / (1024*1024)).toFixed(0);
                        const totalMB = (event.total / (1024*1024)).toFixed(0);
                        btn.textContent = `${dlMB}/${totalMB} MB`;
                    }
                } else if (event.type === 'done') {
                    success = true;
                } else if (event.type === 'error') {
                    throw new Error(event.message || 'Pobieranie nie powiodlo sie');
                }
            }
        }

        if (success) {
            if (progressEl) {
                fill.style.width = '100%';
                pctEl.textContent = '100%';
                speedEl.textContent = '';
                const totalSec = Math.round((Date.now() - startTime) / 1000);
                const mins = Math.floor(totalSec / 60);
                const secs = totalSec % 60;
                etaEl.textContent = `Gotowe w ${mins}m ${secs}s`;
            }
            btn.textContent = 'Pobrano!';
            btn.className = 'downloaded';
            btn.disabled = true;
            cancelBtn.classList.add('hidden');
            loadModels();
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            if (progressEl) {
                fill.style.width = '0%';
                pctEl.textContent = 'Anulowano';
                speedEl.textContent = '';
                etaEl.textContent = '';
            }
            btn.textContent = 'Ponow';
            btn.disabled = false;
        } else {
            if (progressEl) {
                fill.style.width = '0%';
                pctEl.textContent = 'Blad';
                speedEl.textContent = '';
                etaEl.textContent = '';
            }
            btn.textContent = 'Ponow';
            btn.disabled = false;
            alert('Blad pobierania: ' + e.message);
        }
    } finally {
        delete state.activeDownloads[filename];
        cancelBtn.classList.add('hidden');
    }
}

function cancelDownload(filename) {
    const controller = state.activeDownloads[filename];
    if (controller) {
        controller.abort();
    }
}

// Legacy wrapper for backward compat
async function downloadModel(repo, filename, btn) {
    downloadModelWithProgress(repo, filename, btn, btn.closest('.model-card'));
}

async function loadModel() {
    const filename = modelSelect.value;
    if (!filename) {
        alert('Wybierz model z listy.');
        return;
    }

    // Check if engine is installed first
    if (!_engineInstalled) {
        const box = $('engine-setup-box');
        box.classList.remove('hidden');
        alert('llama-server nie zainstalowany. Kliknij "Zainstaluj llama-server" w sekcji Model lokalny.');
        return;
    }

    setStatus('loading', 'Ladowanie modelu...');
    btnLoadModel.disabled = true;

    try {
        const resp = await fetch('/api/models/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename,
                gpu_layers: parseInt($('gpu-layers').value),
                context_size: parseInt($('context-size').value),
                threads: parseInt($('cpu-threads').value),
            }),
        });

        if (resp.ok) {
            const result = await resp.json();
            state.modelLoaded = true;
            setStatus('active', filename);
            btnUnloadModel.disabled = false;
            // Load model auto-switches to local provider - refresh UI
            await loadProviders();
        } else {
            const data = await resp.json();
            setStatus('error', 'Blad ladowania');

            // Show engine setup box if it's a 503 (engine not installed)
            if (resp.status === 503) {
                _engineInstalled = false;
                const box = $('engine-setup-box');
                box.classList.remove('hidden', 'installed');
                $('engine-setup-text').textContent = data.detail || 'llama-server nie zainstalowany.';
                $('btn-install-engine').classList.remove('hidden');
                $('btn-install-engine').disabled = false;
            }

            alert('Blad: ' + (data.detail || 'Nieznany blad'));
        }
    } catch (e) {
        setStatus('error', 'Blad polaczenia');
        alert('Blad: ' + e.message);
    }

    if (_engineInstalled) {
        btnLoadModel.disabled = false;
    }
}

async function unloadModel() {
    try {
        await fetch('/api/models/unload', { method: 'POST' });
        state.modelLoaded = false;
        setStatus('inactive', 'Brak modelu');
        btnUnloadModel.disabled = true;
    } catch (e) {
        console.error('Unload error:', e);
    }
}

async function loadStatus() {
    try {
        const resp = await fetch('/api/status');
        const data = await resp.json();

        if (data.engine.running && data.engine.model) {
            state.modelLoaded = true;
            setStatus('active', data.engine.model);
            btnUnloadModel.disabled = false;
        }
    } catch (e) {
        console.error('Status check failed:', e);
    }
}

function setStatus(type, text) {
    statusDot.className = 'status-indicator';
    if (type === 'active') statusDot.classList.add('active');
    else if (type === 'loading') statusDot.classList.add('loading');
    statusText.textContent = text;
}

// ──── System Monitor ────

function startMiniMonitor() {
    // Poll system stats every 5s for mini display
    updateMiniStats();
    setInterval(updateMiniStats, 5000);
}

async function updateMiniStats() {
    try {
        const resp = await fetch('/api/monitor');
        const data = await resp.json();

        const miniCpu = $('mini-cpu');
        const miniRam = $('mini-ram');
        const miniGpu = $('mini-gpu');

        if (data.cpu && data.cpu.percent !== undefined) {
            miniCpu.textContent = `CPU ${data.cpu.percent}%`;
        }
        if (data.ram && data.ram.percent !== undefined) {
            miniRam.textContent = `RAM ${data.ram.percent}%`;
        }
        if (data.gpu && data.gpu.gpu_use_percent !== undefined && data.gpu.gpu_use_percent !== 'N/A') {
            miniGpu.textContent = `GPU ${data.gpu.gpu_use_percent}%`;
        } else {
            miniGpu.textContent = 'GPU --';
        }

        // Update monitor panel if open
        if (state.monitorOpen) {
            updateMonitorPanel(data);
        }
    } catch (e) {
        // Silent fail for mini monitor
    }
}

function updateMonitorPanel(data) {
    if (data.cpu) {
        $('bar-cpu').style.width = (data.cpu.percent || 0) + '%';
        $('val-cpu').textContent = `${data.cpu.percent || 0}% (${data.cpu.count_logical || '?'} threads)`;
    }
    if (data.ram) {
        $('bar-ram').style.width = (data.ram.percent || 0) + '%';
        $('val-ram').textContent = `${data.ram.used_gb || 0}/${data.ram.total_gb || 0} GB (${data.ram.percent || 0}%)`;
    }
    if (data.disk) {
        $('bar-disk').style.width = (data.disk.percent || 0) + '%';
        $('val-disk').textContent = `${data.disk.used_gb || 0}/${data.disk.total_gb || 0} GB`;
    }
    if (data.gpu) {
        const gpuPct = data.gpu.gpu_use_percent;
        if (gpuPct !== undefined && gpuPct !== 'N/A') {
            $('bar-gpu').style.width = gpuPct + '%';
            const temp = data.gpu.temperature_c !== 'N/A' ? ` ${data.gpu.temperature_c}°C` : '';
            $('val-gpu').textContent = `${data.gpu.vendor || 'GPU'} ${gpuPct}%${temp}`;
        } else {
            $('val-gpu').textContent = data.gpu.message || 'Brak danych GPU';
        }
    }
}

function toggleMonitor() {
    const panel = $('monitor-panel');
    state.monitorOpen = !state.monitorOpen;
    panel.classList.toggle('hidden');
    if (state.monitorOpen) {
        updateMiniStats(); // Immediate refresh
    }
}

// ──── Prompt Templates ────

async function loadTemplates() {
    try {
        const resp = await fetch('/api/templates');
        const data = await resp.json();
        renderTemplates(data.templates, 'all');

        // Category filter buttons
        document.querySelectorAll('.template-cat').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.template-cat').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                renderTemplates(data.templates, btn.dataset.cat);
            });
        });
    } catch (e) {
        console.error('Failed to load templates:', e);
    }
}

function renderTemplates(templates, category) {
    const grid = $('templates-grid');
    grid.innerHTML = '';

    const filtered = category === 'all'
        ? templates
        : templates.filter(t => t.category === category);

    for (const t of filtered) {
        const card = document.createElement('div');
        card.className = 'template-card';
        card.innerHTML = `
            <span class="template-icon">${t.icon || '&#9889;'}</span>
            <span class="template-name">${escapeHtml(t.name)}</span>
        `;
        card.onclick = () => useTemplate(t);
        grid.appendChild(card);
    }
}

function useTemplate(template) {
    let prompt = template.prompt;

    // If template has variables, ask the user for values
    if (template.variables && template.variables.length > 0) {
        for (const v of template.variables) {
            const value = window.prompt(`Podaj wartosc dla "${v}":`);
            if (value === null) return; // Cancelled
            prompt = prompt.replace(`{${v}}`, value);
        }
    }

    chatInput.value = prompt;
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + 'px';
    chatInput.focus();

    // Hide templates panel
    $('templates-panel').classList.add('hidden');
}

// ──── Documents (RAG) ────

async function loadDocumentsList() {
    try {
        const resp = await fetch('/api/documents');
        const data = await resp.json();
        const container = $('documents-list');

        if (!data.documents || data.documents.length === 0) {
            container.innerHTML = '<div class="doc-empty">Brak zaindeksowanych dokumentow</div>';
            return;
        }

        container.innerHTML = '';
        for (const doc of data.documents) {
            const el = document.createElement('div');
            el.className = 'doc-item';

            const sizeKb = Math.round(doc.content_length / 1024);
            el.innerHTML = `
                <div class="doc-info">
                    <span class="doc-name">${escapeHtml(doc.filename)}</span>
                    <span class="doc-meta">${sizeKb} KB, ${doc.chunk_count} fragmentow</span>
                </div>
                <button class="btn-doc-remove" title="Usun">&times;</button>
            `;
            el.querySelector('.btn-doc-remove').addEventListener('click', () => removeDocument(doc.doc_id));
            container.appendChild(el);
        }
    } catch (e) {
        console.error('Failed to load documents:', e);
    }
}

async function uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch('/api/documents/upload', {
            method: 'POST',
            body: formData,
        });

        if (resp.ok) {
            loadDocumentsList();
        } else {
            const data = await resp.json();
            alert('Blad: ' + (data.detail || 'Upload failed'));
        }
    } catch (e) {
        alert('Blad uploadu: ' + e.message);
    }
}

async function removeDocument(docId) {
    try {
        await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
        loadDocumentsList();
    } catch (e) {
        console.error('Failed to remove document:', e);
    }
}

// ──── File Upload (Chat Attachment) ────

async function uploadFileForChat(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        if (resp.ok) {
            const data = await resp.json();
            state.attachedFile = { name: data.filename, path: data.path };
            $('attached-file-name').textContent = `&#128206; ${data.filename}`;
            $('file-attachment').classList.remove('hidden');
        } else {
            alert('Blad uploadu pliku');
        }
    } catch (e) {
        alert('Blad: ' + e.message);
    }
}

function removeAttachment() {
    state.attachedFile = null;
    $('file-attachment').classList.add('hidden');
    $('attached-file-name').textContent = '';
}

// ──── Engine Setup ────

let _engineInstalled = false;

async function checkEngineSetup() {
    try {
        const resp = await fetch('/api/setup/engine-status');
        const data = await resp.json();
        _engineInstalled = data.installed;

        const box = $('engine-setup-box');
        if (!data.installed) {
            box.classList.remove('hidden', 'installed');
            $('engine-setup-text').textContent = 'llama-server nie zainstalowany. Zainstaluj silnik aby ladowac modele lokalnie.';
            $('btn-install-engine').classList.remove('hidden');
            $('btn-install-engine').disabled = false;
            $('engine-install-progress').classList.add('hidden');
            // Disable load button when engine not installed
            btnLoadModel.disabled = true;
            btnLoadModel.title = 'Najpierw zainstaluj llama-server';
        } else {
            box.classList.add('hidden');
            btnLoadModel.disabled = false;
            btnLoadModel.title = '';
        }
    } catch (e) {
        console.error('Engine status check failed:', e);
    }
}

async function installEngine() {
    const btn = $('btn-install-engine');
    const progressEl = $('engine-install-progress');
    const fill = $('engine-install-fill');
    const statusEl = $('engine-install-status');

    btn.disabled = true;
    btn.textContent = 'Instalowanie...';
    progressEl.classList.remove('hidden');

    const startTime = Date.now();
    let fakePct = 0;
    const pollInterval = setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000;
        // Asymptotic progress to 90% over ~120s
        fakePct = 90 * (1 - Math.exp(-elapsed / 40));
        fill.style.width = Math.round(fakePct) + '%';
        const mins = Math.floor(elapsed / 60);
        const secs = Math.floor(elapsed % 60);
        statusEl.textContent = `Pobieranie i instalacja... ${Math.round(fakePct)}% (${mins}m ${secs}s)`;
    }, 500);

    try {
        const resp = await fetch('/api/setup/install-engine', { method: 'POST' });
        clearInterval(pollInterval);

        if (resp.ok) {
            const data = await resp.json();
            fill.style.width = '100%';
            statusEl.textContent = 'Zainstalowano pomyslnie!';

            const box = $('engine-setup-box');
            box.classList.add('installed');
            $('engine-setup-text').textContent = 'llama-server zainstalowany! Mozesz teraz ladowac modele.';
            btn.classList.add('hidden');

            _engineInstalled = true;
            btnLoadModel.disabled = false;
            btnLoadModel.title = '';

            // Hide after 3s
            setTimeout(() => {
                box.classList.add('hidden');
            }, 3000);
        } else {
            const data = await resp.json();
            fill.style.width = '0%';
            statusEl.textContent = 'Blad: ' + (data.detail || 'Instalacja nie powiodla sie');
            btn.disabled = false;
            btn.textContent = 'Sprobuj ponownie';
        }
    } catch (e) {
        clearInterval(pollInterval);
        fill.style.width = '0%';
        statusEl.textContent = 'Blad polaczenia: ' + e.message;
        btn.disabled = false;
        btn.textContent = 'Sprobuj ponownie';
    }
}

// ──── Providers ────

let _providersData = null;

async function loadProviders() {
    try {
        const resp = await fetch('/api/providers');
        _providersData = await resp.json();

        const select = $('provider-select');
        select.innerHTML = '';
        for (const p of _providersData.providers) {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.name;
            if (p.id === _providersData.active_provider) opt.selected = true;
            select.appendChild(opt);
        }

        updateProviderUI(_providersData);
        updateActiveProviderCard(_providersData);
        renderProviderKeysGrid(_providersData);
        renderApiKeysForms(_providersData);
        renderFallbackChain(_providersData);
        updateProviderBar(_providersData);
    } catch (e) {
        console.error('Failed to load providers:', e);
    }
}

function updateProviderUI(data) {
    const selectedId = $('provider-select').value;
    const provider = data.providers.find(p => p.id === selectedId);
    if (!provider) return;

    const modelRow = $('provider-model-row');

    // Show model selector for non-local providers
    if (selectedId !== 'local' && provider.models && provider.models.length > 0) {
        modelRow.classList.remove('hidden');
        const modelSel = $('provider-model-select');
        modelSel.innerHTML = '';
        for (const m of provider.models) {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = m.name + (m.context ? ` (${m.context})` : '');
            if (m.id === provider.active_model) opt.selected = true;
            modelSel.appendChild(opt);
        }
    } else {
        modelRow.classList.add('hidden');
    }
}

function updateActiveProviderCard(data) {
    const activeId = data.active_provider || 'local';
    const provider = data.providers.find(p => p.id === activeId);
    if (!provider) return;

    const isCloud = activeId !== 'local';
    $('apc-indicator').className = 'provider-bar-indicator ' + (isCloud ? 'cloud' : 'local');
    $('apc-name').textContent = provider.name;
    $('apc-model').textContent = provider.active_model || (isCloud ? '' : '(zaladuj model ponizej)');

    const deactBtn = $('btn-deactivate-provider');
    deactBtn.disabled = !isCloud;
}

async function activateProvider() {
    const providerId = $('provider-select').value;
    const model = $('provider-model-select')?.value || null;

    try {
        const resp = await fetch('/api/providers/activate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider_id: providerId, model }),
        });
        if (resp.ok) {
            await loadProviders();
        } else {
            const err = await resp.json();
            alert(err.detail || 'Blad aktywacji providera');
        }
    } catch (e) {
        alert('Blad polaczenia');
    }
}

async function deactivateProvider() {
    // Switch back to local
    try {
        const resp = await fetch('/api/providers/activate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider_id: 'local' }),
        });
        if (resp.ok) {
            await loadProviders();
        }
    } catch (e) {
        alert('Blad polaczenia');
    }
}

// ──── Fallback Chain & Smart Routing ────

function renderFallbackChain(data) {
    const list = $('fallback-chain-list');
    const chain = data.fallback_chain || [];
    list.innerHTML = '';

    if (chain.length === 0) {
        list.innerHTML = '<small style="color:var(--text-muted)">Brak - uzywa tylko aktywnego providera</small>';
    } else {
        chain.forEach((pid, i) => {
            const provider = data.providers.find(p => p.id === pid);
            const name = provider ? provider.name : pid;
            const chip = document.createElement('span');
            chip.className = 'fallback-chip';
            chip.innerHTML = `<span class="fallback-order">${i + 1}.</span> ${name} <span class="fallback-remove" data-pid="${pid}">&times;</span>`;
            list.appendChild(chip);
        });
    }

    // Populate add-select with providers not yet in chain
    const addSelect = $('fallback-add-select');
    addSelect.innerHTML = '<option value="">+ Dodaj provider do lancucha</option>';
    for (const p of data.providers) {
        if (p.id === 'local') continue;  // local is always implicit fallback
        if (chain.includes(p.id)) continue;
        if (!p.has_api_key && p.requires_api_key) continue;
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = p.name;
        addSelect.appendChild(opt);
    }

    // Smart routing toggle
    const toggle = $('smart-routing-toggle');
    toggle.checked = data.smart_routing || false;
}

async function addToFallbackChain(providerId) {
    if (!providerId || !_providersData) return;
    const chain = (_providersData.fallback_chain || []).concat(providerId);
    try {
        const resp = await fetch('/api/providers/fallback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chain }),
        });
        if (resp.ok) await loadProviders();
    } catch (e) {
        console.error('Failed to update fallback chain:', e);
    }
}

async function removeFromFallbackChain(providerId) {
    if (!_providersData) return;
    const chain = (_providersData.fallback_chain || []).filter(id => id !== providerId);
    try {
        const resp = await fetch('/api/providers/fallback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chain }),
        });
        if (resp.ok) await loadProviders();
    } catch (e) {
        console.error('Failed to update fallback chain:', e);
    }
}

async function toggleSmartRouting(enabled) {
    try {
        await fetch('/api/providers/smart-routing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled }),
        });
    } catch (e) {
        console.error('Failed to toggle smart routing:', e);
    }
}

// ──── API Keys Panel (dedykowana zakladka) ────

function renderApiKeysForms(data) {
    const container = $('api-keys-forms');
    container.innerHTML = '';

    const API_HINTS = {
        openai: 'Klucz zaczyna sie od sk-...',
        anthropic: 'Klucz zaczyna sie od sk-ant-...',
        google: 'Klucz zaczyna sie od AIza...',
        openrouter: 'Klucz zaczyna sie od sk-or-...',
    };

    for (const p of data.providers) {
        if (!p.requires_api_key) continue;

        const row = document.createElement('div');
        row.className = 'api-key-form-row' + (p.has_api_key ? ' configured' : '');

        row.innerHTML = `
            <div class="akf-header">
                <span class="akf-name">${escapeHtml(p.name)}</span>
                ${p.has_api_key
                    ? '<span class="akf-badge ok">Aktywny</span>'
                    : '<span class="akf-badge missing">Brak klucza</span>'
                }
            </div>
            <div class="akf-input-row">
                <input type="password" class="input-sm akf-input" placeholder="${escapeHtml(API_HINTS[p.id] || 'Klucz API...')}" data-provider="${p.id}">
                <button class="btn btn-sm btn-success akf-save" data-provider="${p.id}">Zapisz</button>
                ${p.has_api_key ? `<button class="btn btn-sm btn-danger akf-delete" data-provider="${p.id}" title="Usun klucz">X</button>` : ''}
            </div>
        `;
        container.appendChild(row);
    }

    // Bind events
    container.querySelectorAll('.akf-save').forEach(btn => {
        btn.addEventListener('click', async () => {
            const pid = btn.dataset.provider;
            const input = container.querySelector(`.akf-input[data-provider="${pid}"]`);
            const key = input.value.trim();
            if (!key) return;
            await saveApiKey(pid, key);
            input.value = '';
        });
    });

    container.querySelectorAll('.akf-delete').forEach(btn => {
        btn.addEventListener('click', async () => {
            const pid = btn.dataset.provider;
            if (confirm(`Usunac klucz API dla ${pid}?`)) {
                await deleteApiKey(pid);
            }
        });
    });

    container.querySelectorAll('.akf-input').forEach(input => {
        input.addEventListener('keydown', async (e) => {
            if (e.key === 'Enter') {
                const pid = input.dataset.provider;
                const key = input.value.trim();
                if (!key) return;
                await saveApiKey(pid, key);
                input.value = '';
            }
        });
    });
}

async function saveApiKey(providerId, apiKey) {
    try {
        const resp = await fetch('/api/providers/key', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider_id: providerId, api_key: apiKey }),
        });
        if (resp.ok) {
            await loadProviders();
        } else {
            alert('Blad zapisu klucza');
        }
    } catch (e) {
        alert('Blad polaczenia');
    }
}

async function deleteApiKey(providerId) {
    try {
        const resp = await fetch(`/api/providers/key/${providerId}`, { method: 'DELETE' });
        if (resp.ok) {
            await loadProviders();
        } else {
            alert('Blad usuwania klucza');
        }
    } catch (e) {
        alert('Blad polaczenia');
    }
}

// ──── Provider Status Bar ────

function updateProviderBar(data) {
    const activeId = data.active_provider || 'local';
    const provider = data.providers.find(p => p.id === activeId);
    if (!provider) return;

    const isCloud = activeId !== 'local';
    const indicator = $('provider-bar-indicator');
    indicator.className = 'provider-bar-indicator ' + (isCloud ? 'cloud' : 'local');

    $('provider-bar-name').textContent = provider.name;
    $('provider-bar-model').textContent = provider.active_model || '';

    // Cost display only for cloud providers
    const costEl = $('provider-bar-cost');
    if (isCloud) {
        costEl.classList.remove('hidden');
    } else {
        costEl.classList.add('hidden');
    }

    // Update local model section - show hint when cloud provider is active
    const loadBtn = $('btn-load-model');
    const unloadBtn = $('btn-unload-model');
    if (isCloud) {
        loadBtn.title = 'Zaladowanie modelu GGUF przelacza na lokalny provider';
    } else {
        loadBtn.title = '';
    }
}

function updateProviderBarUsage(sessionTokens, sessionCost) {
    $('provider-bar-tokens').textContent = formatTokenCount(sessionTokens) + ' tok';
    const costEl = $('provider-bar-cost');
    if (sessionCost > 0) {
        costEl.textContent = '$' + sessionCost.toFixed(4);
        costEl.classList.remove('hidden');
    }
}

function formatTokenCount(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return String(n);
}

// ──── Provider Keys Grid ────

function renderProviderKeysGrid(data) {
    const grid = $('provider-keys-grid');
    grid.innerHTML = '';
    const activeId = data.active_provider || 'local';

    for (const p of data.providers) {
        // Skip local - always available
        if (p.id === 'local') continue;

        const chip = document.createElement('span');
        const configured = !p.requires_api_key || p.has_api_key;
        let cls = 'provider-key-chip';
        if (configured) cls += ' configured';
        if (p.id === activeId) cls += ' active-chip';
        chip.className = cls;
        chip.innerHTML = `<span class="chip-dot"></span>${escapeHtml(p.id)}`;
        chip.title = p.name + (configured ? ' (skonfigurowany)' : ' (brak klucza)');
        grid.appendChild(chip);
    }
}

// ──── Usage Panel ────

async function renderUsagePanel() {
    const content = $('usage-panel-content');
    try {
        const resp = await fetch('/api/usage');
        const data = await resp.json();

        if (!data.providers || data.providers.length === 0) {
            content.innerHTML = '<p class="usage-empty">Brak danych o zuzyciu. Wyslij wiadomosc aby zaczac sledzenie.</p>';
            return;
        }

        let html = `<div class="usage-stat-row" style="margin-bottom:10px;font-weight:600">
            <span>Sesja razem</span>
            <span class="usage-stat-value">${formatTokenCount(data.session_tokens)} tok / $${data.session_cost_usd.toFixed(4)}</span>
        </div>`;

        for (const pu of data.providers) {
            const isActive = pu.provider_id === data.active_provider;
            html += `<div class="usage-provider-card${isActive ? ' active-card' : ''}">
                <div class="usage-provider-header">
                    <span class="usage-provider-name">${escapeHtml(pu.provider_name || pu.provider_id)}</span>
                    ${isActive ? '<span class="usage-provider-active-badge">AKTYWNY</span>' : ''}
                </div>
                <div class="usage-stat-row">
                    <span>Zapytan</span>
                    <span class="usage-stat-value">${pu.requests}</span>
                </div>
                <div class="usage-stat-row">
                    <span>Tokeny (in / out)</span>
                    <span class="usage-stat-value">${formatTokenCount(pu.input_tokens)} / ${formatTokenCount(pu.output_tokens)}</span>
                </div>
                <div class="usage-stat-row">
                    <span>Razem tokenow</span>
                    <span class="usage-stat-value">${formatTokenCount(pu.total_tokens)}</span>
                </div>
                <div class="usage-stat-row">
                    <span>Koszt (szacunek)</span>
                    <span class="usage-stat-value">$${pu.estimated_cost_usd.toFixed(4)}</span>
                </div>
            </div>`;
        }

        content.innerHTML = html;
    } catch (e) {
        content.innerHTML = '<p class="usage-empty">Blad ladowania danych zuzycia</p>';
    }
}

async function resetUsage() {
    try {
        await fetch('/api/usage/reset', { method: 'POST' });
        $('provider-bar-tokens').textContent = '0 tok';
        $('provider-bar-cost').textContent = '$0.00';
        await renderUsagePanel();
    } catch (e) {
        console.error('Failed to reset usage:', e);
    }
}

// ──── Semantic Router ────

async function loadRouterConfig() {
    try {
        const resp = await fetch('/api/router');
        const data = await resp.json();

        $('router-enabled').checked = data.enabled;

        // Populate model selects for each task type
        const taskTypes = ['coding', 'analysis', 'creative', 'chat'];
        for (const tt of taskTypes) {
            const sel = $(`router-${tt}`);
            sel.innerHTML = '<option value="">-- domyslny model --</option>';
            for (const m of data.available_models) {
                const opt = document.createElement('option');
                opt.value = m.filename;
                opt.textContent = `${m.base_name} (${m.quantization})`;
                if (data.assignments[tt] === m.filename) {
                    opt.selected = true;
                }
                sel.appendChild(opt);
            }
        }
    } catch (e) {
        console.error('Failed to load router config:', e);
    }
}

async function saveRouterConfig() {
    const enabled = $('router-enabled').checked;
    const assignments = {
        coding: $('router-coding').value || null,
        analysis: $('router-analysis').value || null,
        creative: $('router-creative').value || null,
        chat: $('router-chat').value || null,
    };

    try {
        await fetch('/api/router', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled, assignments }),
        });
    } catch (e) {
        console.error('Failed to save router config:', e);
    }
}

async function testRouter() {
    const msg = $('router-test-input').value.trim();
    if (!msg) return;

    const resultDiv = $('router-test-result');
    resultDiv.classList.remove('hidden');
    resultDiv.innerHTML = 'Analizuję...';

    try {
        const resp = await fetch('/api/router/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg }),
        });
        const data = await resp.json();

        const maxScore = Math.max(...Object.values(data.scores), 0.01);
        let html = '';
        for (const [type, score] of Object.entries(data.scores)) {
            const pct = Math.round((score / maxScore) * 100);
            const isWinner = type === data.detected_type;
            html += `
                <div class="router-score-bar">
                    <span class="router-score-label">${type}</span>
                    <div class="router-score-track">
                        <div class="router-score-fill${isWinner ? ' winner' : ''}" style="width:${pct}%"></div>
                    </div>
                    <span class="router-score-value">${(score * 100).toFixed(1)}%</span>
                </div>`;
        }
        html += `<div class="router-detected">→ ${data.detected_type}</div>`;
        resultDiv.innerHTML = html;
    } catch (e) {
        resultDiv.innerHTML = 'Blad testu';
    }
}

// ──── Event Listeners ────

function setupEventListeners() {
    // Send message
    btnSend.addEventListener('click', () => {
        sendMessage(chatInput.value);
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!state.sending) sendMessage(chatInput.value);
        }
    });

    // Auto-resize textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + 'px';
    });

    // New chat
    btnNewChat.addEventListener('click', () => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({ type: 'clear' }));
        }
        clearChat();
    });

    // Model controls
    btnLoadModel.addEventListener('click', loadModel);
    btnUnloadModel.addEventListener('click', unloadModel);

    // Engine install
    $('btn-install-engine').addEventListener('click', installEngine);

    // Temperature slider
    tempSlider.addEventListener('input', () => {
        tempValue.textContent = tempSlider.value;
    });

    // Sidebar toggle (mobile)
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 &&
            sidebar.classList.contains('open') &&
            !sidebar.contains(e.target) &&
            e.target !== sidebarToggle) {
            sidebar.classList.remove('open');
        }
    });

    // Collapsible sections
    document.querySelectorAll('.section-toggle').forEach(toggle => {
        toggle.addEventListener('click', () => {
            const targetId = toggle.dataset.target;
            const content = document.getElementById(targetId);
            if (content) {
                content.classList.toggle('hidden');
                toggle.classList.toggle('collapsed');
            }
        });
    });

    // Templates toggle
    $('btn-templates').addEventListener('click', () => {
        $('templates-panel').classList.toggle('hidden');
        $('monitor-panel').classList.add('hidden');
        state.monitorOpen = false;
    });

    // Monitor toggle
    $('btn-monitor').addEventListener('click', toggleMonitor);

    // File upload button (top bar)
    $('btn-upload-file').addEventListener('click', () => {
        $('chat-file-input').click();
    });

    $('chat-file-input').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadFileForChat(e.target.files[0]);
            e.target.value = '';
        }
    });

    // Provider controls
    $('provider-select').addEventListener('change', () => {
        if (_providersData) updateProviderUI(_providersData);
    });
    $('btn-activate-provider').addEventListener('click', activateProvider);
    $('btn-deactivate-provider').addEventListener('click', deactivateProvider);

    // Fallback chain
    $('fallback-add-select').addEventListener('change', (e) => {
        if (e.target.value) {
            addToFallbackChain(e.target.value);
            e.target.value = '';
        }
    });
    $('fallback-chain-list').addEventListener('click', (e) => {
        const removeBtn = e.target.closest('.fallback-remove');
        if (removeBtn) removeFromFallbackChain(removeBtn.dataset.pid);
    });

    // Smart routing
    $('smart-routing-toggle').addEventListener('change', (e) => {
        toggleSmartRouting(e.target.checked);
    });

    // Download category filters
    document.querySelectorAll('.dl-cat').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.dl-cat').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            _currentDlCategory = btn.dataset.cat;
            renderRecommendedModels();
        });
    });

    // Usage panel controls
    $('btn-usage-panel').addEventListener('click', () => {
        const panel = $('usage-panel');
        panel.classList.toggle('hidden');
        if (!panel.classList.contains('hidden')) renderUsagePanel();
    });
    $('btn-close-usage').addEventListener('click', () => {
        $('usage-panel').classList.add('hidden');
    });
    $('btn-reset-usage').addEventListener('click', resetUsage);

    // Router controls
    $('router-enabled').addEventListener('change', saveRouterConfig);
    document.querySelectorAll('.router-select').forEach(sel => {
        sel.addEventListener('change', saveRouterConfig);
    });
    $('btn-router-test').addEventListener('click', testRouter);
    $('router-test-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') testRouter();
    });

    // Document upload (RAG)
    $('btn-upload-doc').addEventListener('click', () => {
        $('doc-file-input').click();
    });

    $('doc-file-input').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadDocument(e.target.files[0]);
            e.target.value = '';
        }
    });

    // Multi-Agent
    $('btn-run-multiagent').addEventListener('click', runMultiAgent);
}

// ──── Utilities ────

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function renderMarkdown(text) {
    // Simple markdown rendering with syntax highlighting
    let html = escapeHtml(text);

    // Code blocks ```lang...``` with syntax highlighting + copy button
    let blockId = 0;
    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
        const trimmed = code.trim();
        let highlighted;
        if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
            try {
                highlighted = hljs.highlight(trimmed, { language: lang }).value;
            } catch {
                highlighted = trimmed;
            }
        } else if (typeof hljs !== 'undefined') {
            try {
                highlighted = hljs.highlightAuto(trimmed).value;
            } catch {
                highlighted = trimmed;
            }
        } else {
            highlighted = trimmed;
        }
        const bid = `codeblock-${Date.now()}-${blockId++}`;
        const langLabel = lang ? `<span class="code-lang-label">${lang}</span>` : '';
        const copyBtn = `<button class="btn-copy-code" data-code-id="${bid}" onclick="copyCodeBlock(this)" title="Kopiuj kod">Kopiuj</button>`;
        return `<div class="code-block-wrapper"><div class="code-block-header">${langLabel}${copyBtn}</div><pre><code class="hljs lang-${lang}" id="${bid}">${highlighted}</code></pre></div>`;
    });

    // Inline code `...`
    html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

    // Bold **...**
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic *...*
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Links [text](url) - only allow http/https URLs to prevent javascript: XSS
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, url) => {
        if (/^https?:\/\//i.test(url)) {
            return `<a href="${url}" target="_blank" rel="noopener">${text}</a>`;
        }
        return `${text} (${url})`;
    });

    // Line breaks
    html = html.replace(/\n/g, '<br>');

    return html;
}

function copyCodeBlock(btn) {
    const codeId = btn.dataset.codeId;
    const codeEl = document.getElementById(codeId);
    if (!codeEl) return;

    const text = codeEl.textContent;
    navigator.clipboard.writeText(text).then(() => {
        btn.textContent = 'Skopiowano!';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = 'Kopiuj';
            btn.classList.remove('copied');
        }, 2000);
    }).catch(() => {
        // Fallback for older browsers
        const area = document.createElement('textarea');
        area.value = text;
        area.style.position = 'fixed';
        area.style.left = '-9999px';
        document.body.appendChild(area);
        area.select();
        document.execCommand('copy');
        document.body.removeChild(area);
        btn.textContent = 'Skopiowano!';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = 'Kopiuj';
            btn.classList.remove('copied');
        }, 2000);
    });
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

// ──── Multi-Agent ────

const ROLE_COLORS = {
    planner: '#6c5ce7',
    coder: '#00b894',
    reviewer: '#fdcb6e',
    researcher: '#74b9ff',
};

let _maWorkflowActive = false;

let _agentRolesData = [];

async function loadAgentRoles() {
    try {
        const resp = await fetch('/api/agents/roles');
        const data = await resp.json();
        _agentRolesData = data.roles || [];
        renderAgentRoles();
    } catch (e) {
        console.error('Failed to load agent roles:', e);
    }
}

function renderAgentRoles() {
    const container = $('ma-roles-list');
    container.innerHTML = '';
    container.className = 'ma-role-config';

    // Build model options from available local models + providers
    const modelOptions = buildModelOptionsList();

    for (const role of _agentRolesData) {
        const row = document.createElement('div');
        row.className = 'ma-role-row';

        let optionsHtml = '<option value="">-- auto --</option>';
        for (const mo of modelOptions) {
            const selected = (role.preferred_provider && role.preferred_model)
                ? (mo.value === `${role.preferred_provider}:${role.preferred_model}` ? ' selected' : '')
                : '';
            optionsHtml += `<option value="${escapeHtml(mo.value)}"${selected}>${escapeHtml(mo.label)}</option>`;
        }

        row.innerHTML = `
            <span class="ma-role-dot" style="background:${role.color}"></span>
            <span class="ma-role-name" title="${escapeHtml(role.description)}">${escapeHtml(role.name)}</span>
            <select class="ma-role-select" data-role="${escapeHtml(role.role_id)}">
                ${optionsHtml}
            </select>
        `;
        container.appendChild(row);
    }

    // Bind change events for role model selection
    container.querySelectorAll('.ma-role-select').forEach(sel => {
        sel.addEventListener('change', () => {
            const roleId = sel.dataset.role;
            const val = sel.value;
            updateRoleModel(roleId, val);
        });
    });
}

function buildModelOptionsList() {
    const options = [];

    // Local models from the model-select dropdown
    const modelSel = $('model-select');
    if (modelSel) {
        for (const opt of modelSel.options) {
            if (opt.value) {
                options.push({
                    value: `local:${opt.value}`,
                    label: `[Lokalny] ${opt.textContent}`,
                });
            }
        }
    }

    // Cloud provider models
    if (_providersData && _providersData.providers) {
        for (const p of _providersData.providers) {
            if (p.id === 'local') continue;
            if (!p.has_api_key && p.requires_api_key) continue;
            if (p.models && p.models.length > 0) {
                for (const m of p.models) {
                    options.push({
                        value: `${p.id}:${m.id}`,
                        label: `[${p.name}] ${m.name}`,
                    });
                }
            }
        }
    }

    return options;
}

async function updateRoleModel(roleId, value) {
    if (!value) {
        // Reset to auto
        await fetch('/api/agents/roles/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ role_id: roleId, provider: null, model: null }),
        }).catch(e => console.error('Failed to update role:', e));
        return;
    }

    const [provider, ...modelParts] = value.split(':');
    const model = modelParts.join(':');

    try {
        await fetch('/api/agents/roles/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ role_id: roleId, provider, model }),
        });
    } catch (e) {
        console.error('Failed to update role model:', e);
    }
}

async function runMultiAgent() {
    const taskInput = $('ma-task-input');
    const task = taskInput.value.trim();
    if (!task || _maWorkflowActive) return;

    _maWorkflowActive = true;
    const btn = $('btn-run-multiagent');
    btn.disabled = true;
    btn.textContent = 'Pracuje...';

    const statusEl = $('ma-workflow-status');
    statusEl.classList.remove('hidden');
    $('ma-progress-fill').style.width = '0%';
    $('ma-progress-text').textContent = 'Planowanie...';

    // Show workflow panel in main area
    showWorkflowPanel(task);

    try {
        const resp = await fetch('/api/agents/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task, session_id: state.sessionId }),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const event = JSON.parse(line.slice(6));
                        handleMultiAgentEvent(event);
                    } catch (e) {
                        // ignore parse errors
                    }
                }
            }
        }
    } catch (e) {
        console.error('Multi-agent error:', e);
        $('ma-progress-text').textContent = 'Blad: ' + e.message;
    }

    _maWorkflowActive = false;
    btn.disabled = false;
    btn.textContent = '\u2699 Uruchom zespol agentow';
}

function handleMultiAgentEvent(event) {
    switch (event.type) {
        case 'workflow_start':
            $('ma-progress-text').textContent = 'Workflow rozpoczety...';
            break;

        case 'planning':
            $('ma-progress-text').textContent = event.status;
            $('ma-progress-fill').style.width = '5%';
            break;

        case 'plan_ready':
            renderWorkflowSubtasks(event.subtasks);
            $('ma-progress-text').textContent = `Plan gotowy: ${event.subtasks.length} podzadan`;
            $('ma-progress-fill').style.width = '10%';
            break;

        case 'subtask_start':
            updateWorkflowSubtask(event.subtask.id, 'running', null);
            $('ma-progress-text').textContent = `[${event.agent}] ${event.subtask.title}...`;
            break;

        case 'subtask_complete':
            updateWorkflowSubtask(event.subtask_id, 'completed', event.result);
            break;

        case 'subtask_error':
            updateWorkflowSubtask(event.subtask_id, 'failed', null, event.error);
            break;

        case 'progress_update':
            if (event.progress) {
                const pct = 10 + (event.progress.percent * 0.8); // 10-90% range for subtasks
                $('ma-progress-fill').style.width = pct + '%';
                $('ma-progress-text').textContent = `${event.progress.completed}/${event.progress.total} podzadan`;

                // Update workflow progress summary
                updateWorkflowProgress(event.progress);
            }
            break;

        case 'synthesis':
            $('ma-progress-text').textContent = event.status;
            $('ma-progress-fill').style.width = '95%';
            break;

        case 'workflow_complete':
            $('ma-progress-fill').style.width = '100%';
            $('ma-progress-text').textContent = 'Gotowe!';
            showWorkflowResult(event.result);

            // Also inject result as assistant message in chat
            appendWorkflowMessage(event.result);

            setTimeout(() => {
                $('ma-workflow-status').classList.add('hidden');
            }, 3000);
            break;

        case 'workflow_error':
            $('ma-progress-text').textContent = 'Blad: ' + event.message;
            $('ma-progress-fill').style.width = '0%';
            break;
    }
}

// ──── Workflow Visualization ────

function showWorkflowPanel(task) {
    const panel = $('workflow-panel');
    panel.classList.remove('hidden');
    $('workflow-task').textContent = 'Zadanie: ' + task;
    $('workflow-subtasks').innerHTML = '<div style="color:var(--text-muted);text-align:center;padding:20px">Planner analizuje zadanie...</div>';
    $('workflow-result').classList.add('hidden');
    $('workflow-progress').innerHTML = '';
}

function toggleWorkflowPanel() {
    $('workflow-panel').classList.toggle('hidden');
}

function renderWorkflowSubtasks(subtasks) {
    const container = $('workflow-subtasks');
    container.innerHTML = '';

    for (const st of subtasks) {
        const el = document.createElement('div');
        el.className = 'workflow-subtask';
        el.id = `ws-subtask-${st.id}`;

        const color = ROLE_COLORS[st.role] || '#6c5ce7';
        el.innerHTML = `
            <div class="ws-status-icon pending" id="ws-icon-${st.id}">\u23F3</div>
            <div class="ws-details">
                <div class="ws-title">
                    ${escapeHtml(st.title)}
                    <span class="ws-role-badge" style="background:${color}22;color:${color}">${st.role}</span>
                </div>
                <div class="ws-description">${escapeHtml(st.description || '')}</div>
                <div id="ws-result-${st.id}"></div>
            </div>
        `;
        container.appendChild(el);
    }
}

function updateWorkflowSubtask(id, status, result, error) {
    const icon = $(`ws-icon-${id}`);
    if (icon) {
        icon.className = `ws-status-icon ${status}`;
        const icons = { pending: '\u23F3', running: '\u2699', completed: '\u2714', failed: '\u2718' };
        icon.textContent = icons[status] || '\u23F3';
    }

    const resultEl = $(`ws-result-${id}`);
    if (resultEl) {
        if (status === 'completed' && result) {
            resultEl.innerHTML = `<div class="ws-result">${escapeHtml(result)}</div>`;
        } else if (status === 'failed' && error) {
            resultEl.innerHTML = `<div class="ws-error">\u26A0 ${escapeHtml(error)}</div>`;
        }
    }
}

function updateWorkflowProgress(progress) {
    const el = $('workflow-progress');
    el.innerHTML = `
        <span class="wp-stat">\u2714 ${progress.completed} gotowe</span>
        <span class="wp-stat">\u2699 ${progress.running} w toku</span>
        <span class="wp-stat">\u23F3 ${progress.pending} czeka</span>
        ${progress.failed > 0 ? `<span class="wp-stat" style="color:var(--danger)">\u2718 ${progress.failed} nieudane</span>` : ''}
    `;
}

function showWorkflowResult(result) {
    const el = $('workflow-result');
    el.classList.remove('hidden');
    el.innerHTML = `
        <h4>Wynik koncowy</h4>
        <div class="workflow-result-content">${renderMarkdown(result)}</div>
    `;
}

function appendWorkflowMessage(result) {
    // Add workflow result as a special assistant message in the chat
    const msgEl = document.createElement('div');
    msgEl.className = 'message assistant';
    msgEl.innerHTML = `
        <div class="message-avatar">\u129302</div>
        <div class="message-content">
            <p><strong>Wynik pracy zespolu agentow:</strong></p>
            <div>${renderMarkdown(result)}</div>
        </div>
    `;
    chatMessages.appendChild(msgEl);
    scrollToBottom();
}

// ──── Sidebar Tabs ────

function setupSidebarTabs() {
    document.querySelectorAll('.sidebar-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            // Deactivate all tabs
            document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.sidebar-tab-content').forEach(c => c.classList.remove('active'));

            // Activate clicked tab
            tab.classList.add('active');
            const targetId = tab.dataset.tab;
            const target = document.getElementById(targetId);
            if (target) target.classList.add('active');
        });
    });
}

// ──── Chat Toolbar (AI Capabilities) ────

function setupChatToolbar() {
    const caps = {
        'cap-internet': 'internet',
        'cap-code-exec': 'code_exec',
        'cap-filesystem': 'filesystem',
        'cap-shell': 'shell',
        'cap-rag': 'rag',
    };

    // Load saved preferences from localStorage
    const saved = localStorage.getItem('nf-capabilities');
    if (saved) {
        try {
            const parsed = JSON.parse(saved);
            Object.assign(state.capabilities, parsed);
        } catch (e) { /* ignore */ }
    }

    // Set initial checkbox states
    for (const [elId, capKey] of Object.entries(caps)) {
        const el = $(elId);
        if (el) {
            el.checked = state.capabilities[capKey] !== false;
            el.addEventListener('change', () => {
                state.capabilities[capKey] = el.checked;
                localStorage.setItem('nf-capabilities', JSON.stringify(state.capabilities));
                // Notify backend about capability change
                syncCapabilities();
            });
        }
    }

    // Initial sync
    syncCapabilities();
}

async function syncCapabilities() {
    try {
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                config: {
                    tools: {
                        web_search: state.capabilities.internet,
                        web_fetch: state.capabilities.internet,
                        execute_code: state.capabilities.code_exec,
                        read_file: state.capabilities.filesystem,
                        run_command: state.capabilities.shell,
                        search_documents: state.capabilities.rag,
                    }
                }
            }),
        });
    } catch (e) {
        // Silent fail - non-critical
    }
}

// ──── Resource Settings Panel ────

function setupResourcePanel() {
    // GPU VRAM slider
    const gpuSlider = $('resource-gpu-limit');
    const gpuVal = $('resource-gpu-limit-val');
    if (gpuSlider) {
        gpuSlider.addEventListener('input', () => {
            gpuVal.textContent = gpuSlider.value + '%';
        });
    }

    // RAM slider
    const ramSlider = $('resource-ram-limit');
    const ramVal = $('resource-ram-limit-val');
    if (ramSlider) {
        ramSlider.addEventListener('input', () => {
            ramVal.textContent = ramSlider.value + '%';
        });
    }

    // Save button
    const saveBtn = $('btn-save-resources');
    if (saveBtn) {
        saveBtn.addEventListener('click', saveResourceLimits);
    }

    // Load saved limits from config
    loadResourceLimits();

    // Start resource info updates (every 5s alongside mini monitor)
    updateResourceInfo();
    setInterval(updateResourceInfo, 5000);
}

async function loadResourceLimits() {
    try {
        const resp = await fetch('/api/config');
        const config = await resp.json();
        const resources = config.resources || {};

        if (resources.gpu_vram_limit !== undefined) {
            $('resource-gpu-limit').value = resources.gpu_vram_limit;
            $('resource-gpu-limit-val').textContent = resources.gpu_vram_limit + '%';
        }
        if (resources.ram_limit !== undefined) {
            $('resource-ram-limit').value = resources.ram_limit;
            $('resource-ram-limit-val').textContent = resources.ram_limit + '%';
        }
    } catch (e) {
        // Silent fail
    }
}

async function saveResourceLimits() {
    const gpuLimit = parseInt($('resource-gpu-limit').value);
    const ramLimit = parseInt($('resource-ram-limit').value);

    try {
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                config: {
                    resources: {
                        gpu_vram_limit: gpuLimit,
                        ram_limit: ramLimit,
                    }
                }
            }),
        });

        const btn = $('btn-save-resources');
        const orig = btn.textContent;
        btn.textContent = 'Zapisano!';
        btn.style.background = 'var(--success)';
        setTimeout(() => {
            btn.textContent = orig;
            btn.style.background = '';
        }, 2000);
    } catch (e) {
        alert('Blad zapisu limitow');
    }
}

async function updateResourceInfo() {
    try {
        const resp = await fetch('/api/monitor');
        const data = await resp.json();

        // GPU VRAM
        if (data.gpu) {
            const vramTotal = data.gpu.vram_total_gb;
            const vramUsed = data.gpu.vram_used_gb;
            if (vramTotal && vramTotal !== 'N/A') {
                const pct = Math.round((vramUsed / vramTotal) * 100);
                $('resource-gpu-info').textContent = `${vramUsed} / ${vramTotal} GB`;
                $('resource-gpu-fill').style.width = pct + '%';
            } else if (data.gpu.gpu_use_percent !== undefined && data.gpu.gpu_use_percent !== 'N/A') {
                $('resource-gpu-info').textContent = `${data.gpu.vendor || 'GPU'} ${data.gpu.gpu_use_percent}%`;
                $('resource-gpu-fill').style.width = data.gpu.gpu_use_percent + '%';
            }
        }

        // RAM
        if (data.ram) {
            const ramTotal = data.ram.total_gb || 0;
            const ramUsed = data.ram.used_gb || 0;
            $('resource-ram-info').textContent = `${ramUsed} / ${ramTotal} GB`;
            $('resource-ram-fill').style.width = (data.ram.percent || 0) + '%';
        }

        // Disk
        if (data.disk) {
            const diskTotal = data.disk.total_gb || 0;
            const diskUsed = data.disk.used_gb || 0;
            const diskFree = data.disk.free_gb || (diskTotal - diskUsed);
            $('resource-disk-info').textContent = `${diskUsed} / ${diskTotal} GB`;
            $('resource-disk-fill').style.width = (data.disk.percent || 0) + '%';
            $('resource-disk-free').textContent = `Wolne: ${diskFree} GB`;
        }
    } catch (e) {
        // Silent fail
    }
}

// Expose for inline event handlers
window.downloadModel = downloadModel;
window.removeDocument = removeDocument;
window.removeAttachment = removeAttachment;
window.toggleMonitor = toggleMonitor;
window.testRouter = testRouter;
window.copyCodeBlock = copyCodeBlock;
window.exportConversation = exportConversation;
window.deleteConversation = deleteConversation;
window.toggleWorkflowPanel = toggleWorkflowPanel;
window.cancelDownload = cancelDownload;
