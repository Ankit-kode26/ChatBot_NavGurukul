const API = "";

// ─── Sidebar (mobile) ────────────────────────────────────────────────────────
function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
  document.getElementById("overlay").classList.toggle("show");
}
function closeSidebar() {
  document.getElementById("sidebar").classList.remove("open");
  document.getElementById("overlay").classList.remove("show");
}

let chunksOpen = true;
let statusPollInterval = null;
let fullChunksData = [];

// ─── Init ─────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  setupDropZone();
  pollStatus();
  statusPollInterval = setInterval(pollStatus, 5000);
});

// ─── Status polling ────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    const data = await r.json();
    updateStatusUI(data);
  } catch (e) {
    setBadge("error", "Backend offline — run: python backend/main.py");
  }
}

function setBadge(cls, text) {
  ["db-badge", "db-badge-mobile"].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.className = `db-badge ${cls}`; el.textContent = text; }
  });
}

function updateStatusUI(data) {
  const stats  = data.vector_db;
  const chunks = stats.total_chunks;
  const pdfs   = stats.total_pdfs;

  document.getElementById("stat-pdfs").textContent   = pdfs;
  document.getElementById("stat-chunks").textContent = chunks >= 1000 ? (chunks/1000).toFixed(1)+"k" : chunks;

  if (chunks > 0) {
    setBadge("ready", `✓ ${chunks} chunks indexed`);
    const bm = document.getElementById("db-badge-mobile");
    if (bm) { bm.className="db-badge ready"; bm.textContent=`✓ ${chunks}`; }
  } else {
    setBadge("empty", "⚠ No documents ingested");
    const bm = document.getElementById("db-badge-mobile");
    if (bm) { bm.className="db-badge empty"; bm.textContent="⚠ Empty"; }
  }

  const pdfList = stats.pdf_list || [];
  if (pdfList.length > 0) {
    document.getElementById("pdf-list-section").style.display = "";
    const countEl = document.getElementById("pdf-count");
    if (countEl) countEl.textContent = pdfList.length;
    document.getElementById("pdf-list").innerHTML =
      pdfList.map(f => `<li title="${f}">${f}</li>`).join("");
  }

  const ing = data.ingestion;
  const statusEl = document.getElementById("ingest-status");
  if (ing.running) {
    statusEl.classList.remove("hidden");
    const events = ing.progress || [];
    if (events.length) {
      const last = events[events.length - 1];
      statusEl.innerHTML = `<span class="run">⏳ [${last.current}/${last.total}] ${last.filename} — ${last.state}</span>`;
    }
  } else if (ing.last_result) {
    statusEl.classList.remove("hidden");
    const res = ing.last_result;
    statusEl.innerHTML = res.status === "success"
      ? `<span class="ok">✓ Ingested ${res.total_pdfs} PDFs → ${res.total_chunks} chunks</span>`
      : `<span class="err">✗ ${res.message}</span>`;
    document.getElementById("btn-ingest").disabled = false;
    document.getElementById("btn-ingest").innerHTML = `<span>🔄</span> Start Ingestion`;
  }
}

// ─── Ingestion ─────────────────────────────────────────────────────────────
function setupDropZone() {
  const zone  = document.getElementById("drop-zone");
  const input = document.getElementById("file-input");
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault(); zone.classList.remove("drag-over");
    uploadFiles([...e.dataTransfer.files].filter(f => f.name.endsWith(".pdf")));
  });
  input.addEventListener("change", () => { uploadFiles([...input.files]); input.value = ""; });
}

async function uploadFiles(files) {
  const list = document.getElementById("upload-list");
  for (const file of files) {
    const item = document.createElement("div");
    item.className = "upload-item";
    item.innerHTML = `<span>${file.name}</span><span>⏳</span>`;
    list.appendChild(item);
    const form = new FormData();
    form.append("file", file);
    try {
      await fetch(`${API}/api/upload`, { method: "POST", body: form });
      item.className = "upload-item done";
      item.innerHTML = `<span>${file.name}</span><span>✓</span>`;
    } catch {
      item.className = "upload-item err";
      item.innerHTML = `<span>${file.name}</span><span>✗</span>`;
    }
  }
}

async function triggerIngest() {
  const btn = document.getElementById("btn-ingest");
  btn.disabled = true;
  btn.innerHTML = `<span>⏳</span> Running…`;
  const statusEl = document.getElementById("ingest-status");
  statusEl.classList.remove("hidden");
  statusEl.innerHTML = `<span class="run">Starting ingestion pipeline…</span>`;
  try {
    await fetch(`${API}/api/ingest`, { method: "POST" });
  } catch {
    statusEl.innerHTML = `<span class="err">✗ Failed to start ingestion</span>`;
    btn.disabled = false;
    btn.innerHTML = `<span>🔄</span> Start Ingestion`;
  }
}

// ─── Pipeline Step Animation ───────────────────────────────────────────────
const STEPS = [
  { icon: "🔢", label: "Embedding query…",        ms: 250  },
  { icon: "🔍", label: "Searching vector DB (HNSW top-20)…", ms: 350 },
  { icon: "⚖️", label: "Reranking to top-5…",     ms: 600  },
  { icon: "🧠", label: "Generating answer (LLM)…", ms: 99999 },
];

function showPipelineSteps() {
  const msgs = document.getElementById("messages");
  const wrap = document.createElement("div");
  wrap.className = "msg-row bot";
  wrap.id = "pipeline-steps";
  wrap.innerHTML = `
    <div class="msg-avatar">⚡</div>
    <div class="msg-content" style="width:100%;max-width:420px">
      <div class="pipeline-box" id="pipeline-box">
        ${STEPS.map((s, i) => `
          <div class="pipeline-step" id="pstep-${i}">
            <span class="pstep-icon">${s.icon}</span>
            <span class="pstep-label">${s.label}</span>
            <span class="pstep-status" id="pstep-status-${i}">—</span>
          </div>`).join("")}
      </div>
    </div>`;
  msgs.appendChild(wrap);
  msgs.scrollTop = msgs.scrollHeight;

  // Advance through steps using known timing estimates
  let elapsed = 0;
  STEPS.forEach((step, i) => {
    setTimeout(() => {
      document.getElementById(`pstep-${i}`).classList.add("active");
      if (i > 0) document.getElementById(`pstep-${i-1}`).classList.add("done");
    }, elapsed);
    elapsed += step.ms;
  });

  return wrap;
}

function completePipelineSteps(stepsEl) {
  STEPS.forEach((_, i) => {
    const el = document.getElementById(`pstep-${i}`);
    if (el) { el.className = "pipeline-step done"; }
  });
  stepsEl.remove();
}

// ─── Chat ──────────────────────────────────────────────────────────────────
function setQuery(btn) {
  const input = document.getElementById("query-input");
  input.value = btn.textContent;
  autoResize(input);
  input.focus();
}

function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuery(); }
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 140) + "px";
}

async function sendQuery() {
  const input = document.getElementById("query-input");
  const query = input.value.trim();
  if (!query) return;

  document.querySelector(".welcome-msg")?.remove();
  input.value = "";
  input.style.height = "";
  appendUserMessage(query);
  document.getElementById("chunks-panel").classList.add("hidden");
  setInputBusy(true);

  const stepsEl = showPipelineSteps();

  try {
    const r = await fetch(`${API}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await r.json();
    completePipelineSteps(stepsEl);
    fullChunksData = data.full_chunks || [];
    appendBotMessage(data);
    showChunks(data.chunks || [], data.timings || {});
  } catch (e) {
    completePipelineSteps(stepsEl);
    appendBotMessage({ answer: "⚠ Error contacting backend. Is the server running?", sources: [], timings: {} });
  } finally {
    setInputBusy(false);
  }
}

function setInputBusy(busy) {
  document.getElementById("query-input").disabled = busy;
  const btn = document.getElementById("send-btn");
  btn.disabled = busy;
  btn.classList.toggle("loading", busy);
}

function appendUserMessage(text) {
  const msgs = document.getElementById("messages");
  msgs.insertAdjacentHTML("beforeend", `
    <div class="msg-row user">
      <div class="msg-avatar">👤</div>
      <div class="msg-content">
        <div class="msg-bubble">${escHtml(text)}</div>
      </div>
    </div>`);
  msgs.scrollTop = msgs.scrollHeight;
}

function appendBotMessage(data) {
  const msgs = document.getElementById("messages");
  const { answer, sources = [], timings = {} } = data;

  const srcHtml = sources.length
    ? `<div class="sources-bar">${sources.map(s =>
        `<span class="source-tag" onclick="openPdf('${s.filename}', ${s.page})" title="Click to view PDF page">
          📄 ${s.filename} p.${s.page}
        </span>`).join("")}</div>`
    : "";

  const totalMs = timings.total_ms || 0;
  const cls = totalMs < 3000 ? "fast" : "med";
  const latHtml = timings.total_ms
    ? `<div class="latency-badge">
        <span class="${cls}">⚡ ${totalMs}ms</span>
        ${timings.embed_ms    ? `<span>embed ${timings.embed_ms}ms</span>` : ""}
        ${timings.retrieval_ms? `<span>search ${timings.retrieval_ms}ms</span>` : ""}
        ${timings.rerank_ms   ? `<span>rerank ${timings.rerank_ms}ms</span>` : ""}
        ${timings.llm_ms      ? `<span>llm ${timings.llm_ms}ms</span>` : ""}
      </div>` : "";

  msgs.insertAdjacentHTML("beforeend", `
    <div class="msg-row bot">
      <div class="msg-avatar">⚡</div>
      <div class="msg-content">
        <div class="msg-bubble">${escHtml(answer)}</div>
        ${srcHtml}
        ${latHtml}
      </div>
    </div>`);
  msgs.scrollTop = msgs.scrollHeight;
}

// ─── PDF Viewer ────────────────────────────────────────────────────────────
function openPdf(filename, page) {
  const url = `/pdfs/${encodeURIComponent(filename)}#page=${page}`;
  window.open(url, "_blank");
}

// ─── Full Chunk Modal ──────────────────────────────────────────────────────
function openChunkModal(idx) {
  const chunk = fullChunksData[idx];
  if (!chunk) return;
  const modal = document.getElementById("chunk-modal");
  document.getElementById("modal-filename").textContent = chunk.metadata.filename;
  document.getElementById("modal-page").textContent     = `Page ${chunk.metadata.page_number}`;
  document.getElementById("modal-scores").textContent   = `Cosine: ${chunk.cosine_similarity}  |  Rerank: ${chunk.rerank_score}`;
  document.getElementById("modal-text").textContent     = chunk.text;
  document.getElementById("modal-pdf-btn").onclick = () => openPdf(chunk.metadata.filename, chunk.metadata.page_number);
  modal.classList.remove("hidden");
}

function closeModal() {
  document.getElementById("chunk-modal").classList.add("hidden");
}

document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });

// ─── Chunks Panel ──────────────────────────────────────────────────────────
function showChunks(chunks, timings) {
  const panel = document.getElementById("chunks-panel");
  const body  = document.getElementById("chunks-body");
  const count = document.getElementById("chunks-count");
  if (!chunks.length) { panel.classList.add("hidden"); return; }

  panel.classList.remove("hidden", "collapsed");
  chunksOpen = true;
  document.getElementById("chunks-arrow").textContent = "▼";
  count.textContent = `${chunks.length} chunks`;

  body.innerHTML = chunks.map((c, i) => `
    <div class="chunk-card" onclick="openChunkModal(${i})" title="Click to view full text & open PDF">
      <div class="chunk-meta">
        <span class="chunk-filename">${c.metadata.filename}</span>
        <span class="chunk-page">p.${c.metadata.page_number}</span>
        <div class="chunk-scores">
          <span class="score-pill">cos ${c.cosine_similarity}</span>
          <span class="score-pill">rank ${c.rerank_score}</span>
        </div>
      </div>
      <div class="chunk-text">${escHtml(c.text)}</div>
      <div class="chunk-view-hint">Click to view full text & open PDF →</div>
    </div>`).join("");
}

function toggleChunks() {
  const panel = document.getElementById("chunks-panel");
  chunksOpen = !chunksOpen;
  panel.classList.toggle("collapsed", !chunksOpen);
  document.getElementById("chunks-arrow").textContent = chunksOpen ? "▼" : "▶";
}

// ─── Metrics (sidebar panel) ───────────────────────────────────────────────
async function showMetrics() {
  const panel = document.getElementById("metrics-panel");
  // Toggle: hide if already visible
  if (!panel.classList.contains("hidden")) {
    panel.classList.add("hidden");
    return;
  }
  try {
    const r = await fetch(`${API}/api/metrics`);
    const d = await r.json();
    if (d.status === "no_data") {
      panel.innerHTML = `<span class="metrics-hint">Ask a few questions first.</span>`;
    } else {
      const lat = d.latency;
      const ret = d.retrieval;
      const gen = d.generation;
      panel.innerHTML = `
        <div class="metrics-grid">
          <div class="metric-item"><span class="m-label">p50</span><span class="m-val">${lat.p50_ms}ms</span></div>
          <div class="metric-item"><span class="m-label">p95</span><span class="m-val">${lat.p95_ms}ms</span></div>
          <div class="metric-item"><span class="m-label">R@5</span><span class="m-val">${(ret.R_at_5 * 100).toFixed(0)}%</span></div>
          <div class="metric-item"><span class="m-label">MRR</span><span class="m-val">${ret.MRR}</span></div>
          <div class="metric-item"><span class="m-label">Citations</span><span class="m-val">${(gen.citation_accuracy * 100).toFixed(0)}%</span></div>
          <div class="metric-item"><span class="m-label">Halluc.</span><span class="m-val">${(gen.hallucination_rate_proxy * 100).toFixed(0)}%</span></div>
          <div class="metric-item" style="grid-column:1/-1"><span class="m-label">Total queries</span><span class="m-val">${d.total_queries}</span></div>
        </div>`;
    }
    panel.classList.remove("hidden");
  } catch (e) {
    alert("Backend offline.");
  }
}

function closeMetricsModal() {
  document.getElementById("metrics-modal")?.classList.add("hidden");
}



// ─── Util ──────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#039;");
}
