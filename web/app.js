// AnkaLM Local AI Chat Application & Training Control Center Frontend
let activeConversationId = null;
let activeAbortController = null;
let availableModels = [];

function escapeHtml(text) {
  const div = document.createElement("div");
  div.innerText = text || "";
  return div.innerHTML;
}

function renderMarkdownAndCode(rawText) {
  if (!rawText) return "";
  const parts = rawText.split(/```(\w*)\n([\s\S]*?)```/g);
  let html = "";
  for (let i = 0; i < parts.length; i += 3) {
    let textSegment = escapeHtml(parts[i]);
    textSegment = textSegment.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    textSegment = textSegment.replace(/`([^`]+)`/g, "<code>$1</code>");
    textSegment = textSegment.replace(/\n/g, "<br/>");
    html += textSegment;
    if (i + 2 < parts.length) {
      const lang = escapeHtml(parts[i + 1] || "code");
      const code = escapeHtml(parts[i + 2]);
      html += `<div class="code-block"><div style="font-size:0.7rem;color:#94a3b8;margin-bottom:4px;">${lang}</div><pre>${code}</pre></div>`;
    }
  }
  return html;
}

function switchView(tabName) {
  document.querySelectorAll(".nav-tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tabName);
  });
  document.querySelectorAll(".view-panel").forEach((p) => {
    p.classList.remove("active");
  });
  document.getElementById(`view-${tabName}`).classList.add("active");
  if (tabName === "dashboard") refreshDashboard();
  if (tabName === "eval") refreshEvaluationAndHardware();
  if (tabName === "models") refreshModelManagerView();
}

let activeUploadedDocFilename = null;
let activeExcelFile = null;
let activeVisionFile = null;

async function analyzeUploadedDocumentWorkspace(input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];
  activeUploadedDocFilename = file.name;
  const formData = new FormData();
  formData.append("file", file);
  formData.append("workspace", "pdf");
  const res = await fetch("/documents/upload", { method: "POST", body: formData });
  const data = await res.json();
  document.getElementById("doc-workspace-meta").innerText = JSON.stringify(data, null, 2);
  await loadRagDocuments();
}

async function askDocumentWorkspaceQuestion() {
  const q = document.getElementById("doc-qa-input").value.trim() || "Bu belgeyi özetle.";
  const res = await fetch("/documents/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: q, filename: activeUploadedDocFilename }),
  });
  const data = await res.json();
  document.getElementById("doc-qa-output").innerText =
    `${data.answer}\n\nKaynak Atıfları:\n${(data.citations || []).join("\n")}`;
}

async function analyzeExcelWorkspace(input) {
  if (!input.files || !input.files[0]) return;
  activeExcelFile = input.files[0];
  await askExcelWorkspaceQuestion();
}

async function askExcelWorkspaceQuestion() {
  if (!activeExcelFile) return;
  const q = document.getElementById("excel-qa-input").value.trim() || "Bu Excel dosyasını özetle.";
  const formData = new FormData();
  formData.append("file", activeExcelFile);
  formData.append("question", q);
  const res = await fetch("/spreadsheets/analyze", { method: "POST", body: formData });
  const data = await res.json();
  document.getElementById("excel-workspace-meta").innerText = JSON.stringify(data.ingestion?.format_specific_analysis || {}, null, 2);
  document.getElementById("excel-qa-output").innerText =
    `${data.qa?.answer || ""}\n\nHücre Kaynak Atıfları:\n${(data.qa?.citations || []).join("\n")}`;
}

async function analyzeVisionWorkspace(input) {
  if (!input.files || !input.files[0]) return;
  activeVisionFile = input.files[0];
  if (activeVisionFile.name.toLowerCase().endsWith(".wav")) {
    const fd = new FormData();
    fd.append("file", activeVisionFile);
    const res = await fetch("/audio/analyze", { method: "POST", body: fd });
    const data = await res.json();
    document.getElementById("vision-workspace-meta").innerText = JSON.stringify(data, null, 2);
    document.getElementById("vision-qa-output").innerText = data.summary || "";
    return;
  }
  await askVisionWorkspaceQuestion();
}

async function askVisionWorkspaceQuestion() {
  if (!activeVisionFile) return;
  const q = document.getElementById("vision-qa-input").value.trim() || "Bu resimde ne var?";
  const formData = new FormData();
  formData.append("file", activeVisionFile);
  formData.append("question", q);
  const res = await fetch("/images/analyze", { method: "POST", body: formData });
  const data = await res.json();
  document.getElementById("vision-workspace-meta").innerText = JSON.stringify(data.ingestion?.format_specific_analysis || {}, null, 2);
  document.getElementById("vision-qa-output").innerText =
    `${data.qa?.answer || ""}\n\nGörsel & OCR Atıfları:\n${(data.qa?.citations || []).join("\n")}`;
}

async function runSelectedTool() {
  const toolName = document.getElementById("tool-name-selector").value;
  let args = {};
  try {
    args = JSON.parse(document.getElementById("tool-args-input").value || "{}");
  } catch (e) {
    document.getElementById("tool-exec-output").innerText = "Geçersiz JSON argümanı: " + e;
    return;
  }
  const res = await fetch("/tools/execute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tool_name: toolName, arguments: args, user_authorized: false }),
  });
  const data = await res.json();
  document.getElementById("tool-exec-output").innerText = JSON.stringify(data, null, 2);
}

async function refreshModelManagerView() {
  const [famRes, moeRes, lvRes] = await Promise.all([
    fetch("/api/family/report"),
    fetch("/api/moe/report"),
    fetch("/api/low-vram/report"),
  ]);
  const fam = await famRes.json();
  const moe = await moeRes.json();
  const lv = await lvRes.json();

  const famRows = (fam || []).map((f) =>
    `<li><strong>${escapeHtml(f.name)}</strong> [${escapeHtml(f.family)}]: Toplam Parametre = <strong>${Number(f.total_parameters).toLocaleString("tr-TR")}</strong> | Aktif Parametre/Belirteç = <strong>${Number(f.active_parameters_per_token).toLocaleString("tr-TR")}</strong> | Eğitim RAM = ${f.estimated_train_ram_gb} GB</li>`
  ).join("");
  document.getElementById("family-summary-box").innerHTML = `<h3>🏛️ AnkaLM Model Ailesi (8B Flagship, 4B Balanced, Sparse MoE, Dense, 2GB Edge)</h3><ul style="margin-top:0.5rem;line-height:1.65;">${famRows}</ul>`;

  const rd = moe.routing_diagnostics || {};
  document.getElementById("moe-summary-box").innerHTML = `
    <h3>🧠 AnkaLM-MoE (Sparse Mixture-of-Experts) Yönlendirme Özeti</h3>
    <p><strong>Toplam Saklanan Parametre:</strong> ${Number(rd.total_stored_parameters || 0).toLocaleString("tr-TR")} • <strong>Aktif Parametre/Belirteç:</strong> ${Number(rd.active_parameters_per_token || 0).toLocaleString("tr-TR")} (${((rd.active_fraction || 0)*100).toFixed(1)}%)</p>
    <p><strong>Uzman Sayısı:</strong> ${rd.num_experts} Yönlendirilen + 1 Paylaşılan Uzman (Top-${rd.top_k_experts})</p>
    <p><strong>Uzman Seçim Dağılımı:</strong> ${JSON.stringify(rd.aggregate_expert_shares)} • <strong>Çökme (Collapse):</strong> ${rd.collapse_detected_any_layer ? "Var" : "Yok (%0 Ölü Uzman)"}</p>
  `;

  const sd = lv.speculative_decoding || {};
  const profRows = (lv.low_vram_profiles || []).map((p) =>
    `<li><strong>${escapeHtml(p.profile_name)}</strong> (${escapeHtml(p.model_name)}, ${p.precision}): Model Ağırlık Belleği = <strong>${p.model_weights_mb} MB</strong> | İlk Belirteç = ${p.first_token_latency_ms} ms | Hız = <strong>${p.tokens_per_second} tok/sn</strong> (2GB Uygun: ${p.fits_in_2gb_budget ? "EVET" : "HAYIR"})</li>`
  ).join("");
  document.getElementById("low-vram-summary-box").innerHTML = `
    <h3>⚡ 2GB Düşük VRAM Profilleri & Spekülatif Çözümleme</h3>
    <p><strong>Spekülatif Çözümleme (${escapeHtml(sd.draft_model)} -> ${escapeHtml(sd.target_model)}):</strong> Kabul Oranı = ${((sd.acceptance_rate || 0)*100).toFixed(1)}% • Hızlanma = <strong>${sd.speedup_ratio}x</strong> (${sd.speculative_tokens_per_sec} tok/sn) • Kalite Eşdeğerliği = ${sd.exact_quality_equivalence}</p>
    <ul style="margin-top:0.5rem;line-height:1.6;">${profRows}</ul>
  `;
}

function toggleTheme() {
  const root = document.documentElement;
  const current = root.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  document.getElementById("theme-toggle-btn").innerText = next === "dark" ? "☀️ Açık Tema" : "🌙 Koyu Tema";
}

function toggleSettingsDrawer() {
  const d = document.getElementById("settings-drawer");
  d.style.display = d.style.display === "none" ? "flex" : "none";
}

async function loadModels() {
  const res = await fetch("/api/models");
  const json = await res.json();
  availableModels = json.data || [];
  const sel = document.getElementById("model-selector");
  sel.innerHTML = "";

  const trained = availableModels.filter((m) => m.inference_ready);
  const untrained = availableModels.filter((m) => !m.inference_ready);

  trained.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.innerText = `${m.id} (${m.model_name} • adım ${m.global_step} • [${m.maturity_level}])`;
    sel.appendChild(opt);
  });
  untrained.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.disabled = true;
    opt.innerText = `${m.id} (${m.maturity_level} — Eğitilmemiş Rastgele Ağırlık)`;
    sel.appendChild(opt);
  });

  // Prefer small_sft or small_final or latest
  for (const pref of ["small_sft", "small_final", "latest", "best", "tiny_final"]) {
    if (trained.some((m) => m.id === pref)) {
      sel.value = pref;
      break;
    }
  }
  onModelSelectionChange();
}

function onModelSelectionChange() {
  const sel = document.getElementById("model-selector");
  const badge = document.getElementById("model-maturity-badge");
  const found = availableModels.find((m) => m.id === sel.value);
  if (!found) {
    badge.innerText = "Eğitilmiş Model Bekleniyor";
    return;
  }
  badge.innerText = `${found.maturity_level.toUpperCase()} • ${(found.parameter_count / 1e6).toFixed(2)}M Param • PPL ${found.val_perplexity}`;
}

async function loadConversations() {
  const res = await fetch("/api/conversations");
  const list = await res.json();
  renderConversationList(list);
  if (list.length === 0) {
    await createNewConversation();
  } else if (!activeConversationId) {
    await selectConversation(list[0].id);
  }
}

function renderConversationList(list) {
  const container = document.getElementById("conversation-list");
  container.innerHTML = "";
  list.forEach((c) => {
    const item = document.createElement("div");
    item.className = `conv-item ${c.id === activeConversationId ? "active" : ""}`;
    item.onclick = () => selectConversation(c.id);
    item.innerHTML = `<span>${escapeHtml(c.title)}</span>`;
    container.appendChild(item);
  });
}

async function searchConversations(query) {
  const res = await fetch(`/api/conversations/search?q=${encodeURIComponent(query)}`);
  const list = await res.json();
  renderConversationList(list);
}

async function createNewConversation() {
  const sel = document.getElementById("model-selector");
  const sysPrompt = document.getElementById("cfg-system-prompt").value;
  const res = await fetch("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: "Yeni Türkçe Sohbet",
      system_prompt: sysPrompt,
      model_checkpoint: sel.value || "latest",
    }),
  });
  const conv = await res.json();
  activeConversationId = conv.id;
  await loadConversations();
  await selectConversation(conv.id);
}

async function selectConversation(cid) {
  activeConversationId = cid;
  const res = await fetch(`/api/conversations/${cid}`);
  const conv = await res.json();
  document.getElementById("active-conv-title").innerText = conv.title;
  document.getElementById("cfg-system-prompt").value = conv.system_prompt;
  renderMessages(conv.messages || []);
  document.querySelectorAll(".conv-item").forEach((el) => {
    el.classList.toggle("active", el.innerText === conv.title);
  });
}

function renderMessages(messages) {
  const container = document.getElementById("messages-container");
  if (!messages || messages.length === 0) {
    container.innerHTML = `
      <div class="empty-chat-state">
        <h3>🦅 AnkaLM Yerel Türkçe Sohbet Uygulamasına Hoş Geldiniz</h3>
        <p>Bu model hiçbir hazır ağırlık kullanılmadan, sıfırdan rastgele ağırlıklarla bu bilgisayarda eğitilmiştir.</p>
      </div>`;
    return;
  }
  container.innerHTML = "";
  messages.forEach((m) => {
    const card = document.createElement("div");
    card.className = `message-card ${m.role}`;
    const roleTitle = m.role === "user" ? "👤 Kullanıcı" : "🦅 AnkaLM";
    const metricsInfo = m.metrics && m.metrics.tokens_per_second
      ? ` • ${m.metrics.tokens_per_second} tok/sn • [${m.metrics.maturity_level || "trained"}]`
      : "";
    let citationsHtml = "";
    if (m.rag_citations && m.rag_citations.length > 0) {
      citationsHtml = `<div class="rag-citation-box"><strong>📚 Yerel RAG Belge Kaynakları (Model Ağırlıklarından Ayrı):</strong><br/>` +
        m.rag_citations.map((c) => escapeHtml(c.citation)).join("<br/>") + `</div>`;
    }
    card.innerHTML = `
      <div class="message-header">
        <span><strong>${roleTitle}</strong>${escapeHtml(metricsInfo)}</span>
        <div class="message-actions">
          <button onclick="copyMessageText(${JSON.stringify(m.content)})">📋 Kopyala</button>
          ${m.role === "user" ? `<button onclick="promptEditMessage('${m.id}', ${JSON.stringify(m.content)})">✏️ Düzenle</button>` : ""}
        </div>
      </div>
      <div class="message-body">${renderMarkdownAndCode(m.content)}</div>
      ${citationsHtml}
    `;
    container.appendChild(card);
  });
  container.scrollTop = container.scrollHeight;
}

function copyMessageText(text) {
  navigator.clipboard.writeText(text || "");
}

async function promptEditMessage(mid, currentText) {
  const updated = prompt("Mesajı düzenleyin:", currentText);
  if (updated === null || !updated.trim()) return;
  await fetch(`/api/messages/${mid}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: updated.trim(), truncate_after: true }),
  });
  await selectConversation(activeConversationId);
  await sendChatMessage(updated.trim(), true);
}

async function promptRenameConversation() {
  if (!activeConversationId) return;
  const current = document.getElementById("active-conv-title").innerText;
  const next = prompt("Yeni sohbet başlığı:", current);
  if (!next || !next.trim()) return;
  await fetch(`/api/conversations/${activeConversationId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: next.trim() }),
  });
  await loadConversations();
  await selectConversation(activeConversationId);
}

async function deleteActiveConversation() {
  if (!activeConversationId) return;
  await fetch(`/api/conversations/${activeConversationId}`, { method: "DELETE" });
  activeConversationId = null;
  await loadConversations();
}

async function exportActiveConversation() {
  if (!activeConversationId) return;
  window.open(`/api/conversations/${activeConversationId}/export`, "_blank");
}

async function importConversationFile(input) {
  if (!input.files || !input.files[0]) return;
  const text = await input.files[0].text();
  const payload = JSON.parse(text);
  const res = await fetch("/api/conversations/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const conv = await res.json();
  activeConversationId = conv.id;
  await loadConversations();
  await selectConversation(conv.id);
}

async function saveConversationSettings() {
  if (!activeConversationId) return;
  const sys = document.getElementById("cfg-system-prompt").value;
  const model = document.getElementById("model-selector").value;
  await fetch(`/api/conversations/${activeConversationId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ system_prompt: sys, model_checkpoint: model }),
  });
}

function handleChatKeydown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
}

function stopGeneration() {
  if (activeAbortController) {
    activeAbortController.abort();
    activeAbortController = null;
  }
}

async function regenerateLastResponse() {
  if (!activeConversationId) return;
  const res = await fetch(`/api/conversations/${activeConversationId}`);
  const conv = await res.json();
  const msgs = conv.messages || [];
  if (msgs.length === 0) return;
  const last = msgs[msgs.length - 1];
  if (last.role === "assistant") {
    await fetch(`/api/messages/${last.id}`, { method: "DELETE" });
  }
  const updatedRes = await fetch(`/api/conversations/${activeConversationId}`);
  const updatedConv = await updatedRes.json();
  const userMsgs = (updatedConv.messages || []).filter((m) => m.role === "user");
  if (userMsgs.length === 0) return;
  const lastUserText = userMsgs[userMsgs.length - 1].content;
  await fetch(`/api/messages/${userMsgs[userMsgs.length - 1].id}`, { method: "DELETE" });
  await sendChatMessage(lastUserText);
}

async function sendChatMessage(overrideText = null) {
  const input = document.getElementById("chat-input");
  const text = overrideText !== null ? overrideText : input.value.trim();
  if (!text) return;
  if (overrideText === null) input.value = "";

  if (!activeConversationId) {
    await createNewConversation();
  }

  const convRes = await fetch(`/api/conversations/${activeConversationId}`);
  const conv = await convRes.json();
  const historyMsgs = [
    { role: "system", content: document.getElementById("cfg-system-prompt").value },
    ...(conv.messages || []).map((m) => ({ role: m.role, content: m.content })),
    { role: "user", content: text },
  ];

  // Append temporary streaming assistant card
  const container = document.getElementById("messages-container");
  const tempCard = document.createElement("div");
  tempCard.className = "message-card assistant";
  tempCard.innerHTML = `<div class="message-header"><strong>🦅 AnkaLM (Üretiliyor...)</strong></div><div class="message-body" id="streaming-target">...</div>`;
  container.appendChild(tempCard);
  container.scrollTop = container.scrollHeight;

  document.getElementById("send-btn").style.display = "none";
  document.getElementById("stop-gen-btn").style.display = "inline-block";
  activeAbortController = new AbortController();

  try {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: activeAbortController.signal,
      body: JSON.stringify({
        model: document.getElementById("model-selector").value || "latest",
        messages: historyMsgs,
        stream: true,
        max_tokens: parseInt(document.getElementById("cfg-max-tokens").value, 10),
        temperature: parseFloat(document.getElementById("cfg-temp").value),
        top_k: parseInt(document.getElementById("cfg-topk").value, 10),
        top_p: parseFloat(document.getElementById("cfg-topp").value),
        repetition_penalty: parseFloat(document.getElementById("cfg-rep").value),
        greedy: document.getElementById("cfg-greedy").checked,
        seed: parseInt(document.getElementById("cfg-seed").value, 10) || 42,
        conversation_id: activeConversationId,
        use_rag: document.getElementById("use-rag-checkbox").checked,
      }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let accumulated = "";
    const targetEl = document.getElementById("streaming-target");

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunkStr = decoder.decode(value, { stream: true });
      const lines = chunkStr.split("\n");
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const dataStr = line.slice(6).trim();
        if (dataStr === "[DONE]") break;
        try {
          const parsed = JSON.parse(dataStr);
          const delta = parsed.choices?.[0]?.delta?.content || "";
          accumulated += delta;
          if (targetEl) targetEl.innerHTML = renderMarkdownAndCode(accumulated);
          container.scrollTop = container.scrollHeight;
        } catch (e) {}
      }
    }
  } catch (err) {
    console.warn("Stream interrupted or failed:", err);
  } finally {
    document.getElementById("send-btn").style.display = "inline-block";
    document.getElementById("stop-gen-btn").style.display = "none";
    activeAbortController = null;
    await selectConversation(activeConversationId);
  }
}

// --- Local RAG Documents ---
async function loadRagDocuments() {
  const res = await fetch("/api/rag/documents");
  const docs = await res.json();
  document.getElementById("rag-doc-count").innerText = `${docs.length} yerel belge dizinli`;
  const pills = document.getElementById("rag-doc-pills");
  pills.innerHTML = "";
  docs.forEach((d) => {
    const pill = document.createElement("span");
    pill.className = "rag-pill";
    pill.innerHTML = `📄 ${escapeHtml(d.filename)} (${d.chunk_count} parça) <button style="background:none;border:none;color:#ef4444;cursor:pointer;" onclick="deleteRagDoc('${d.id}')">✕</button>`;
    pills.appendChild(pill);
  });
}

async function uploadRagAttachment(input) {
  if (!input.files || !input.files[0]) return;
  const formData = new FormData();
  formData.append("file", input.files[0]);
  await fetch("/api/rag/upload", { method: "POST", body: formData });
  document.getElementById("use-rag-checkbox").checked = true;
  input.value = "";
  await loadRagDocuments();
}

async function deleteRagDoc(docId) {
  await fetch(`/api/rag/documents/${docId}`, { method: "DELETE" });
  await loadRagDocuments();
}

// --- Training Control Center Dashboard ---
async function loadTrainingConfigs() {
  const res = await fetch("/api/training/configs");
  const configs = await res.json();
  const sel = document.getElementById("train-config-selector");
  sel.innerHTML = "";
  configs.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.path;
    opt.innerText = `${c.filename} (${c.model_name} • ${c.max_steps} adım • ${c.precision})`;
    sel.appendChild(opt);
  });
}

async function refreshDashboard() {
  const [stateRes, logsRes] = await Promise.all([
    fetch("/api/training/state"),
    fetch("/api/training/logs?lines=60"),
  ]);
  const stateData = await stateRes.json();
  const logsData = await logsRes.json();
  const s = stateData.live || {};

  document.getElementById("st-status").innerText = `${(s.status || "idle").toUpperCase()} (PID: ${s.pid || 0} • Canlı: ${s.process_alive ? "Evet" : "Hayır"})`;
  document.getElementById("st-model").innerText = s.active_model || "-";
  document.getElementById("st-params").innerText = s.parameter_count ? s.parameter_count.toLocaleString("tr-TR") : "-";
  document.getElementById("st-vocab").innerText = s.tokenizer_vocab_size || "-";
  document.getElementById("st-steps").innerText = `${s.global_step || 0} / ${s.max_steps || 0} (Epoch ${s.epoch || 0})`;
  document.getElementById("st-tokens").innerText = s.processed_tokens ? s.processed_tokens.toLocaleString("tr-TR") : "0";
  document.getElementById("st-train-loss").innerText = s.train_loss !== undefined ? Number(s.train_loss).toFixed(4) : "-";
  document.getElementById("st-val-loss").innerText = s.val_loss !== undefined ? Number(s.val_loss).toFixed(4) : "-";
  document.getElementById("st-ppl").innerText = s.perplexity !== undefined ? Number(s.perplexity).toFixed(2) : "-";
  document.getElementById("st-lr").innerText = s.learning_rate !== undefined ? Number(s.learning_rate).toExponential(2) : "-";
  document.getElementById("st-tps").innerText = s.tokens_per_second ? `${s.tokens_per_second} tok/sn` : "-";
  document.getElementById("st-l2").innerText = s.param_l2_delta !== undefined ? Number(s.param_l2_delta).toFixed(4) : "-";
  document.getElementById("st-ram").innerText = `${s.cpu_usage_pct || 0}% CPU • ${s.ram_usage_mb || 0} MB RAM`;
  document.getElementById("st-gpu").innerText = `${s.gpu_utilization_pct || 0}% GPU • ${s.vram_usage_mb || 0} MB VRAM`;
  document.getElementById("st-ckpt").innerText = s.checkpoint_status || "-";
  document.getElementById("st-maturity").innerText = s.maturity_level || "-";

  document.getElementById("training-logs-pre").innerText = (logsData.lines || []).join("\n");
  drawLossChart(stateData.metrics_history || []);
}

function drawLossChart(history) {
  const canvas = document.getElementById("loss-chart-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!history || history.length < 2) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px sans-serif";
    ctx.fillText("Eğitim metrik geçmişi bekleniyor...", 24, 40);
    return;
  }

  const losses = history.map((h) => Number(h.train_loss || 0));
  const valLosses = history.map((h) => Number(h.val_loss || 0));
  const minL = Math.max(0, Math.min(...losses, ...valLosses) - 0.3);
  const maxL = Math.max(...losses, ...valLosses) + 0.3;
  const pad = 36;
  const w = canvas.width - pad * 2;
  const h = canvas.height - pad * 2;

  ctx.strokeStyle = "#334155";
  ctx.strokeRect(pad, pad, w, h);

  function plotSeries(arr, color) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    arr.forEach((v, idx) => {
      const x = pad + (idx / Math.max(arr.length - 1, 1)) * w;
      const y = pad + h - ((v - minL) / Math.max(maxL - minL, 1e-4)) * h;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  plotSeries(losses, "#38bdf8");
  plotSeries(valLosses, "#22c55e");

  ctx.fillStyle = "#38bdf8";
  ctx.fillText("■ Eğitim Kaybı (Train Loss)", pad + 10, pad + 18);
  ctx.fillStyle = "#22c55e";
  ctx.fillText("■ Doğrulama Kaybı (Val Loss)", pad + 210, pad + 18);
}

async function startTrainingJob() {
  const cfg = document.getElementById("train-config-selector").value || "configs/small.yaml";
  await fetch("/api/training/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config_path: cfg }),
  });
  setTimeout(refreshDashboard, 800);
}

async function resumeTrainingJob() {
  const cfg = document.getElementById("train-config-selector").value || "configs/small.yaml";
  await fetch("/api/training/resume", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config_path: cfg, checkpoint: "latest" }),
  });
  setTimeout(refreshDashboard, 800);
}

async function pauseOrResumeTrainingJob() {
  await fetch("/api/training/pause", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  setTimeout(refreshDashboard, 500);
}

async function saveImmediateCheckpoint() {
  await fetch("/api/training/save-checkpoint", { method: "POST" });
  setTimeout(refreshDashboard, 800);
}

async function stopTrainingJob() {
  await fetch("/api/training/stop", { method: "POST" });
  setTimeout(refreshDashboard, 800);
}

// --- Evaluation & Hardware View ---
async function refreshEvaluationAndHardware() {
  const [evalRes, hwRes] = await Promise.all([
    fetch("/api/evaluation/report"),
    fetch("/api/hardware/report"),
  ]);
  const evalData = await evalRes.json();
  const hwData = await hwRes.json();

  const evalBox = document.getElementById("eval-summary-box");
  if (evalData.status === "not_run_yet") {
    evalBox.innerHTML = "<p>Henüz değerlendirme raporu çalıştırılmadı. Yukarıdaki butona tıklayarak çalıştırabilirsiniz.</p>";
  } else {
    const catRows = Object.entries(evalData.category_scores || {})
      .map(([k, v]) => `<li><strong>${escapeHtml(k)}:</strong> ${(Number(v) * 100).toFixed(1)}%</li>`)
      .join("");
    evalBox.innerHTML = `
      <h3>📌 Değerlendirilen Model: ${escapeHtml(evalData.model_name)} (${escapeHtml(evalData.checkpoint_name)})</h3>
      <p><strong>Ayrılmış Test Kaybı (Held-out Loss):</strong> ${evalData.held_out_eval_loss} • <strong>Test Şaşkınlığı (Perplexity):</strong> ${evalData.held_out_perplexity} • <strong>Genel Başarı Puanı:</strong> ${(Number(evalData.overall_benchmark_score || 0) * 100).toFixed(1)}%</p>
      <ul style="margin-top:0.65rem;columns:2;list-style-position:inside;font-size:0.86rem;">${catRows}</ul>
    `;
  }

  const hwBox = document.getElementById("hardware-summary-box");
  hwBox.innerHTML = `
    <h3>🖥️ Algılanan Donanım ve Hızlandırıcı Raporu</h3>
    <p><strong>İşlemci (CPU):</strong> ${escapeHtml(hwData.cpu_name)} (${hwData.cpu_physical_cores} Fiziksel / ${hwData.cpu_logical_cores} Mantıksal Çekirdek)</p>
    <p><strong>Bellek (RAM):</strong> ${hwData.total_ram_gb} GB Toplam (${hwData.available_ram_gb} GB Kullanılabilir) • <strong>Boş Disk:</strong> ${hwData.disk_free_gb} GB</p>
    <p><strong>Seçilen Doğrulanmış Arka Uç:</strong> ${escapeHtml(hwData.selected_backend?.backend)} (${hwData.selected_backend?.bf16_gflops} GFLOPS BF16 / ${hwData.selected_backend?.fp32_gflops} GFLOPS FP32)</p>
  `;
}

async function runEvaluationNow() {
  const ckpt = document.getElementById("model-selector").value || "latest";
  await fetch("/api/training/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checkpoint: ckpt }),
  });
  await refreshEvaluationAndHardware();
  await loadModels();
}

window.addEventListener("DOMContentLoaded", async () => {
  await loadModels();
  await loadConversations();
  await loadRagDocuments();
  await loadTrainingConfigs();
  setInterval(() => {
    const dash = document.getElementById("view-dashboard");
    if (dash && dash.classList.contains("active")) {
      refreshDashboard();
    }
  }, 3000);
});
