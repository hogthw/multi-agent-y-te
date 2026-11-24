/**
 * Pure-Vietnamese, Icon-Free Frontend Controller for HL7 FHIR R4 Multi-Agent Auditor.
 */

const state = {
  fhirCases: [],
  fhirMethods: [],
  models: [],
  logs: [],
  benchmark: { model_key: null, summaries: [] },
  selectedCase: null,
  abortController: null,
  replayTimer: null,
  isRunning: false,
  selectedAgentId: "doctor_2",
  memoryLog: [],
  agentHistory: {
    doctor_1: [],
    doctor_2: [],
    doctor_3: [],
    meta: [],
  },
  agentMemory: {
    doctor_1: null,
    doctor_2: null,
    doctor_3: null,
    meta: null,
  },
  agentMeta: {
    memory_vault: { name: "Kho bộ nhớ", role: "Ngữ nghĩa & Tình huống", color: "#0d9488" },
    doctor_1: { name: "BS. Linh", role: "Cấu trúc FHIR", color: "#0284c7" },
    doctor_2: { name: "BS. Minh", role: "Lâm sàng chéo", color: "#e11d48" },
    doctor_3: { name: "BS. Hùng", role: "Mã hóa & Chỉ số", color: "#7c3aed" },
    meta: { name: "BS. Thảo", role: "Thẩm định trưởng", color: "#d97706" },
  },
};

const METHOD_LABELS_VI = {
  rule_based: "Luật FHIR (Đối chứng 1)",
  single_llm: "Single-LLM (Đối chứng 2)",
  single_llm_memory: "Single-LLM + Bộ nhớ",
  multi_agent_no_mem: "Multi-Agent (Không bộ nhớ)",
  multi_agent_sem: "Multi-Agent + Ngữ nghĩa",
  multi_agent_memory: "Multi-Agent + Bộ nhớ 2 tầng",
  fhir_demo: "Mô phỏng 2 vòng",
};

const els = {
  canvas: document.getElementById("mdt-canvas"),
  caseSelect: document.getElementById("case-select"),
  methodSelect: document.getElementById("method-select"),
  modelSelect: document.getElementById("model-select"),
  modelField: document.getElementById("model-field"),
  roundsSelect: document.getElementById("rounds-select"),
  langSelect: document.getElementById("lang-select"),
  caseGtBadge: document.getElementById("case-gt-badge"),
  fhirJsonInput: document.getElementById("fhir-json-input"),
  btnStart: document.getElementById("btn-start"),
  btnStop: document.getElementById("btn-stop"),
  liveDot: document.getElementById("live-dot"),
  phaseText: document.getElementById("phase-text"),
  mRound: document.getElementById("m-round"),
  mConsensus: document.getElementById("m-consensus"),
  mMemory: document.getElementById("m-memory"),
  mTokens: document.getElementById("m-tokens"),
  mElapsed: document.getElementById("m-elapsed"),
  mVerdict: document.getElementById("m-verdict"),
  mVerdictBox: document.getElementById("m-verdict-box"),
  consensusBadge: document.getElementById("consensus-badge"),
  consensusBar: document.getElementById("consensus-bar"),
  transcriptList: document.getElementById("transcript-list"),
  inspectorPills: document.getElementById("inspector-agent-pills"),
  inspectorBody: document.getElementById("inspector-body"),
  benchmarkBody: document.getElementById("benchmark-body"),
  btnRefreshBench: document.getElementById("btn-refresh-bench"),
  logsList: document.getElementById("logs-list"),
  replaySpeed: document.getElementById("replay-speed"),
  btnRefreshLogs: document.getElementById("btn-refresh-logs"),
};

const room = new PixelMDTRoom(els.canvas, (agentId) => {
  state.selectedAgentId = agentId;
  switchTab("inspector");
  renderInspector();
});

function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.classList.toggle("active", p.id === `tab-${tabName}`);
  });
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

function setRunningUI(running) {
  state.isRunning = running;
  els.btnStart.disabled = running;
  els.btnStop.disabled = !running;
  els.liveDot.classList.toggle("running", running);
}

function escapeHtml(str) {
  return String(str || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatVerdictVi(ans) {
  if (!ans || ans === "VALID") return "HỢP LỆ";
  if (ans === "ĐỒNG Ý" || ans === "PHẢN ĐỐI") return ans;
  if (String(ans).startsWith("ERR:")) {
    return String(ans).replace("ERR:", "LỖI:");
  }
  return String(ans);
}

function resetRunState(maxRounds = 2, useMemory = true) {
  state.agentHistory = { doctor_1: [], doctor_2: [], doctor_3: [], meta: [] };
  state.agentMemory = { doctor_1: null, doctor_2: null, doctor_3: null, meta: null };
  state.memoryLog = [];
  room.resetSession(maxRounds, useMemory);
  els.transcriptList.innerHTML = "";
  els.mRound.textContent = `0 / ${maxRounds}`;
  els.mConsensus.textContent = "0 / 3";
  els.mMemory.textContent = "0";
  els.mTokens.textContent = "0";
  els.mElapsed.textContent = "0.0s";
  els.mVerdict.textContent = "Đang rà soát";
  els.mVerdictBox.className = "metric-pill verdict-pill";
  els.consensusBadge.textContent = "Đang kiểm toán";
  els.consensusBadge.className = "badge-muted badge-warn";
  els.consensusBar.style.width = "0%";
  renderInspector();
}

function appendRoundDivider(label) {
  const div = document.createElement("div");
  div.className = "round-divider";
  div.textContent = label;
  els.transcriptList.appendChild(div);
  els.transcriptList.scrollTop = els.transcriptList.scrollHeight;
}

function appendMemoryCard(ev) {
  const meta = state.agentMeta[ev.agent_id] || { name: ev.agent_id };
  const div = document.createElement("div");
  div.className = "memory-banner-card";
  const semIds = (ev.semantic_hits || []).map((h) => h.id).join(", ") || "Không có";
  const epiIds = (ev.episodic_hits || []).map((h) => h.id).join(", ") || "Không có";
  div.innerHTML = `
    <strong>Truy xuất Kho bộ nhớ cho ${escapeHtml(meta.name)}</strong> (${ev.total_hits} mục liên quan)<br />
    <span style="font-size:11px;">Ngữ nghĩa: <code>${escapeHtml(semIds)}</code> | Tình huống: <code>${escapeHtml(epiIds)}</code></span>
  `;
  div.style.cursor = "pointer";
  div.addEventListener("click", () => {
    state.selectedAgentId = ev.agent_id;
    room.selectAgent(ev.agent_id);
    switchTab("inspector");
    renderInspector();
  });
  els.transcriptList.appendChild(div);
  els.transcriptList.scrollTop = els.transcriptList.scrollHeight;
}

function appendTranscriptCard(ev) {
  const meta = state.agentMeta[ev.agent_id] || { name: ev.name, role: ev.specialty_vi, color: "#0284c7" };
  const card = document.createElement("div");
  card.className = "msg-card";
  card.style.borderLeftColor = meta.color;

  const ansVi = formatVerdictVi(ev.answer);
  let chipClass = "ans-chip";
  let chipText = ansVi;
  if (ev.phase === "review") {
    chipClass += ev.agree ? " agree" : " disagree";
    chipText = ev.agree ? "Đồng ý" : "Phản đối";
  } else if (ev.state === "final") {
    chipClass += " final";
    chipText = `Chốt: ${ansVi}`;
  } else if (String(ev.answer).startsWith("ERR")) {
    chipClass += " disagree";
  } else if (ev.answer === "VALID") {
    chipClass += " agree";
  }

  const tokens = ev.usage?.total_tokens || 0;
  card.innerHTML = `
    <div class="msg-head">
      <div>
        <span class="msg-author" style="color:${meta.color}">${escapeHtml(ev.name || meta.name)}</span>
        <span class="msg-meta"> — ${escapeHtml(ev.specialty_vi || meta.role)} (Vòng ${ev.round})</span>
      </div>
      <span class="${chipClass}">${escapeHtml(chipText)}</span>
    </div>
    <div class="msg-body">${escapeHtml(ev.explanation || ev.short_bubble || "")}</div>
    <div class="msg-foot">
      <span>Độ trễ: ${ev.latency_ms || 0} ms</span>
      <span>Token: ${tokens}</span>
      <span>Nhấn để xem bộ nhớ &amp; JSON</span>
    </div>
  `;

  card.addEventListener("click", () => {
    state.selectedAgentId = ev.agent_id;
    room.selectAgent(ev.agent_id);
    switchTab("inspector");
    renderInspector();
  });

  els.transcriptList.appendChild(card);
  els.transcriptList.scrollTop = els.transcriptList.scrollHeight;
}

function renderInspectorPills() {
  els.inspectorPills.innerHTML = "";
  for (const [id, info] of Object.entries(state.agentMeta)) {
    const btn = document.createElement("button");
    btn.className = `agent-pill-btn ${state.selectedAgentId === id ? "active" : ""}`;
    const count = id === "memory_vault" ? state.memoryLog.length : (state.agentHistory[id]?.length || 0);
    btn.innerHTML = `<span class="dot-ind" style="background:${info.color}"></span>${info.name} (${count})`;
    btn.addEventListener("click", () => {
      state.selectedAgentId = id;
      room.selectAgent(id);
      renderInspector();
    });
    els.inspectorPills.appendChild(btn);
  }
}

function renderMemoryHitsHtml(memObj) {
  if (!memObj || (!memObj.semantic_hits?.length && !memObj.episodic_hits?.length)) {
    return `<p class="muted" style="margin:0; font-size:12px;">Cấu hình hiện tại không dùng bộ nhớ bổ trợ hoặc không có mục khớp.</p>`;
  }
  let html = "";
  for (const h of memObj.semantic_hits || []) {
    html += `
      <div class="mem-hit-item">
        <strong>[Ngữ nghĩa | ${escapeHtml(h.id)} | độ tương đồng ${h.similarity}] ${escapeHtml(h.title)}</strong><br />
        <span>${escapeHtml(h.content)}</span>
      </div>
    `;
  }
  for (const h of memObj.episodic_hits || []) {
    html += `
      <div class="mem-hit-item episodic">
        <strong>[Tình huống | ${escapeHtml(h.id)} | ${escapeHtml(h.outcome)} | độ tương đồng ${h.similarity}] ${escapeHtml(h.title)}</strong><br />
        <span>${escapeHtml(h.lesson)}</span>
      </div>
    `;
  }
  return html;
}

function renderInspector() {
  renderInspectorPills();
  const id = state.selectedAgentId;
  const info = state.agentMeta[id];

  if (id === "memory_vault") {
    if (!state.memoryLog.length) {
      els.inspectorBody.innerHTML = `
        <div class="empty-state">
          <p><strong style="color:#0d9488">Kho bộ nhớ 2 tầng (Ngữ nghĩa &amp; Tình huống)</strong></p>
          <p class="muted">Chạy cấu hình có Tăng cường Bộ nhớ để xem các quy chuẩn HL7 FHIR R4, VN-FHIR và bài học tình huống được truy xuất cho từng tác nhân.</p>
        </div>
      `;
      return;
    }
    els.inspectorBody.innerHTML = state.memoryLog
      .map((m) => {
        const target = state.agentMeta[m.agent_id]?.name || m.agent_id;
        return `
          <div class="inspect-block">
            <h4>Truy xuất cho ${escapeHtml(target)} (Chuyên môn: ${escapeHtml(m.domain)})</h4>
            ${renderMemoryHitsHtml(m)}
          </div>
        `;
      })
      .join("");
    return;
  }

  const history = state.agentHistory[id] || [];
  const memObj = state.agentMemory[id];

  if (!history.length) {
    els.inspectorBody.innerHTML = `
      <div class="empty-state">
        <p><strong style="color:${info.color}">${info.name} (${info.role})</strong> chưa có lượt phát biểu.</p>
        <p class="muted">Khi chạy phiên rà soát FHIR, bạn sẽ xem được bộ nhớ truy xuất, câu lệnh hệ thống và JSON đầu ra tại đây.</p>
      </div>
    `;
    return;
  }

  const latest = history[history.length - 1];
  const totalTok = history.reduce((acc, x) => acc + (x.usage?.total_tokens || 0), 0);
  const avgLat = Math.round(history.reduce((acc, x) => acc + (x.latency_ms || 0), 0) / history.length);

  els.inspectorBody.innerHTML = `
    <div class="inspect-block">
      <h4>Thông số tác nhân — ${escapeHtml(info.name)} (${escapeHtml(info.role)})</h4>
      <div style="font-size:12px; display:flex; gap:14px; flex-wrap:wrap;">
        <span>Số lượt gọi: <strong>${history.length}</strong></span>
        <span>Độ trễ trung bình: <strong>${avgLat} ms</strong></span>
        <span>Tổng Token: <strong>${totalTok}</strong></span>
      </div>
    </div>

    <div class="inspect-block">
      <h4>Tri thức Bộ nhớ đã nạp cho ${escapeHtml(info.name)}</h4>
      ${renderMemoryHitsHtml(memObj || latest.memory_retrieved)}
    </div>

    <div class="inspect-block">
      <h4>Dữ liệu JSON gốc (Vòng ${latest.round} — ${escapeHtml(latest.phase)})</h4>
      <pre class="code-box">${escapeHtml(latest.raw_output || "{}")}</pre>
    </div>

    <div class="inspect-block">
      <h4>Câu lệnh hệ thống (System Prompt)</h4>
      <pre class="code-box">${escapeHtml(latest.system_prompt || "")}</pre>
    </div>
  `;
}

function renderBenchmark() {
  const summaries = state.benchmark?.summaries || [];
  if (!summaries.length) {
    els.benchmarkBody.innerHTML = `
      <div class="empty-state">
        <p>Chưa có kết quả thực nghiệm tổng hợp.</p>
        <p class="muted">Chạy lệnh sau trong terminal để sinh bảng thống kê Bootstrap F1, FPR, độ trễ và token:</p>
        <pre class="code-box" style="text-align:left">.venv/bin/python -m fhir_audit.run_benchmark --model gemini-3.1-flash-lite --per_category 3</pre>
      </div>
    `;
    return;
  }

  const rowsHtml = summaries
    .map((s) => {
      const highlight = s.method === "multi_agent_memory" ? "bench-row-highlight" : "";
      const pc = s.per_category || {};
      const label = METHOD_LABELS_VI[s.method] || s.method;
      return `
        <tr class="${highlight}">
          <td>${escapeHtml(label)}</td>
          <td><strong>${s.f1}%</strong><br /><span class="muted" style="font-size:10px">±${s.f1_bootstrap_std}</span></td>
          <td>${s.fpr}%</td>
          <td>${pc.STRUCTURAL?.category_recall ?? "-"}%</td>
          <td>${pc.TERMINOLOGY?.category_recall ?? "-"}%</td>
          <td><strong>${pc.CLINICAL_SEMANTIC?.category_recall ?? "-"}%</strong></td>
          <td>${s.mean_latency_s}s</td>
          <td>${s.mean_tokens}</td>
        </tr>
      `;
    })
    .join("");

  els.benchmarkBody.innerHTML = `
    <div style="font-size:11.5px; color:#475569;">
      Mô hình nền: <strong>${escapeHtml(state.benchmark.model_key || "—")}</strong> | Báo cáo xuất tại <code>logs/fhir_audit/results/benchmark_report.md</code>
    </div>
    <table class="bench-table">
      <thead>
        <tr>
          <th>Phương pháp</th>
          <th>F1 (Boot)</th>
          <th>Báo giả</th>
          <th>Cấu trúc</th>
          <th>Mã hóa</th>
          <th>Lâm sàng</th>
          <th>Trễ (s)</th>
          <th>Token</th>
        </tr>
      </thead>
      <tbody>${rowsHtml}</tbody>
    </table>
  `;
}

function handleEvent(ev) {
  if (typeof ev.ts === "number") {
    els.mElapsed.textContent = `${ev.ts.toFixed(1)}s`;
  }

  switch (ev.type) {
    case "session_start": {
      resetRunState(ev.max_rounds || 2, ev.use_memory !== false);
      const mLabel = METHOD_LABELS_VI[ev.method] || ev.method;
      els.phaseText.textContent = `Đang chạy [${mLabel}] trên hồ sơ ${ev.case_id || ev.qid}`;
      break;
    }
    case "memory_retrieve": {
      state.memoryLog.push(ev);
      state.agentMemory[ev.agent_id] = ev;
      room.triggerMemoryPulse(ev.total_hits || 0);
      els.mMemory.textContent = String(room.memoryHitCount);
      appendMemoryCard(ev);
      break;
    }
    case "round_start": {
      room.setRound(ev.round, ev.max_rounds);
      els.mRound.textContent = `${ev.round} / ${ev.max_rounds}`;
      els.phaseText.textContent = ev.label;
      appendRoundDivider(ev.label);
      break;
    }
    case "phase_change": {
      els.phaseText.textContent = ev.label;
      appendRoundDivider(ev.label);
      break;
    }
    case "agent_state": {
      room.setAgentState(ev.agent_id, ev.state, ev.status_text || "");
      break;
    }
    case "packet_fly": {
      room.spawnPacket(ev.from, ev.to, ev.label, ev.color);
      break;
    }
    case "agent_output": {
      room.setAgentState(ev.agent_id, ev.state, ev.short_bubble || ev.explanation, formatVerdictVi(ev.answer));
      if (state.agentHistory[ev.agent_id]) {
        state.agentHistory[ev.agent_id].push(ev);
      }
      if (ev.total_tokens) {
        els.mTokens.textContent = (ev.total_tokens.total_tokens || 0).toLocaleString();
      }
      appendTranscriptCard(ev);
      if (state.selectedAgentId === ev.agent_id) {
        renderInspector();
      } else {
        renderInspectorPills();
      }
      break;
    }
    case "consensus_update": {
      room.setConsensus(ev.agree_count, ev.total_doctors);
      els.mConsensus.textContent = `${ev.agree_count} / ${ev.total_doctors}`;
      const pct = Math.round((ev.agree_count / ev.total_doctors) * 100);
      els.consensusBar.style.width = `${pct}%`;
      if (ev.all_agree) {
        els.consensusBadge.textContent = "Đạt đồng thuận 3/3";
        els.consensusBadge.className = "badge-muted badge-ok";
      } else {
        els.consensusBadge.textContent = `Chưa đồng thuận (${ev.agree_count}/3) — Mở vòng tiếp`;
        els.consensusBadge.className = "badge-muted badge-warn";
      }
      break;
    }
    case "session_complete": {
      setRunningUI(false);
      els.mElapsed.textContent = `${((ev.total_elapsed_ms || 0) / 1000).toFixed(1)}s`;
      els.mTokens.textContent = (ev.total_tokens?.total_tokens || 0).toLocaleString();
      const gtIssues = ev.ground_truth_issues || [];
      const gtHas = gtIssues.length > 0;
      const predHas = Boolean(ev.has_issues);
      const isMatch = gtHas === predHas;
      const ans = formatVerdictVi(ev.final_answer || (predHas ? "ERR" : "VALID"));
      els.mVerdict.textContent = `${ans} (${isMatch ? "Khớp chuẩn" : "Lệch chuẩn"})`;
      els.mVerdictBox.className = `metric-pill verdict-pill ${isMatch ? "ok" : "bad"}`;
      els.phaseText.textContent = `Hoàn tất rà soát — Kết luận: ${ans} | Nhãn chuẩn: ${gtHas ? gtIssues[0].error_code : "HỢP LỆ"}`;
      loadConfig(true);
      break;
    }
    case "error": {
      setRunningUI(false);
      els.phaseText.textContent = `Lỗi thực thi: ${ev.message}`;
      const errDiv = document.createElement("div");
      errDiv.className = "msg-card";
      errDiv.style.borderLeftColor = "#dc2626";
      errDiv.innerHTML = `<strong style="color:#b91c1c">Lỗi thực thi</strong><div class="msg-body">${escapeHtml(ev.message)}</div>`;
      els.transcriptList.appendChild(errDiv);
      break;
    }
  }
}

async function startAudit() {
  stopAudit();
  let bundleObj = null;
  try {
    bundleObj = JSON.parse(els.fhirJsonInput.value.trim());
  } catch (e) {
    alert(`Dữ liệu JSON FHIR không hợp lệ: ${e.message}`);
    return;
  }

  const method = els.methodSelect.value;
  const maxRounds = parseInt(els.roundsSelect.value, 10) || 2;
  const useMem = ["fhir_demo", "multi_agent_memory", "multi_agent_sem", "single_llm_memory"].includes(method);
  const payload = {
    method,
    bundle: bundleObj,
    model_key: els.modelSelect.value,
    max_rounds: maxRounds,
    language: els.langSelect.value,
    case_id: state.selectedCase?.case_id || "custom_fhir",
    ground_truth_issues: state.selectedCase?.ground_truth_issues || [],
  };

  switchTab("transcript");
  resetRunState(maxRounds, useMem);
  setRunningUI(true);
  state.abortController = new AbortController();

  try {
    const resp = await fetch("/api/fhir/audit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: state.abortController.signal,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      handleEvent({ type: "error", message: err.error || "Yêu cầu thất bại" });
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const chunk of parts) {
        const line = chunk.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        try {
          handleEvent(JSON.parse(line.slice(6)));
        } catch (e) {
          console.error("SSE parse error:", e);
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      handleEvent({ type: "error", message: err.message });
    }
  } finally {
    setRunningUI(false);
  }
}

function stopAudit() {
  if (state.abortController) {
    state.abortController.abort();
    state.abortController = null;
  }
  if (state.replayTimer) {
    clearTimeout(state.replayTimer);
    state.replayTimer = null;
  }
  setRunningUI(false);
}

async function replayLog(filename) {
  stopAudit();
  const resp = await fetch(`/api/logs/${encodeURIComponent(filename)}`);
  if (!resp.ok) return;
  const log = await resp.json();
  const events = log.events || [];
  if (!events.length) return;

  switchTab("transcript");
  setRunningUI(true);
  const speed = parseFloat(els.replaySpeed.value) || 2;
  let idx = 0;
  const step = () => {
    if (idx >= events.length) {
      setRunningUI(false);
      return;
    }
    const ev = events[idx++];
    handleEvent(ev);
    if (idx < events.length) {
      const delay = ev.type === "agent_output" ? 1100 / speed : 320 / speed;
      state.replayTimer = setTimeout(step, delay);
    } else {
      setRunningUI(false);
    }
  };
  step();
}

function renderLogs() {
  if (!state.logs.length) {
    els.logsList.innerHTML = `
      <div class="empty-state">
        <p>Chưa có bản ghi phiên rà soát nào được lưu.</p>
        <p class="muted">Khi chạy các cấu hình có LLM, hệ thống sẽ tự lưu lịch sử để bạn phát lại trong buổi báo cáo.</p>
      </div>
    `;
    return;
  }
  els.logsList.innerHTML = "";
  for (const log of state.logs) {
    const row = document.createElement("div");
    row.className = "log-item";
    const mLabel = METHOD_LABELS_VI[log.method] || log.method;
    row.innerHTML = `
      <div class="log-info">
        <div><strong>${escapeHtml(log.case_id)}</strong> — <span style="color:#0284c7">${escapeHtml(mLabel)}</span></div>
        <div class="log-sub">Mô hình: ${escapeHtml(log.model_key)} | Kết luận: <strong>${escapeHtml(formatVerdictVi(log.final_answer))}</strong> | ${log.total_rounds} vòng</div>
      </div>
      <button class="btn-mini">Phát lại</button>
    `;
    row.querySelector("button").addEventListener("click", () => replayLog(log.filename));
    els.logsList.appendChild(row);
  }
}

function applySelectedCase(index) {
  const c = state.fhirCases[index];
  if (!c) return;
  state.selectedCase = c;
  els.fhirJsonInput.value = JSON.stringify(c.bundle, null, 2);
  const gt = c.ground_truth_issues || [];
  if (!gt.length) {
    els.caseGtBadge.textContent = "Nhãn chuẩn: HỢP LỆ (Không có lỗi)";
    els.caseGtBadge.className = "badge-muted badge-ok";
  } else {
    els.caseGtBadge.textContent = `Nhãn chuẩn: ${gt[0].category} / ${gt[0].error_code}`;
    els.caseGtBadge.className = "badge-muted badge-err";
  }
}

async function loadConfig(refreshOnly = false) {
  const resp = await fetch("/api/config");
  const data = await resp.json();
  state.logs = data.logs || [];
  state.benchmark = data.benchmark || { model_key: null, summaries: [] };
  renderLogs();
  renderBenchmark();
  if (refreshOnly) return;

  state.fhirCases = data.fhir_cases || [];
  state.fhirMethods = data.fhir_methods || [];
  state.models = data.models || [];

  els.caseSelect.innerHTML = "";
  let defaultCaseIdx = 0;
  state.fhirCases.forEach((c, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = c.title;
    if (c.case_id === "fhir_sem_02") defaultCaseIdx = idx;
    els.caseSelect.appendChild(opt);
  });
  els.caseSelect.value = String(defaultCaseIdx);

  els.methodSelect.innerHTML = "";
  state.fhirMethods.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    els.methodSelect.appendChild(opt);
  });

  els.modelSelect.innerHTML = "";
  if (!state.models.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Chưa cấu hình khóa API trong tệp .env";
    els.modelSelect.appendChild(opt);
  } else {
    state.models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.key;
      opt.textContent = `${m.label} (${m.key})`;
      if (m.key === "gemini-3.5-flash-lite") opt.selected = true;
      els.modelSelect.appendChild(opt);
    });
  }

  applySelectedCase(defaultCaseIdx);
  renderInspectorPills();
}

els.caseSelect.addEventListener("change", (e) => applySelectedCase(Number(e.target.value)));
els.methodSelect.addEventListener("change", () => {
  const m = els.methodSelect.value;
  els.modelField.style.opacity = m === "fhir_demo" || m === "rule_based" ? "0.45" : "1";
});
els.btnStart.addEventListener("click", startAudit);
els.btnStop.addEventListener("click", () => {
  stopAudit();
  els.phaseText.textContent = "Đã dừng phiên rà soát.";
});
els.btnRefreshLogs.addEventListener("click", () => loadConfig(true));
els.btnRefreshBench.addEventListener("click", () => loadConfig(true));

loadConfig();
