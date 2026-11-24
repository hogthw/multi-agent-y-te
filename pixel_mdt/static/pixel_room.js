/**
 * Light-Theme Slide & Projector-Ready Pixel 2D Hospital FHIR MDT Room Renderer.
 * Bright daylight hospital conference room + glowing Semantic/Episodic Memory Vault
 * + 4 animated pixel-art auditors + flying FHIR/Memory packets + crisp HD Vietnamese overlays.
 */

class PixelMDTRoom {
  constructor(canvas, onSelectAgent) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onSelectAgent = onSelectAgent;

    // Logical pixel-art buffer (16:10 retro grid scaled crisply)
    this.PW = 320;
    this.PH = 200;
    this.pxCanvas = document.createElement("canvas");
    this.pxCanvas.width = this.PW;
    this.pxCanvas.height = this.PH;
    this.pctx = this.pxCanvas.getContext("2d");

    this.time = 0;
    this.selectedAgentId = null;
    this.hoveredAgentId = null;
    this.currentRound = 0;
    this.maxRounds = 2;
    this.boardAnswer = "SẴN SÀNG";
    this.consensusText = "0/3";
    this.memoryActive = false;
    this.memoryPulse = 0;
    this.memoryHitCount = 0;
    this.packets = [];

    // Memory Vault station (left wall) — clickable!
    this.memoryVault = {
      id: "memory_vault",
      x: 42,
      y: 25,
      w: 58,
      h: 32,
    };

    // 4 Agents positioned around the sunlit hospital conference table
    this.agents = {
      meta: {
        id: "meta",
        name: "BS. Thảo",
        role: "Thẩm định trưởng",
        homeX: 160,
        homeY: 67,
        x: 160,
        y: 67,
        targetX: 160,
        targetY: 67,
        hair: "#3b2314",
        skin: "#f6cfb2",
        coat: "#ffffff",
        shirt: "#d97706",
        accent: "#d97706",
        prop: "clipboard",
        state: "idle",
        bubble: "",
        answer: "",
      },
      doctor_1: {
        id: "doctor_1",
        name: "BS. Linh",
        role: "Cấu trúc FHIR",
        homeX: 72,
        homeY: 120,
        x: 72,
        y: 120,
        targetX: 72,
        targetY: 120,
        hair: "#1e293b",
        skin: "#f8d5ba",
        coat: "#ffffff",
        shirt: "#0284c7",
        accent: "#0284c7",
        prop: "stethoscope",
        state: "idle",
        bubble: "",
        answer: "",
      },
      doctor_2: {
        id: "doctor_2",
        name: "BS. Minh",
        role: "Lâm sàng Chéo",
        homeX: 160,
        homeY: 160,
        x: 160,
        y: 160,
        targetX: 160,
        targetY: 160,
        hair: "#291d18",
        skin: "#efc19d",
        coat: "#0d9488", // Teal clinical scrubs
        shirt: "#115e59",
        accent: "#e11d48",
        prop: "cap",
        state: "idle",
        bubble: "",
        answer: "",
      },
      doctor_3: {
        id: "doctor_3",
        name: "BS. Hùng",
        role: "Mã hóa & Chỉ số",
        homeX: 248,
        homeY: 120,
        x: 248,
        y: 120,
        targetX: 248,
        targetY: 120,
        hair: "#334155",
        skin: "#f3c6a5",
        coat: "#ffffff",
        shirt: "#7c3aed",
        accent: "#7c3aed",
        prop: "glasses",
        state: "idle",
        bubble: "",
        answer: "",
      },
    };

    this._bindEvents();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  resetSession(maxRounds = 2, useMemory = true) {
    this.currentRound = 0;
    this.maxRounds = maxRounds;
    this.boardAnswer = "ĐANG CHẠY";
    this.consensusText = "0/3";
    this.memoryActive = useMemory;
    this.memoryPulse = 0;
    this.memoryHitCount = 0;
    this.packets = [];
    Object.values(this.agents).forEach((a) => {
      a.state = "idle";
      a.bubble = "";
      a.answer = "";
      a.targetX = a.homeX;
      a.targetY = a.homeY;
    });
  }

  triggerMemoryPulse(hitCount) {
    this.memoryActive = true;
    this.memoryPulse = 1.8;
    this.memoryHitCount += hitCount || 0;
  }

  setRound(round, maxRounds) {
    this.currentRound = round;
    this.maxRounds = maxRounds || this.maxRounds;
  }

  setConsensus(agreeCount, total) {
    this.consensusText = `${agreeCount}/${total}`;
  }

  setAgentState(agentId, state, bubble = "", answer = "") {
    const a = this.agents[agentId];
    if (!a) return;
    a.state = state;
    if (bubble) a.bubble = bubble;
    if (answer) {
      a.answer = answer;
      if (agentId === "meta" || state === "final") {
        this.boardAnswer = answer;
      }
    }

    const step = state === "speaking" || state === "final" ? 6 : 0;
    if (agentId === "meta") {
      a.targetX = a.homeX;
      a.targetY = a.homeY + step;
    } else if (agentId === "doctor_1") {
      a.targetX = a.homeX + step;
      a.targetY = a.homeY;
    } else if (agentId === "doctor_2") {
      a.targetX = a.homeX;
      a.targetY = a.homeY - step;
    } else if (agentId === "doctor_3") {
      a.targetX = a.homeX - step;
      a.targetY = a.homeY;
    }
  }

  spawnPacket(fromId, toId, label, color) {
    const getPos = (id) => {
      if (id === "memory_vault") return { x: this.memoryVault.x, y: this.memoryVault.y + 8 };
      const ag = this.agents[id];
      return ag ? { x: ag.x, y: ag.y - 6 } : null;
    };
    const from = getPos(fromId);
    const to = getPos(toId);
    if (!from || !to) return;
    this.packets.push({
      fromX: from.x,
      fromY: from.y,
      toX: to.x,
      toY: to.y,
      progress: 0,
      label: label || "",
      color: color || "#0284c7",
    });
  }

  selectAgent(agentId) {
    this.selectedAgentId = agentId;
  }

  _bindEvents() {
    const getLogicalPos = (ev) => {
      const rect = this.canvas.getBoundingClientRect();
      return {
        lx: ((ev.clientX - rect.left) / rect.width) * this.PW,
        ly: ((ev.clientY - rect.top) / rect.height) * this.PH,
      };
    };

    const hitTest = (lx, ly) => {
      // Check Memory Vault hit
      if (lx >= 13 && lx <= 75 && ly >= 6 && ly <= 42) {
        return "memory_vault";
      }
      for (const a of Object.values(this.agents)) {
        if (Math.abs(lx - a.x) <= 18 && ly >= a.y - 24 && ly <= a.y + 15) {
          return a.id;
        }
      }
      return null;
    };

    this.canvas.addEventListener("mousemove", (ev) => {
      const { lx, ly } = getLogicalPos(ev);
      this.hoveredAgentId = hitTest(lx, ly);
      this.canvas.style.cursor = this.hoveredAgentId ? "pointer" : "default";
    });

    this.canvas.addEventListener("click", (ev) => {
      const { lx, ly } = getLogicalPos(ev);
      const hit = hitTest(lx, ly);
      if (hit) {
        this.selectedAgentId = hit;
        if (this.onSelectAgent) this.onSelectAgent(hit);
      }
    });
  }

  _px(x, y, w, h, color) {
    this.pctx.fillStyle = color;
    this.pctx.fillRect(Math.round(x), Math.round(y), Math.round(w), Math.round(h));
  }

  _drawLightHospitalRoom() {
    // --- BRIGHT DAYLIGHT HOSPITAL WALL ---
    this._px(0, 0, this.PW, 45, "#e2ecf7");
    this._px(0, 0, this.PW, 4, "#cbd9ea"); // ceiling molding
    this._px(0, 44, this.PW, 3, "#94a3b8"); // baseboard trim
    this._px(0, 47, this.PW, 2, "#cbd5e1"); // floor edge

    // --- CLEAN LIGHT CHECKERED HOSPITAL FLOOR ---
    for (let ty = 49; ty < this.PH; ty += 12) {
      for (let tx = 0; tx < this.PW; tx += 12) {
        const even = ((tx / 12 + (ty - 49) / 12) | 0) % 2 === 0;
        this._px(tx, ty, 12, 12, even ? "#f8fafc" : "#eef4fb");
      }
    }

    // Soft light-blue medical conference rug
    this._px(54, 72, 212, 100, "#bfdbfe");
    this._px(56, 74, 208, 96, "#dbeafe");
    this._px(58, 76, 204, 92, "#eff6ff");

    // --- LEFT WALL: TWO-TIER MEMORY VAULT (Semantic + Episodic) ---
    const pulseOn = this.memoryPulse > 0;
    const vaultFrame = this.hoveredAgentId === "memory_vault" || this.selectedAgentId === "memory_vault"
      ? "#0d9488"
      : pulseOn
      ? "#14b8a6"
      : "#64748b";
    this._px(14, 7, 60, 33, vaultFrame);
    this._px(16, 9, 56, 29, this.memoryActive ? "#f0fdfa" : "#f1f5f9");
    // Server/Memory rack bars (Tier 1 Semantic + Tier 2 Episodic)
    this._px(19, 12, 50, 9, this.memoryActive ? "#ccfbf1" : "#e2e8f0");
    this._px(19, 24, 50, 9, this.memoryActive ? "#ede9fe" : "#e2e8f0");
    // Blinking LED dots on Memory Vault
    const blinkA = Math.sin(this.time * (pulseOn ? 14 : 3)) > 0;
    this._px(21, 15, 3, 3, this.memoryActive ? (blinkA ? "#0d9488" : "#2dd4bf") : "#94a3b8");
    this._px(21, 27, 3, 3, this.memoryActive ? (!blinkA ? "#7c3aed" : "#a78bfa") : "#94a3b8");
    // Mini knowledge stripes
    this._px(27, 15, 36, 2, this.memoryActive ? "#0f766e" : "#94a3b8");
    this._px(27, 27, 32, 2, this.memoryActive ? "#6d28d9" : "#94a3b8");

    // --- CENTER WALL: BRIGHT WHITE DIGITAL FHIR AUDIT BOARD ---
    this._px(102, 6, 116, 35, "#64748b");
    this._px(104, 8, 112, 31, "#ffffff");
    // Medical cross badge
    this._px(108, 14, 7, 2, "#ef4444");
    this._px(110.5, 11.5, 2, 7, "#ef4444");
    // Right result status tile on board
    const ansUpper = String(this.boardAnswer).toUpperCase();
    const isErr = ansUpper.startsWith("LỖI") || ansUpper.startsWith("ERR");
    const isValid = ansUpper === "HỢP LỆ" || ansUpper === "VALID";
    const tileBg = isErr ? "#fee2e2" : isValid ? "#dcfce7" : "#f1f5f9";
    const tileBorder = isErr ? "#dc2626" : isValid ? "#16a34a" : "#94a3b8";
    this._px(174, 11, 39, 25, tileBorder);
    this._px(175, 12, 37, 23, tileBg);

    // --- RIGHT WALL: CLINICAL MONITOR (ECG) + HL7 FHIR SHELF ---
    this._px(244, 7, 62, 33, "#64748b");
    this._px(246, 9, 58, 29, "#ffffff");
    // Mini ECG screen (high contrast dark-green medical screen inside white frame)
    this._px(249, 12, 25, 22, "#064e3b");
    const sweep = Math.floor(this.time * 16) % 22;
    for (let i = 0; i < 22; i++) {
      let dy = 0;
      const p = (i + sweep) % 22;
      if (p === 8) dy = -2;
      else if (p === 9) dy = 2;
      else if (p === 10) dy = -6;
      else if (p === 11) dy = 4;
      else if (p === 13) dy = -2;
      this._px(250 + i, 24 + dy, 1, 1, p === 10 ? "#bbf7d0" : "#4ade80");
    }
    // Reference FHIR R4 books on right side of cabinet
    const books = ["#0284c7", "#0d9488", "#e11d48", "#7c3aed", "#d97706"];
    books.forEach((c, idx) => {
      this._px(277 + idx * 5, 13, 4, 10, c);
      this._px(277 + idx * 5, 25, 4, 9, books[(idx + 2) % books.length]);
    });

    // Potted Hospital Plants (fresh bright green)
    this._drawPlant(12, 57);
    this._drawPlant(298, 57);
    this._drawPlant(12, 174);
    this._drawPlant(298, 174);

    // --- CENTER SUNLIT OAK CONFERENCE TABLE ---
    this._px(102, 138, 118, 7, "rgba(15, 23, 42, 0.14)"); // soft floor shadow
    this._px(110, 133, 6, 9, "#64748b");
    this._px(204, 133, 6, 9, "#64748b");
    this._px(100, 92, 120, 44, "#926741"); // warm oak rim
    this._px(102, 94, 116, 40, "#d7aa7d"); // bright oak top
    this._px(104, 96, 112, 36, "#e3bc94"); // inner wood surface

    // Central FHIR Bundle Tablet on Table
    this._px(136, 103, 48, 22, "#334155");
    this._px(138, 105, 44, 18, "#ffffff");
    this._px(141, 108, 20, 2, "#0284c7");
    this._px(141, 112, 32, 1, "#94a3b8");
    this._px(141, 115, 26, 1, "#94a3b8");
    this._px(141, 118, 22, 1, "#cbd5e1");
    const ledOn = Math.sin(this.time * 5) > 0;
    this._px(175, 108, 4, 4, ledOn ? "#16a34a" : "#86efac");

    // Paper dossiers & coffee cups on table
    this._px(111, 102, 13, 9, "#ffffff");
    this._px(113, 104, 9, 1, "#0284c7");
    this._px(113, 106, 7, 1, "#94a3b8");
    this._px(196, 102, 13, 9, "#ffffff");
    this._px(198, 104, 9, 1, "#7c3aed");
    this._px(198, 106, 7, 1, "#94a3b8");
    this._px(155, 128, 10, 6, "#ffffff");
    this._px(116, 122, 4, 4, "#0ea5e9");
  }

  _drawPlant(x, y) {
    this._px(x, y + 6, 10, 8, "#c2410c");
    this._px(x - 1, y + 5, 12, 2, "#ea580c");
    this._px(x + 2, y, 6, 6, "#16a34a");
    this._px(x, y + 2, 4, 4, "#22c55e");
    this._px(x + 6, y + 1, 4, 4, "#15803d");
    this._px(x + 3, y - 2, 4, 3, "#4ade80");
  }

  _drawDoctorSprite(a) {
    const bob =
      a.state === "thinking"
        ? Math.round(Math.sin(this.time * 8) * 1)
        : a.state === "speaking" || a.state === "final"
        ? Math.round(Math.sin(this.time * 10) * 1.2)
        : Math.round(Math.sin(this.time * 2.5 + a.homeX) * 0.4);

    const x = Math.round(a.x);
    const y = Math.round(a.y + bob);

    // Highlight / Active ring
    if (this.selectedAgentId === a.id || this.hoveredAgentId === a.id) {
      this._px(x - 11, y + 5, 22, 7, a.accent);
      this._px(x - 9, y + 6, 18, 5, "#eff6ff");
    } else if (a.state === "thinking" || a.state === "speaking" || a.state === "final") {
      this._px(x - 10, y + 6, 20, 5, a.accent);
      this._px(x - 8, y + 7, 16, 3, "#eff6ff");
    }

    // Soft shadow under feet
    this._px(x - 7, y + 7, 14, 3, "rgba(15, 23, 42, 0.2)");

    // Chair backrest (slate blue)
    this._px(x - 7, y - 6, 14, 10, "#94a3b8");
    this._px(x - 6, y - 5, 12, 8, "#cbd5e1");

    // Shoes & Trousers
    this._px(x - 4, y + 5, 3, 3, "#1e293b");
    this._px(x + 1, y + 5, 3, 3, "#1e293b");
    this._px(x - 4, y + 2, 3, 3, "#475569");
    this._px(x + 1, y + 2, 3, 3, "#475569");

    // Crisp outline around white lab coat so it pops on light background
    this._px(x - 7, y - 9, 14, 12, "#475569");
    this._px(x - 6, y - 8, 12, 10, a.coat);
    // Inner colored scrubs / tie
    this._px(x - 2, y - 8, 4, 8, a.shirt);

    // Arms
    const armLift = a.state === "speaking" || a.state === "final" ? -2 : 0;
    this._px(x - 9, y - 7 + armLift, 3, 7, "#475569");
    this._px(x - 8, y - 7 + armLift, 2, 6, a.coat);
    this._px(x + 6, y - 7, 3, 7, "#475569");
    this._px(x + 6, y - 7, 2, 6, a.coat);
    this._px(x - 8, y - 1 + armLift, 2, 2, a.skin);
    this._px(x + 6, y - 1, 2, 2, a.skin);

    // Specialty props
    if (a.prop === "stethoscope") {
      this._px(x - 3, y - 7, 1, 5, "#334155");
      this._px(x + 2, y - 7, 1, 5, "#334155");
      this._px(x - 2, y - 2, 4, 1, "#64748b");
    } else if (a.prop === "clipboard") {
      this._px(x + 5, y - 5, 5, 6, "#92400e");
      this._px(x + 6, y - 4, 3, 4, "#fef3c7");
    }

    // Head outline + skin
    this._px(x - 6, y - 19, 12, 11, "#334155");
    this._px(x - 5, y - 18, 10, 9, a.skin);
    // Hair
    this._px(x - 5, y - 20, 10, 4, a.hair);
    this._px(x - 6, y - 18, 2, 4, a.hair);
    this._px(x + 4, y - 18, 2, 4, a.hair);

    if (a.prop === "cap") {
      this._px(x - 5, y - 21, 10, 3, "#0d9488");
      this._px(x - 4, y - 22, 8, 1, "#2dd4bf");
    }

    // Eyes
    const blink = Math.floor(this.time * 2 + a.homeX) % 13 === 0;
    if (!blink) {
      this._px(x - 3, y - 14, 2, 2, "#0f172a");
      this._px(x + 1, y - 14, 2, 2, "#0f172a");
    } else {
      this._px(x - 3, y - 13, 2, 1, "#0f172a");
      this._px(x + 1, y - 13, 2, 1, "#0f172a");
    }

    if (a.prop === "glasses") {
      this._px(x - 4, y - 15, 3, 1, "#334155");
      this._px(x + 1, y - 15, 3, 1, "#334155");
      this._px(x - 1, y - 14, 2, 1, "#334155");
    }

    // Mouth
    if (a.state === "speaking" || a.state === "final") {
      const open = Math.floor(this.time * 9) % 2 === 0;
      this._px(x - 1, y - 11, 3, open ? 2 : 1, "#be123c");
    } else {
      this._px(x - 1, y - 11, 2, 1, "#9a3412");
    }

    // Pixel Status Icon above head
    if (a.state === "thinking") {
      const dotIdx = Math.floor(this.time * 4) % 3;
      this._px(x - 7, y - 30, 14, 7, "#334155");
      this._px(x - 6, y - 29, 12, 5, "#ffffff");
      for (let i = 0; i < 3; i++) {
        this._px(x - 4 + i * 3, y - 27, 2, 2, i === dotIdx ? a.accent : "#cbd5e1");
      }
    } else if (a.state === "agree") {
      this._px(x - 5, y - 31, 10, 9, "#15803d");
      this._px(x - 4, y - 30, 8, 7, "#22c55e");
      this._px(x - 2, y - 26, 2, 2, "#ffffff");
      this._px(x, y - 28, 2, 3, "#ffffff");
    } else if (a.state === "disagree") {
      this._px(x - 5, y - 31, 10, 9, "#b91c1c");
      this._px(x - 4, y - 30, 8, 7, "#ef4444");
      this._px(x - 2, y - 29, 2, 2, "#ffffff");
      this._px(x + 1, y - 29, 2, 2, "#ffffff");
      this._px(x - 1, y - 27, 2, 2, "#ffffff");
      this._px(x - 2, y - 25, 2, 2, "#ffffff");
      this._px(x + 1, y - 25, 2, 2, "#ffffff");
    }
  }

  _drawPackets() {
    for (const p of this.packets) {
      const t = p.progress;
      const x = p.fromX + (p.toX - p.fromX) * t;
      const y = p.fromY + (p.toY - p.fromY) * t - Math.sin(t * Math.PI) * 18;
      this._px(x - 5, y - 4, 10, 8, "#0f172a");
      this._px(x - 4, y - 3, 8, 6, "#ffffff");
      this._px(x - 3, y - 2, 6, 4, p.color);
      this._px(x - 2, y - 1, 4, 1, "#ffffff");
    }
  }

  _drawHDOverlays() {
    const ctx = this.ctx;
    const sx = this.canvas.width / this.PW;
    const sy = this.canvas.height / this.PH;

    ctx.save();

    // 1. Memory Vault Label (Left Wall) — Pure Vietnamese
    ctx.font = "800 11px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = this.memoryActive ? "#0f766e" : "#64748b";
    ctx.fillText(
      this.memoryActive ? `KHO BỘ NHỚ (${this.memoryHitCount} mục)` : "BỘ NHỚ: TẮT",
      44 * sx,
      46 * sy - 4
    );

    // 2. Center FHIR Audit Board Crisp Text — Pure Vietnamese
    ctx.textAlign = "left";
    ctx.font = "800 12px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillStyle = "#0f172a";
    ctx.fillText("RÀ SOÁT FHIR R4", 118 * sx, 16 * sy);
    ctx.font = "700 11px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillStyle = "#475569";
    ctx.fillText(`VÒNG: ${this.currentRound}/${this.maxRounds}`, 110 * sx, 25 * sy);
    ctx.fillStyle = "#0284c7";
    ctx.fillText(`ĐỒNG THUẬN: ${this.consensusText}`, 110 * sx, 34 * sy);

    // Board Verdict Badge
    ctx.textAlign = "center";
    ctx.font = "800 12px -apple-system, BlinkMacSystemFont, sans-serif";
    const ansUpper = String(this.boardAnswer).toUpperCase();
    const isErr = ansUpper.startsWith("LỖI") || ansUpper.startsWith("ERR");
    const isValid = ansUpper === "HỢP LỆ" || ansUpper === "VALID";
    ctx.fillStyle = isErr ? "#b91c1c" : isValid ? "#15803d" : "#475569";
    ctx.fillText(String(this.boardAnswer).slice(0, 12), 193.5 * sx, 25 * sy);

    // 3. Agent Nameplates, Answer Badges, and High-Contrast Light Speech Bubbles
    for (const a of Object.values(this.agents)) {
      const cx = a.x * sx;
      const cy = a.y * sy;

      // Crisp Light Nameplate Pill
      const label = `${a.name} · ${a.role}`;
      ctx.font = "700 12px -apple-system, BlinkMacSystemFont, sans-serif";
      const tw = ctx.measureText(label).width + 16;
      const px = cx - tw / 2;
      const py = cy + 12 * sy;

      ctx.fillStyle = this.selectedAgentId === a.id ? a.accent : "#ffffff";
      this._roundRect(ctx, px, py, tw, 22, 6);
      ctx.fill();
      ctx.strokeStyle = a.accent;
      ctx.lineWidth = this.selectedAgentId === a.id ? 2.5 : 1.5;
      ctx.stroke();

      ctx.fillStyle = this.selectedAgentId === a.id ? "#ffffff" : "#0f172a";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, cx, py + 11);

      // Verdict / Answer Chip near head
      if (a.answer) {
        const isAgree = a.state === "agree";
        const isDis = a.state === "disagree";
        const ansUp = String(a.answer).toUpperCase();
        const chipBg = isAgree
          ? "#16a34a"
          : isDis
          ? "#dc2626"
          : ansUp.startsWith("LỖI") || ansUp.startsWith("ERR")
          ? "#dc2626"
          : ansUp === "HỢP LỆ" || ansUp === "VALID"
          ? "#16a34a"
          : a.accent;
        ctx.font = "800 11px ui-monospace, monospace";
        const bw = Math.max(38, ctx.measureText(a.answer).width + 12);
        const bx = cx + 14 * sx;
        const by = cy - 22 * sy;
        ctx.fillStyle = chipBg;
        this._roundRect(ctx, bx, by, bw, 19, 5);
        ctx.fill();
        ctx.fillStyle = "#ffffff";
        ctx.fillText(a.answer, bx + bw / 2, by + 9.5);
      }

      if (a.bubble) {
        this._drawLightSpeechBubble(ctx, a, cx, cy, sx, sy);
      }
    }

    ctx.restore();
  }

  _drawLightSpeechBubble(ctx, a, cx, cy, sx, sy) {
    const maxW = 225;
    ctx.font = "600 12px -apple-system, BlinkMacSystemFont, sans-serif";
    const words = a.bubble.split(/\s+/);
    const lines = [];
    let line = "";
    for (const w of words) {
      const test = line ? `${line} ${w}` : w;
      if (ctx.measureText(test).width > maxW - 18 && line) {
        lines.push(line);
        line = w;
        if (lines.length >= 2) break;
      } else {
        line = test;
      }
    }
    if (line && lines.length < 3) lines.push(line);

    const lineH = 16;
    const bw = Math.min(maxW, Math.max(...lines.map((l) => ctx.measureText(l).width)) + 20);
    const bh = lines.length * lineH + 13;

    let bx = cx - bw / 2;
    let by = cy - 32 * sy - bh;
    if (a.id === "meta") {
      bx = cx + 22 * sx;
      by = cy - 16 * sy;
    } else if (a.id === "doctor_1") {
      bx = Math.max(10, cx - bw * 0.3);
      by = cy - 25 * sy - bh;
    } else if (a.id === "doctor_3") {
      bx = Math.min(this.canvas.width - bw - 10, cx - bw * 0.7);
      by = cy - 25 * sy - bh;
    } else if (a.id === "doctor_2") {
      bx = cx - bw / 2;
      by = cy - 23 * sy - bh;
    }

    bx = Math.max(10, Math.min(this.canvas.width - bw - 10, bx));
    by = Math.max(10, Math.min(this.canvas.height - bh - 10, by));

    // Clean drop shadow + crisp white bubble for slides
    ctx.save();
    ctx.shadowColor = "rgba(15, 23, 42, 0.16)";
    ctx.shadowBlur = 8;
    ctx.shadowOffsetY = 3;
    ctx.fillStyle = "#ffffff";
    this._roundRect(ctx, bx, by, bw, bh, 8);
    ctx.fill();
    ctx.restore();

    ctx.strokeStyle =
      a.state === "agree"
        ? "#16a34a"
        : a.state === "disagree"
        ? "#dc2626"
        : a.accent;
    ctx.lineWidth = 2;
    this._roundRect(ctx, bx, by, bw, bh, 8);
    ctx.stroke();

    ctx.fillStyle = "#0f172a";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    lines.forEach((l, idx) => {
      ctx.fillText(l, bx + 10, by + 6 + idx * lineH);
    });
  }

  _roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  _update(dt) {
    this.time += dt;
    if (this.memoryPulse > 0) this.memoryPulse = Math.max(0, this.memoryPulse - dt);
    for (const a of Object.values(this.agents)) {
      a.x += (a.targetX - a.x) * Math.min(1, dt * 6);
      a.y += (a.targetY - a.y) * Math.min(1, dt * 6);
    }
    for (const p of this.packets) {
      p.progress += dt * 1.45;
    }
    this.packets = this.packets.filter((p) => p.progress < 1);
  }

  _loop(now) {
    if (!this._lastTime) this._lastTime = now;
    const dt = Math.min(0.05, (now - this._lastTime) / 1000);
    this._lastTime = now;

    this._update(dt);

    this.pctx.clearRect(0, 0, this.PW, this.PH);
    this._drawLightHospitalRoom();

    const sortedAgents = Object.values(this.agents).sort((a, b) => a.y - b.y);
    for (const a of sortedAgents) {
      this._drawDoctorSprite(a);
    }
    this._drawPackets();

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.imageSmoothingEnabled = false;
    this.ctx.drawImage(this.pxCanvas, 0, 0, this.canvas.width, this.canvas.height);

    this.ctx.imageSmoothingEnabled = true;
    this._drawHDOverlays();

    requestAnimationFrame(this._loop);
  }
}
