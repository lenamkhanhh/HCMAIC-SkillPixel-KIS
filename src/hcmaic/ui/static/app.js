"use strict";

const $ = (id) => document.getElementById(id);
const HISTORY_KEY = "hcmaic.queryHistory.v1";
const MAX_HISTORY = 20;

let lastQueryId = null;
let selectedFrameId = null;
let lastResultIds = [];
let queryRevision = 0;
const sessionId = "ui-" + Math.random().toString(36).slice(2);

function setStatus(text, isError = false, spinning = false) {
  const el = $("status");
  el.hidden = !text;
  el.className = isError ? "error" : "muted";
  el.innerHTML = spinning ? '<span class="spinner">&#9696;</span> ' + text : text;
}

async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON error */ }
  if (!res.ok) {
    const detail = body && body.detail ? JSON.stringify(body.detail) : res.statusText;
    throw new Error(`${res.status}: ${detail}`);
  }
  return body;
}

/* ---------- system info + video filter ---------- */

async function loadSystemInfo() {
  try {
    const [health, info] = await Promise.all([api("/health"), api("/system/info")]);
    $("sysinfo").textContent =
      `${health.index_size} frames · ${health.n_videos} videos · ` +
      `index ${health.index_version} · provider ${health.embedding_provider}`;
    $("sysinfo").textContent += ` | fusion ${info.runtime.fusion}`;
    const select = $("videoFilter");
    for (const vid of info.video_ids) {
      const opt = document.createElement("option");
      opt.value = vid;
      opt.textContent = vid;
      select.appendChild(opt);
    }
  } catch (err) {
    setStatus("Failed to load system info — " + err.message, true);
  }
}

/* ---------- search ---------- */

async function runSearch(text) {
  const query = (text ?? $("query").value).trim();
  if (!query) { setStatus("Enter a query first.", true); return; }
  $("query").value = query;
  setStatus("Searching…", false, true);
  $("results").innerHTML = "";
  hideDetail();
  const body = {
    text: query,
    task_type: $("taskType").value,
    top_k: parseInt($("topK").value, 10),
    filters: {},
  };
  const videoId = $("videoFilter").value;
  if (videoId) body.filters.video_ids = [videoId];
  try {
    const data = await api("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    lastQueryId = data.query_id;
    lastResultIds = data.results.map((result) => result.frame_id);
    queryRevision += 1;
    pushHistory(query);
    renderResults(data);
    setStatus(
      `${data.total_found} result(s) · ${data.latency_ms} ms · query ${data.query_id}` +
      (data.total_found === 0 ? " — no matches; broaden the query or clear the video filter." : "")
    );
  } catch (err) {
    setStatus("Search failed — " + err.message, true);
  }
}

function renderResults(data) {
  const grid = $("results");
  grid.innerHTML = "";
  for (const r of data.results) {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.frameId = r.frame_id;
    card.innerHTML = `
      <img loading="lazy" src="${r.image_url}" alt="${r.frame_id}">
      <div class="meta">
        <span class="rank">#${r.rank}</span>
        <span class="score">${r.final_score.toFixed(4)}</span><br>
        <b>${r.video_id}</b> · ${r.frame_id.split(":")[1]}<br>
        <span class="muted">${(r.timestamp_ms / 1000).toFixed(2)}s · frame ${r.frame_idx}</span>
      </div>`;
    const breakdown = document.createElement("div");
    breakdown.className = "muted";
    breakdown.textContent = Object.entries(r.signal_scores)
      .map(([name, score]) => `${name}:${Number(score).toFixed(3)}`)
      .join(" | ");
    card.querySelector(".meta").appendChild(breakdown);
    card.addEventListener("click", () => openDetail(r.frame_id));
    grid.appendChild(card);
  }
}

/* ---------- detail + timeline ---------- */

async function openDetail(frameId) {
  try {
    setStatus("Loading frame…", false, true);
    const data = await api(`/frames/${encodeURIComponent(frameId)}?window=5`);
    selectedFrameId = frameId;
    for (const card of document.querySelectorAll(".card")) {
      card.classList.toggle("selected", card.dataset.frameId === frameId);
    }
    const f = data.frame;
    $("detailTitle").textContent = f.frame_id;
    $("detailImage").src = data.image_url;
    const meta = $("detailMeta");
    meta.innerHTML = "";
    const rows = {
      video: f.video_id,
      keyframe: f.keyframe_id,
      "frame idx": f.frame_idx,
      time: (f.timestamp_ms / 1000).toFixed(3) + " s",
      "timestamp source": f.metadata.timestamp_source || "legacy mapping",
      shot: f.shot_id || "none",
      title: f.metadata.title || "—",
    };
    for (const [k, v] of Object.entries(rows)) {
      const term = document.createElement("dt");
      const description = document.createElement("dd");
      term.textContent = k;
      description.textContent = String(v);
      meta.append(term, description);
    }
    const strip = $("neighbors");
    strip.innerHTML = "";
    for (const n of data.neighbors) {
      const thumb = document.createElement("div");
      thumb.className = "thumb" + (n.is_current ? " current" : "");
      thumb.innerHTML =
        `<img loading="lazy" src="${n.image_url}" alt="${n.frame_id}">` +
        `${(n.timestamp_ms / 1000).toFixed(1)}s`;
      thumb.addEventListener("click", () => openDetail(n.frame_id));
      strip.appendChild(thumb);
    }
    $("submissionPreview").textContent =
      "No preview yet. This is a canonical preview only — nothing is submitted anywhere.";
    $("detail").hidden = false;
    setStatus("");
  } catch (err) {
    setStatus("Frame load failed — " + err.message, true);
  }
}

function hideDetail() {
  $("detail").hidden = true;
  selectedFrameId = null;
}

async function recordFeedback(kind) {
  if (!selectedFrameId) return;
  const body = {
    session_id: sessionId,
    query_revision: Math.max(1, queryRevision),
    positive_ids: kind === "positive" ? [selectedFrameId] : [],
    negative_ids: kind === "negative" ? [selectedFrameId] : [],
    prior_result_ids: lastResultIds,
  };
  try {
    const data = await api("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("feedbackStatus").textContent =
      `${kind} recorded locally (#${data.record_count})`;
  } catch (err) {
    $("feedbackStatus").textContent = "Feedback failed: " + err.message;
  }
}

/* ---------- submission preview ---------- */

async function previewSubmission() {
  if (!selectedFrameId) return;
  try {
    const body = {
      query_id: lastQueryId || "manual",
      task_type: $("taskType").value,
      frame_id: selectedFrameId,
    };
    const answer = $("answer").value.trim();
    if (answer) body.answer = answer;
    const data = await api("/submit/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("submissionPreview").textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    $("submissionPreview").textContent = "Preview failed — " + err.message;
  }
}

/* ---------- query history (localStorage) ---------- */

function getHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch { return []; }
}

function pushHistory(query) {
  let history = getHistory().filter((q) => q !== query);
  history.unshift(query);
  history = history.slice(0, MAX_HISTORY);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  renderHistory();
}

function renderHistory() {
  const list = $("history");
  list.innerHTML = "";
  for (const q of getHistory()) {
    const li = document.createElement("li");
    li.textContent = q;
    li.title = "Run this query again";
    li.addEventListener("click", () => runSearch(q));
    list.appendChild(li);
  }
}

/* ---------- wiring ---------- */

$("searchBtn").addEventListener("click", () => runSearch());
$("query").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });
$("closeDetail").addEventListener("click", hideDetail);
$("previewBtn").addEventListener("click", previewSubmission);
$("positiveBtn").addEventListener("click", () => recordFeedback("positive"));
$("negativeBtn").addEventListener("click", () => recordFeedback("negative"));
$("clearHistory").addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});

renderHistory();
loadSystemInfo();
