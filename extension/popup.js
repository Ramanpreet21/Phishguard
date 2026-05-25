// popup.js
// Runs inside popup.html – queries current tab, calls background worker,
// renders the full verdict + model votes + features + metadata.

const $ = id => document.getElementById(id);

// ── Helpers ─────────────────────────────────────────────────────

function show(id)  { $(id).classList.remove("hidden"); }
function hide(id)  { $(id).classList.add("hidden"); }
function cls(el, ...c) { c.forEach(k => el.classList.add(k)); }

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ── Render ───────────────────────────────────────────────────────

function renderResult(data, url) {
  hide("loading-state");
  hide("error-state");
  show("result");

  const isPhish = data.is_phishing;
  const conf    = data.confidence;
  const confPct = Math.round(conf * 100);
  const tone    = isPhish ? "danger" : "safe";

  // URL
  $("url-display").textContent = url.length > 55 ? url.slice(0, 52) + "…" : url;

  // Verdict card
  const vc = $("verdict-card");
  vc.className = `verdict-card ${tone}`;
  $("verdict-icon").textContent  = isPhish ? "🚨" : "✅";
  $("verdict-label").textContent = isPhish ? "Phishing" : "Safe";
  $("verdict-sub").textContent   = isPhish
    ? "This page may be attempting to steal credentials."
    : "No phishing signals detected.";
  const cp = $("conf-pct");
  cp.textContent  = confPct + "%";
  cp.className    = `confidence-pct ${tone}`;

  // Bar  (fill = how phishy it is, so full = 100% phishing confidence)
  const bar = $("conf-bar");
  bar.style.width  = confPct + "%";
  bar.className    = `conf-bar-fill ${tone}`;

  // ── Model votes ──────────────────────────────────────────────
  const grid = $("votes-grid");
  grid.innerHTML = "";
  const MODEL_LABELS = { rf:"RF", xgb:"XGB", svm:"SVM", lstm:"LSTM", cnn:"CNN", transformer:"Tfmr" };
  for (const [key, vote] of Object.entries(data.model_votes || {})) {
    const t   = vote.label === "phishing" ? "danger" : "safe";
    const pct = Math.round(vote.confidence * 100);
    grid.insertAdjacentHTML("beforeend", `
      <div class="vote-cell">
        <div class="vote-name">${esc(MODEL_LABELS[key] || key)}</div>
        <div class="vote-badge ${t}">${esc(vote.label.toUpperCase())}</div>
        <div class="vote-conf">${pct}%</div>
      </div>
    `);
  }

  // ── Top features ─────────────────────────────────────────────
  const fl   = $("feat-list");
  fl.innerHTML = "";
  const feats = (data.top_features || []).slice(0, 6);
  const maxImp = Math.max(...feats.map(f => f.importance), 1e-9);
  for (const f of feats) {
    const barW = Math.round((f.importance / maxImp) * 100);
    const name = f.feature.replace(/_/g, " ");
    fl.insertAdjacentHTML("beforeend", `
      <div class="feature-row">
        <span class="feat-name">${esc(name)}</span>
        <span class="feat-val">${f.value}</span>
        <div class="feat-bar-wrap">
          <div class="feat-bar-bg">
            <div class="feat-bar-fill" style="width:${barW}%"></div>
          </div>
        </div>
      </div>
    `);
  }

  // ── Domain intelligence ──────────────────────────────────────
  const mg   = $("meta-grid");
  const meta = data.metadata || {};
  mg.innerHTML = "";

  function metaItem(key, val, cssClass = "") {
    mg.insertAdjacentHTML("beforeend", `
      <div class="meta-item">
        <div class="meta-key">${esc(key)}</div>
        <div class="meta-val ${cssClass}">${esc(val)}</div>
      </div>
    `);
  }

  // Domain age
  const age = meta.domain_age_days;
  if (age !== undefined) {
    const ageStr = age < 0 ? "Unknown" : age < 30 ? `${age}d ⚠` : `${age}d`;
    const ageCls = age < 0 ? "" : age < 30 ? "warn" : "ok";
    metaItem("Domain Age", ageStr, ageCls);
  }

  // SSL
  if (meta.ssl_valid !== undefined) {
    metaItem("SSL", meta.ssl_valid ? `Valid (${meta.ssl_days_left}d)` : "Invalid / Missing",
             meta.ssl_valid ? "ok" : "bad");
    if (meta.ssl_valid) {
      metaItem("SSL Match", meta.ssl_org_match ? "Matches domain" : "Mismatch ⚠",
               meta.ssl_org_match ? "ok" : "warn");
    }
  }

  // DNS
  if (meta.has_mx !== undefined) {
    metaItem("MX Record",  meta.has_mx   ? "Present" : "Missing", meta.has_mx  ? "ok" : "warn");
    metaItem("A Record",   meta.has_a    ? "Present" : "Missing", meta.has_a   ? "ok" : "warn");
    metaItem("NS Count",   meta.num_ns   != null ? String(meta.num_ns) : "—");
  }

  if (!mg.children.length) {
    $("meta-section").classList.add("hidden");
  }

  // Latency
  $("latency-label").textContent = `${data.latency_ms} ms`;
}

function renderError(msg) {
  hide("loading-state");
  hide("result");
  show("error-state");
  $("error-msg").textContent = msg;
}

// ── Main flow ────────────────────────────────────────────────────

async function runScan() {
  show("loading-state");
  hide("result");
  hide("error-state");

  let tabUrl;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    tabUrl = tab?.url;
    if (!tabUrl || (!tabUrl.startsWith("http://") && !tabUrl.startsWith("https://"))) {
      renderError("No scannable URL on this page.");
      return;
    }
  } catch (e) {
    renderError("Could not read the current tab URL.");
    return;
  }

  chrome.runtime.sendMessage({ type: "PREDICT", url: tabUrl }, response => {
    if (chrome.runtime.lastError) {
      renderError("Background service unavailable: " + chrome.runtime.lastError.message);
      return;
    }
    if (!response?.ok) {
      renderError(response?.error || "Unknown API error.");
      return;
    }
    renderResult(response.data, tabUrl);
  });
}

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  runScan();
  $("rescan-btn").addEventListener("click", runScan);
});
