// background.js  — Manifest V3 Service Worker
// Listens for popup requests, calls /predict, caches results.

const API_BASE = "http://localhost:8000";   // ← change for production
const CACHE_TTL_MS = 5 * 60 * 1000;        // 5 minutes

// ── In-memory cache (survives until SW dies) ─────────────────────
const _cache = new Map();

function _cacheKey(url) {
  try {
    const u = new URL(url);
    return u.origin + u.pathname;           // ignore query params
  } catch {
    return url;
  }
}

async function queryAPI(url) {
  const key   = _cacheKey(url);
  const now   = Date.now();
  const entry = _cache.get(key);
  if (entry && now - entry.ts < CACHE_TTL_MS) {
    return { ...entry.data, cached: true };
  }

  const resp = await fetch(`${API_BASE}/predict`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({
      url,
      include_shap: false,  // keep popup fast
      fetch_html:   false,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API ${resp.status}: ${text}`);
  }

  const data = await resp.json();
  _cache.set(key, { ts: now, data });
  return data;
}

// ── Message handler (from popup.js) ─────────────────────────────
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type !== "PREDICT") return false;

  queryAPI(msg.url)
    .then(data => sendResponse({ ok: true,  data }))
    .catch(err  => sendResponse({ ok: false, error: err.message }));

  return true;   // keep message channel open for async response
});

// ── Badge colour on tab update ───────────────────────────────────
chrome.tabs.onActivated.addListener(({ tabId }) => {
  chrome.tabs.get(tabId, tab => {
    if (!tab?.url || !tab.url.startsWith("http")) {
      chrome.action.setBadgeText({ text: "", tabId });
      return;
    }
    const key   = _cacheKey(tab.url);
    const entry = _cache.get(key);
    if (entry && Date.now() - entry.ts < CACHE_TTL_MS) {
      _updateBadge(tabId, entry.data);
    }
  });
});

function _updateBadge(tabId, data) {
  const isPhishing = data.is_phishing;
  chrome.action.setBadgeText({
    text:  isPhishing ? "⚠" : "✓",
    tabId,
  });
  chrome.action.setBadgeBackgroundColor({
    color: isPhishing ? "#ef4444" : "#22c55e",
    tabId,
  });
}
