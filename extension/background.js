importScripts('config.js');

// background.js  — Manifest V3 Service Worker
// Listens for popup requests, calls /predict, caches results.

// ── Ephemeral Session Cache (survives SW termination) ─────────────────────

async function _getCached(key) {
  const data = await chrome.storage.session.get(key);
  return data[key];
}

async function _setCached(key, value) {
  await chrome.storage.session.set({ [key]: value });
}

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
  const entry = await _getCached(key);
  
  if (entry && now - entry.ts < PHISHGUARD_CONFIG.CACHE_TTL_MS) {
    return { ...entry.data, cached: true };
  }

  let screenshot_b64 = null;
  try {
    // Only capture if active tab matches the URL to avoid capturing the wrong tab
    const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (activeTab && activeTab.url === url) {
      screenshot_b64 = await chrome.tabs.captureVisibleTab(null, { format: 'jpeg', quality: 20 });
    }
  } catch (err) {
    console.warn("Could not capture screenshot:", err);
  }

  const resp = await fetch(`${PHISHGUARD_CONFIG.API_BASE}/predict`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({
      url,
      include_shap: false,  // keep popup fast
      fetch_html:   false,
      screenshot_b64: screenshot_b64
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API ${resp.status}: ${text}`);
  }

  const data = await resp.json();
  await _setCached(key, { ts: now, data });
  return data;
}

// ── Event Listeners (Wakes up ephemeral worker) ─────────────────────────

chrome.webNavigation.onCompleted.addListener(async (details) => {
  // Only process main frame navigations (not iframes)
  if (details.frameId === 0 && details.url.startsWith("http")) {
    try {
      const data = await queryAPI(details.url);
      _updateBadge(details.tabId, data);
    } catch (err) {
      console.error("WebNavigation background check failed:", err);
    }
  }
});

// ── Message handler (from popup.js) ─────────────────────────────
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type !== "PREDICT") return false;

  queryAPI(msg.url)
    .then(data => sendResponse({ ok: true,  data }))
    .catch(err  => sendResponse({ ok: false, error: err.message }));

  return true;   // keep message channel open for async response
});

// ── Badge colour on tab update ───────────────────────────────────
chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  try {
    const tab = await chrome.tabs.get(tabId);
    if (!tab?.url || !tab.url.startsWith("http")) {
      chrome.action.setBadgeText({ text: "", tabId });
      return;
    }
    const key   = _cacheKey(tab.url);
    const entry = await _getCached(key);
    if (entry && Date.now() - entry.ts < PHISHGUARD_CONFIG.CACHE_TTL_MS) {
      _updateBadge(tabId, entry.data);
    }
  } catch (err) {
    console.error("Tab activation check failed:", err);
  }
});

function _updateBadge(tabId, data) {
  const label = data.label || (data.is_phishing ? "phishing" : "safe");
  const badge = PHISHGUARD_CONFIG.BADGE_MAP[label] || PHISHGUARD_CONFIG.BADGE_MAP.safe;
  chrome.action.setBadgeText({
    text:  badge.text,
    tabId,
  });
  chrome.action.setBadgeBackgroundColor({
    color: badge.color,
    tabId,
  });
}
