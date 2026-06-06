// content.js
// Runs in the context of the web page. Monitors DOM changes with a debounce
// to prevent spamming the background script and ML backend.

let debounceTimer = null;
const DEBOUNCE_MS = 1500; // Wait 1.5s after the last DOM mutation

// This function communicates with the ML model via the background script.
// Strict async/await ensures the browser's main UI thread never blocks.
async function analyzePageSettled() {
  try {
    // Collect relevant page info (URL, minimal DOM content if needed)
    const pageData = {
      url: window.location.href,
      // html: document.documentElement.outerHTML // omitted to save bandwidth unless needed
    };

    // Send to background SW (which forwards to the FastAPI backend)
    const response = await chrome.runtime.sendMessage({
      type: "PREDICT",
      url: pageData.url
    });

    if (response && response.ok && response.data) {
      const data = response.data;
      if (data.is_phishing || data.label === "phishing") {
        console.warn("🛡️ PhishGuard: Phishing signals detected on this page!", data);
        // Optional: Render an invisible-until-warning UI element here.
        showWarningBanner(data);
      } else {
        console.log("🛡️ PhishGuard: Page analyzed and looks safe.");
      }
    }
  } catch (err) {
    console.error("🛡️ PhishGuard Analysis Error:", err);
  }
}

// Minimal, non-blocking UI for warnings
function showWarningBanner(data) {
  if (document.getElementById("phishguard-warning")) return;
  const banner = document.createElement("div");
  banner.id = "phishguard-warning";
  banner.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; z-index: 2147483647;
    background: #d32f2f; color: white; text-align: center; padding: 12px;
    font-family: sans-serif; font-weight: bold; font-size: 16px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
  `;
  banner.innerHTML = `⚠️ WARNING: This site has been flagged as suspicious/phishing by PhishGuard (${Math.round(data.confidence * 100)}% confidence). Proceed with extreme caution! <button id="pg-dismiss" style="margin-left:15px; padding: 4px 8px; cursor: pointer;">Dismiss</button>`;
  document.body.prepend(banner);
  document.getElementById("pg-dismiss").onclick = () => banner.remove();
}

// Observe DOM mutations to trigger analysis only when the page settles
const observer = new MutationObserver(() => {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(analyzePageSettled, DEBOUNCE_MS);
});

// Start observing the body for changes
if (document.body) {
  observer.observe(document.body, { childList: true, subtree: true });
} else {
  // If body isn't ready, wait for DOMContentLoaded
  document.addEventListener('DOMContentLoaded', () => {
    observer.observe(document.body, { childList: true, subtree: true });
    // Trigger initial scan
    analyzePageSettled();
  });
}
