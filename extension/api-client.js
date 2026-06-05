// api-client.js — API communication layer for PhishGuard popup
//
// Sends prediction requests to the background service worker
// via chrome.runtime.sendMessage.

/**
 * Request a phishing prediction for the given URL.
 * Routes through the background worker (which handles caching & API calls).
 *
 * @param {string} url - The URL to classify.
 * @returns {Promise<{ok: boolean, data?: object, error?: string}>}
 */
function requestPrediction(url) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: "PREDICT", url }, (response) => {
      if (chrome.runtime.lastError) {
        resolve({
          ok: false,
          error: "Background service unavailable: " + chrome.runtime.lastError.message,
        });
        return;
      }
      resolve(response || { ok: false, error: "No response from background worker." });
    });
  });
}

/**
 * Get the current active tab's URL.
 * @returns {Promise<string|null>}
 */
async function getActiveTabUrl() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab?.url;
    if (!url || (!url.startsWith("http://") && !url.startsWith("https://"))) {
      return null;
    }
    return url;
  } catch {
    return null;
  }
}
