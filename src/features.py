"""
features.py
===========
All feature extraction layers for the phishing detector.

URL features  → always available, zero-cost
WHOIS         → domain age (network, ~1–3 s)
DNS           → MX / A / NS records (network, ~0.5 s)
SSL           → cert validity + days remaining (network, ~1 s)
HTML/JS       → page-level signals (network + parse, ~3–5 s)

Each layer degrades gracefully if its dependency is missing.
"""

from __future__ import annotations

import math
import re
import socket
import ssl
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np

# ── Optional heavy imports ──────────────────────────────────────
try:
    import whois as _whois          # python-whois
    _WHOIS = True
except ImportError:
    _WHOIS = False

try:
    import dns.resolver as _dns     # dnspython
    _DNS = True
except ImportError:
    _DNS = False

try:
    import requests as _req
    from bs4 import BeautifulSoup
    _HTML = True
except ImportError:
    _HTML = False

# ── Constants ───────────────────────────────────────────────────
SUSPICIOUS_WORDS = frozenset([
    "login", "signin", "verify", "secure", "update", "account",
    "confirm", "bank", "paypal", "password", "credential", "ebay",
    "amazon", "microsoft", "apple", "google", "support", "service",
    "billing", "payment", "wallet", "alert", "urgent", "suspended",
])

URL_SHORTENERS = frozenset([
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "rebrand.ly",
    "buff.ly", "adf.ly", "is.gd", "cli.gs", "tr.im", "snipurl.com",
])

# Official corporate domains mapped to their legal root infrastructure suffixes
BRAND_DOMAINS: Dict[str, List[str]] = {
    "paypal": ["paypal.com", "paypal.co.uk"],
    "amazon": ["amazon.com", "amazon.co.uk", "amazon.in", "amazon.de"],
    "google": ["google.com", "google.co.in", "google.ca", "google.ch"],
    "hdfc": ["hdfcbank.com", "hdfc.co.in", "hdfc.bank.in"],
    "linkedin": ["linkedin.com"],
}

# ── URL feature schema (order defines numpy vector) ─────────────
URL_FEATURE_NAMES: List[str] = [
    "url_length", "domain_length", "path_length",
    "num_dots", "num_hyphens", "num_underscores",
    "num_slashes", "num_at", "num_percent", "num_digits",
    "num_params", "has_https", "has_ip", "has_at_symbol",
    "subdomain_level", "prefix_suffix", "has_shortener",
    "double_slash", "entropy", "path_depth", "has_suspicious_words",
    "has_brand_impersonation",
]

# ── Character vocabulary for DL models ──────────────────────────
_CHARSET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    r".-_~:/?#[]@!$&'()*+,;=%"
)
CHAR2IDX: Dict[str, int] = {c: i + 2 for i, c in enumerate(_CHARSET)}
# 0 = PAD, 1 = UNK
VOCAB_SIZE: int = len(_CHARSET) + 2
MAX_URL_LEN: int = 200


# ────────────────────────────────────────────────────────────────
# URL features
# ────────────────────────────────────────────────────────────────

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _is_ip(domain: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", domain))


def extract_url_features(url: str) -> Dict[str, Any]:
    """Return a dict of 21 URL-derived features (no network calls)."""
    url = str(url)
    try:
        _url = url if url.startswith(("http://", "https://")) else "http://" + url
        parsed = urlparse(_url)
        domain = parsed.netloc or ""
        path   = parsed.path   or ""
        scheme = parsed.scheme or ""
    except Exception:
        domain = path = scheme = ""

# Brand Impersonation Check Heuristic
    domain_lower = domain.lower()
    impersonation_flag = 0
    
    for brand, valid_roots in BRAND_DOMAINS.items():
        if brand in domain_lower:
            # Check if it fails to end with or neatly transition from an official root domain path
            is_valid = any(
                domain_lower.endswith(root) or f"{root}." in domain_lower 
                for root in valid_roots
            )
            if not is_valid:
                impersonation_flag = 1
                break

    return {
        "url_length":           len(url),
        "domain_length":        len(domain),
        "path_length":          len(path),
        "num_dots":             url.count("."),
        "num_hyphens":          url.count("-"),
        "num_underscores":      url.count("_"),
        "num_slashes":          url.count("/"),
        "num_at":               url.count("@"),
        "num_percent":          url.count("%"),
        "num_digits":           sum(c.isdigit() for c in url),
        "num_params":           url.count("?") + url.count("=") + url.count("&"),
        "has_https":            int(scheme == "https"),
        "has_ip":               int(_is_ip(domain)),
        "has_at_symbol":        int("@" in url),
        "subdomain_level":      max(0, len(domain.split(".")) - 2),
        "prefix_suffix":        int("-" in domain),
        "has_shortener":        int(any(s in domain for s in URL_SHORTENERS)),
        "double_slash":         int("//" in url[7:] if len(url) > 7 else False),
        "entropy":              round(_entropy(url), 4),
        "path_depth":           path.count("/"),
        "has_suspicious_words": int(any(w in url.lower() for w in SUSPICIOUS_WORDS)),
        "has_brand_impersonation": impersonation_flag,
    }


def url_feature_vector(url: str) -> np.ndarray:
    """Return shape (21,) float32 numpy array."""
    feat = extract_url_features(url)
    return np.array([feat[k] for k in URL_FEATURE_NAMES], dtype=np.float32)


# ────────────────────────────────────────────────────────────────
# WHOIS – domain age
# ────────────────────────────────────────────────────────────────

def get_domain_age_days(domain: str) -> int:
    """Days since domain registration, or -1 on failure."""
    if not _WHOIS:
        return -1
    try:
        w = _whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation is None:
            return -1
        now = datetime.now(timezone.utc)
        if creation.tzinfo is None:
            creation = creation.replace(tzinfo=timezone.utc)
        return max(0, (now - creation).days)
    except Exception:
        return -1


# ────────────────────────────────────────────────────────────────
# DNS records
# ────────────────────────────────────────────────────────────────

def get_dns_features(domain: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "has_mx":   False,
        "has_a":    False,
        "has_aaaa": False,
        "has_ns":   False,
        "num_ns":   0,
    }
    if not _DNS:
        return base
    for rtype, key in [("MX", "has_mx"), ("A", "has_a"), ("AAAA", "has_aaaa"), ("NS", "has_ns")]:
        try:
            ans = _dns.resolve(domain, rtype, lifetime=3.0)
            base[key] = True
            if rtype == "NS":
                base["num_ns"] = len(list(ans))
        except Exception:
            pass
    return base


# ────────────────────────────────────────────────────────────────
# SSL certificate
# ────────────────────────────────────────────────────────────────

def get_ssl_features(domain: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "ssl_valid":     False,
        "ssl_days_left": -1,
        "ssl_org_match": False,
    }
    try:
        ctx = ssl.create_default_context()
        conn = socket.create_connection((domain, 443), timeout=3)
        with ctx.wrap_socket(conn, server_hostname=domain) as ssock:
            cert = ssock.getpeercert()
        base["ssl_valid"] = True
        not_after = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z")
        not_after = not_after.replace(tzinfo=timezone.utc)
        base["ssl_days_left"] = max(0, (not_after - datetime.now(timezone.utc)).days)
        # Check SAN / CN coverage
        sans = [v for _, v in cert.get("subjectAltName", [])]
        cns  = [v for tup in cert.get("subject", []) for k, v in [tup] if k == "commonName"]
        all_names = sans + cns
        base["ssl_org_match"] = any(
            domain.endswith(n.lstrip("*").lstrip(".")) for n in all_names
        )
    except Exception:
        pass
    return base


# ────────────────────────────────────────────────────────────────
# HTML / JS page features
# ────────────────────────────────────────────────────────────────

def get_html_features(url: str, timeout: int = 5) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "num_external_links": 0,
        "num_forms":          0,
        "num_iframes":        0,
        "has_password_field": False,
        "num_scripts":        0,
        "page_title_len":     0,
        "has_favicon":        False,
        "redirect_count":     0,
    }
    if not _HTML:
        return base
    try:
        resp = _req.get(
            url, timeout=timeout, allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PhishBot/1.0)"},
        )
        base["redirect_count"] = len(resp.history)
        soup = BeautifulSoup(resp.text, "lxml")
        parsed_base = urlparse(url).netloc

        external = [
            a["href"] for a in soup.find_all("a", href=True)
            if a["href"].startswith("http") and parsed_base not in a["href"]
        ]
        base["num_external_links"] = len(external)
        base["num_forms"]          = len(soup.find_all("form"))
        base["num_iframes"]        = len(soup.find_all("iframe"))
        base["has_password_field"] = bool(soup.find("input", {"type": "password"}))
        base["num_scripts"]        = len(soup.find_all("script"))
        title = soup.find("title")
        base["page_title_len"]     = len(title.get_text()) if title else 0
        base["has_favicon"]        = bool(
            soup.find("link", rel=lambda r: r and "icon" in r)
        )
    except Exception:
        pass
    return base

# ────────────────────────────────────────────────────────────────
# impersonation heuristic feature
# ────────────────────────────────────────────────────────────────

def check_brand_impersonation(url: str) -> float:
    url_lower = url.lower()
    target_brands = ["paypal", "amazon", "google", "hdfc", "linkedin"]
    
    for brand in target_brands:
        # If the brand name is found anywhere in the string
        if brand in url_lower:
            # But the domain doesn't end with the legitimate corporate root
            if not (url_lower.endswith(f"{brand}.com") or f"{brand}.com/" in url_lower or
                    url_lower.endswith(f"{brand}.co.in") or f"{brand}.co.in/" in url_lower or
                    url_lower.endswith(f"{brand}.bank.in") or f"{brand}.bank.in/" in url_lower):
                return 1.0  # High alert: Brand string found on a non-brand host!
    return 0.0

# ────────────────────────────────────────────────────────────────
# Aggregated metadata (used by API for rich response)
# ────────────────────────────────────────────────────────────────

def get_metadata(url: str, fetch_html: bool = False) -> Dict[str, Any]:
    """Collect all non-ML metadata for display purposes."""
    _url = url if url.startswith(("http://", "https://")) else "http://" + url
    domain = urlparse(_url).netloc.split(":")[0]

    meta: Dict[str, Any] = {"domain": domain}
    meta["domain_age_days"] = get_domain_age_days(domain)
    meta.update(get_dns_features(domain))
    meta.update(get_ssl_features(domain))
    if fetch_html:
        meta.update(get_html_features(url))
    return meta


# ────────────────────────────────────────────────────────────────
# Trust-signal aggregation (for conflict arbitration in fusion)
# ────────────────────────────────────────────────────────────────
def _is_shortener_domain(domain: str) -> bool:
    """Exact-match shortener check (avoids substring false positives like reddit.com matching t.co)."""
    bare = domain.lower().lstrip("www.")
    return bare in URL_SHORTENERS


def get_trust_signals(url: str) -> Dict[str, Any]:
    """
    Collect arbitration-relevant signals from existing extractors.
    Used by the adaptive fusion engine to break ML-vs-DL ties.
    All sub-calls degrade gracefully if network is unavailable.
    """
    url_feats = extract_url_features(url)
    _url = url if url.startswith(("http://", "https://")) else "http://" + url
    domain = urlparse(_url).netloc.split(":")[0]

    # Domain trust signals (network calls, each with internal timeouts)
    age_days = get_domain_age_days(domain)
    ssl_info = get_ssl_features(domain)
    dns_info = get_dns_features(domain)

    return {
        # URL-derived (instant, no network)
        "brand_impersonation": int(url_feats.get("has_brand_impersonation", 0)),
        "has_shortener":       int(url_feats.get("has_shortener", 0)),
        "is_shortener_domain": int(_is_shortener_domain(domain)),  # exact match, not substring
        "has_ip":              int(url_feats.get("has_ip", 0)),
        "has_suspicious_words": int(url_feats.get("has_suspicious_words", 0)),
        "entropy":             float(url_feats.get("entropy", 0.0)),
        # Network-derived (may be -1 / False on failure)
        "domain_age_days":     age_days,
        "ssl_valid":           bool(ssl_info.get("ssl_valid", False)),
        "ssl_days_left":       int(ssl_info.get("ssl_days_left", -1)),
        "has_dns":             bool(dns_info.get("has_a", False) or dns_info.get("has_ns", False)),
        "has_mx":              bool(dns_info.get("has_mx", False)),
    }


# ────────────────────────────────────────────────────────────────
# Character tokenisation for DL models
# ────────────────────────────────────────────────────────────────

def url_to_ids(url: str, max_len: int = MAX_URL_LEN) -> List[int]:
    """Encode a URL to a fixed-length integer sequence (pad=0, unk=1)."""
    ids = [CHAR2IDX.get(c, 1) for c in str(url)[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids
