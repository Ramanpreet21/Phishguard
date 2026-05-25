"""
benchmark.py
============
Measures API latency at various concurrency levels.

Metrics reported:
  • Average response time
  • P50 / P95 / P99 latency
  • Worst-case latency
  • Requests per second (throughput)
  • Error rate

Usage:
  # Basic (single-threaded, 100 requests)
  python benchmark.py

  # Custom
  python benchmark.py --url http://localhost:8000 \
                      --requests 500              \
                      --concurrency 10            \
                      --warmup 10
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

# ── Test URLs (mix of safe + phishing-like) ──────────────────────
TEST_URLS = [
    "https://www.google.com",
    "https://www.github.com",
    "http://login-verify-paypal.com/update?account=true",
    "http://192.168.1.1/phishing/steal-creds",
    "https://mail.google.com/mail/u/0/#inbox",
    "http://secure-banking-update.tk/login?redirect=true",
    "https://docs.python.org/3/library/urllib.html",
    "http://bit.ly/free-iphone",
    "https://www.amazon.com/dp/B0001234",
    "http://amazon-verify-account-login.net/secure",
]


def call_predict(base_url: str, url: str) -> Dict[str, Any]:
    """Fire one /predict request and return timing + result."""
    payload = json.dumps({"url": url, "include_shap": False}).encode()
    req     = Request(
        f"{base_url}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=30) as resp:
            body       = json.loads(resp.read())
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "ok":         True,
                "latency_ms": latency_ms,
                "label":      body.get("label"),
                "confidence": body.get("confidence"),
                "url":        url,
            }
    except (URLError, Exception) as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {"ok": False, "latency_ms": latency_ms, "error": str(e), "url": url}


def run_benchmark(
    base_url: str,
    n_requests: int,
    concurrency: int,
    warmup: int,
) -> None:
    print(f"\n{'='*62}")
    print(f"  PHISHING DETECTOR — LATENCY BENCHMARK")
    print(f"{'='*62}")
    print(f"  Target      : {base_url}")
    print(f"  Requests    : {n_requests}  (warmup: {warmup})")
    print(f"  Concurrency : {concurrency}")
    print()

    # ── Warmup ───────────────────────────────────────────────────
    if warmup > 0:
        print(f"  Running {warmup} warmup requests…", end=" ", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            wfuts = [
                ex.submit(call_predict, base_url, TEST_URLS[i % len(TEST_URLS)])
                for i in range(warmup)
            ]
            concurrent.futures.wait(wfuts)
        print("done")

    # ── Benchmark ─────────────────────────────────────────────────
    print(f"  Benchmarking {n_requests} requests…")
    results: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(call_predict, base_url, TEST_URLS[i % len(TEST_URLS)])
            for i in range(n_requests)
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            results.append(fut.result())
            if i % max(1, n_requests // 10) == 0:
                print(f"    … {i}/{n_requests}", flush=True)

    elapsed_s = time.perf_counter() - t_start

    # ── Stats ─────────────────────────────────────────────────────
    ok_results   = [r for r in results if r["ok"]]
    fail_results = [r for r in results if not r["ok"]]
    latencies    = sorted(r["latency_ms"] for r in ok_results)

    def pct(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        k = min(int(p / 100 * len(data)), len(data) - 1)
        return data[k]

    print(f"\n{'─'*62}")
    print(f"  {'Metric':<30} {'Value':>12}")
    print(f"{'─'*62}")
    print(f"  {'Total requests':<30} {n_requests:>12}")
    print(f"  {'Successful':<30} {len(ok_results):>12}")
    print(f"  {'Failed':<30} {len(fail_results):>12}")
    print(f"  {'Error rate':<30} {len(fail_results)/n_requests*100:>11.2f}%")
    print(f"  {'Total time (s)':<30} {elapsed_s:>12.2f}")
    print(f"  {'Throughput (req/s)':<30} {n_requests/elapsed_s:>12.2f}")
    print()
    if latencies:
        print(f"  {'Average latency (ms)':<30} {statistics.mean(latencies):>12.2f}")
        print(f"  {'Median / P50 (ms)':<30} {pct(latencies, 50):>12.2f}")
        print(f"  {'P95 (ms)':<30} {pct(latencies, 95):>12.2f}")
        print(f"  {'P99 (ms)':<30} {pct(latencies, 99):>12.2f}")
        print(f"  {'Worst case (ms)':<30} {max(latencies):>12.2f}")
        print(f"  {'Best case (ms)':<30} {min(latencies):>12.2f}")
        print(f"  {'Std dev (ms)':<30} {statistics.stdev(latencies) if len(latencies)>1 else 0:>12.2f}")

    if fail_results:
        print(f"\n  {'─'*60}")
        print(f"  Sample errors:")
        for r in fail_results[:3]:
            print(f"    [{r['url'][:50]}] {r.get('error','?')}")
    print(f"{'='*62}\n")

    # Write JSON report
    report = {
        "ts":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target":      base_url,
        "requests":    n_requests,
        "concurrency": concurrency,
        "successful":  len(ok_results),
        "failed":      len(fail_results),
        "elapsed_s":   round(elapsed_s, 3),
        "throughput":  round(n_requests / elapsed_s, 2),
        "latency": {
            "mean_ms":   round(statistics.mean(latencies), 2) if latencies else 0,
            "p50_ms":    round(pct(latencies, 50), 2),
            "p95_ms":    round(pct(latencies, 95), 2),
            "p99_ms":    round(pct(latencies, 99), 2),
            "worst_ms":  round(max(latencies), 2) if latencies else 0,
            "best_ms":   round(min(latencies), 2) if latencies else 0,
        },
    }
    out_file = "benchmark_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved → {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",         default="http://localhost:8000")
    parser.add_argument("--requests",    type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--warmup",      type=int, default=10)
    args = parser.parse_args()

    run_benchmark(args.url, args.requests, args.concurrency, args.warmup)
