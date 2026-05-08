from __future__ import annotations

import time


def measure(fn, rows: list[dict]) -> dict:
    times = []
    for row in rows:
        start = time.perf_counter()
        fn(row["query"])
        times.append((time.perf_counter() - start) * 1000.0)
    if not times:
        return {"avg_ms": 0.0, "p95_ms": 0.0, "qps": 0.0}
    times_sorted = sorted(times)
    p95 = times_sorted[min(len(times_sorted) - 1, int(0.95 * len(times_sorted)))]
    avg = sum(times) / len(times)
    return {"avg_ms": avg, "p95_ms": p95, "qps": 1000.0 / avg if avg else 0.0}

