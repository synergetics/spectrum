"""Informal C vs pure-Python benchmarks (run directly, not in CI).

Usage: python tests/bench/bench_core.py
"""

import os
import time

import numpy as np


def bench(name, fn, reps=3):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    import spectrum

    rng = np.random.default_rng(0)
    n = 65536
    y = rng.standard_normal(n).reshape(-1, 1)

    results = {}
    for backend in ("python", "c"):
        os.environ["SPECTRUM_BACKEND"] = backend
        from spectrum import bicoherence, bispectrumd, cum2est, trispec  # noqa

        r = {}
        r["cum2est(20, N=64k)"] = bench("cum2est", lambda: cum2est(y, 20, 4096, 50, "biased"))
        r["bispectrumd(4096,256)"] = bench(
            "bispectrumd", lambda: bispectrumd(y, nfft=256, nsamp=4096, overlap=50)
        )
        r["bicoherence(256,64)"] = bench(
            "bicoherence", lambda: bicoherence(y, nfft=256, nsamp=64, overlap=50)
        )
        results[backend] = r

    y2 = rng.standard_normal(2048).reshape(-1, 1)
    for backend in ("python", "c"):
        os.environ["SPECTRUM_BACKEND"] = backend
        from spectrum import trispec

        results[backend]["trispec(32, 256)"] = bench(
            "trispec", lambda: trispec(y2, nfft=32, nsamp=256, overlap=50)
        )

    os.environ.pop("SPECTRUM_BACKEND", None)

    keys = list(results["c"].keys())
    print(f"{'benchmark':<24}{'python (s)':>12}{'C (s)':>12}{'speedup':>10}")
    print("-" * 58)
    for k in keys:
        tp, tc = results["python"][k], results["c"][k]
        print(f"{k:<24}{tp:>12.4f}{tc:>12.4f}{tp / tc:>9.1f}x")


if __name__ == "__main__":
    main()
