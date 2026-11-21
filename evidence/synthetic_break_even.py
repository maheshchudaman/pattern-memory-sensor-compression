#!/usr/bin/env python3
"""Controlled recurrence/noise stress test for prototype-residual storage."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from prototype_residual_benchmark import SEED, evaluate


def base_patterns(count=16, length=64, channels=6):
    rng = np.random.default_rng(SEED)
    t = np.linspace(0, 1, length, endpoint=False)
    patterns = []
    for _ in range(count):
        sample = np.zeros((length, channels), dtype=np.float32)
        for ch in range(channels):
            for _harmonic in range(3):
                frequency = rng.integers(1, 9)
                amplitude = rng.uniform(0.2, 1.0)
                phase = rng.uniform(0, 2 * np.pi)
                sample[:, ch] += amplitude * np.sin(2 * np.pi * frequency * t + phase)
        patterns.append(sample)
    return np.stack(patterns)


def independent_window(rng, length=64, channels=6):
    t = np.linspace(0, 1, length, endpoint=False)
    sample = np.zeros((length, channels), dtype=np.float32)
    for ch in range(channels):
        for _ in range(3):
            frequency = rng.integers(1, 25)
            sample[:, ch] += rng.uniform(0.2, 1.0) * np.sin(
                2 * np.pi * frequency * t + rng.uniform(0, 2 * np.pi)
            )
    return sample


def make_split(patterns, windows, recurrence, noise, seed):
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(windows):
        if rng.random() < recurrence:
            sample = patterns[rng.integers(len(patterns))].copy()
        else:
            sample = independent_window(rng, patterns.shape[1], patterns.shape[2])
        sample += rng.normal(0.0, noise, size=sample.shape)
        output.append(sample.astype(np.float32))
    return np.stack(output)


def main():
    output_dir = Path("evidence/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = base_patterns()
    rows = []
    for noise in (0.01, 0.05, 0.10, 0.20):
        for recurrence in (0.0, 0.25, 0.50, 0.75, 1.0):
            train = make_split(patterns, 800, recurrence, noise, SEED + 1)
            test = make_split(patterns, 1000, recurrence, noise, SEED + 2)
            result = evaluate(
                f"synthetic_r{recurrence:.2f}_n{noise:.2f}",
                train,
                test,
                k=16,
                epsilon=0.05,
            )
            selected = {r["method"]: r for r in result}
            direct = selected["direct_sparse_fft"]
            adaptive = selected["adaptive_prototype_residual"]
            rows.append({
                "recurrence": recurrence,
                "noise_sigma": noise,
                "direct_ratio": direct["compression_ratio"],
                "adaptive_ratio": adaptive["compression_ratio"],
                "adaptive_gain_percent": 100.0 * (
                    adaptive["compression_ratio"] / direct["compression_ratio"] - 1.0
                ),
                "prototype_route_rate": adaptive["prototype_route_rate"],
                "adaptive_mean_nrmse": adaptive["mean_nrmse"],
                "adaptive_p95_nrmse": adaptive["p95_nrmse"],
            })
            print(
                f"noise={noise:.2f} recurrence={recurrence:.2f} "
                f"gain={rows[-1]['adaptive_gain_percent']:.1f}%"
            )
    path = output_dir / "synthetic_break_even.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
