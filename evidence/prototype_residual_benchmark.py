#!/usr/bin/env python3
"""Reproducible prototype-residual benchmark for public sensor datasets.

This is research code, not a production codec. It reports actual serialized
payload estimates for a transparent record format and includes the prototype
dictionary overhead. All randomized steps use a fixed seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import time
import zlib
from pathlib import Path

import numpy as np


SEED = 20261115


def load_har(root: Path):
    base = root / "UCI HAR Dataset"
    channels = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]

    def split(name: str):
        parts = []
        signal_dir = base / name / "Inertial Signals"
        for channel in channels:
            parts.append(np.loadtxt(signal_dir / f"{channel}_{name}.txt", dtype=np.float32))
        return np.stack(parts, axis=-1)

    return split("train"), split("test")


def load_eeg(arff: Path, window: int = 64, stride: int = 32):
    rows = []
    in_data = False
    with arff.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            if stripped.lower() == "@data":
                in_data = True
                continue
            if in_data:
                values = [float(v) for v in stripped.split(",")]
                rows.append(values[:-1])
    signal = np.asarray(rows, dtype=np.float32)
    windows = np.stack(
        [signal[i:i + window] for i in range(0, len(signal) - window + 1, stride)],
        axis=0,
    )
    cut = int(0.60 * len(windows))
    return windows[:cut], windows[cut:]


def load_wisdm_split(root: Path, window: int = 128, windows_per_activity: int = 12):
    """Load a balanced, subject-disjoint WISDM phone-accelerometer subset."""
    files = sorted(root.rglob("data_*_accel_phone.txt"))
    if len(files) < 51:
        raise FileNotFoundError(
            f"expected 51 WISDM phone-accelerometer files below {root}, found {len(files)}"
        )
    subject_ids = sorted({int(re.search(r"data_(\d+)_", path.name).group(1)) for path in files})
    if len(subject_ids) != 51:
        raise ValueError(f"expected 51 WISDM subjects, found {len(subject_ids)}")
    train_subjects = set(subject_ids[:41])
    activity_codes = [code for code in "ABCDEFGHIJKLMNOPQRS" if code != "N"]
    activity_index = {code: index for index, code in enumerate(activity_codes)}

    train_windows, test_windows = [], []
    train_labels, test_labels, test_subject_ids = [], [], []
    for path in files:
        match = re.search(r"data_(\d+)_", path.name)
        if not match:
            continue
        subject = int(match.group(1))
        by_activity = {code: [] for code in activity_codes}
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.strip().rstrip(";").split(",")
                if len(parts) != 6 or parts[1] not in by_activity:
                    continue
                try:
                    by_activity[parts[1]].append([float(parts[3]), float(parts[4]), float(parts[5])])
                except ValueError:
                    continue
        for activity in activity_codes:
            values = np.asarray(by_activity[activity], dtype=np.float32)
            if len(values) < window:
                continue
            count = min(windows_per_activity, len(values) // window)
            starts = np.linspace(0, len(values) - window, count, dtype=np.int64)
            for start in np.unique(starts):
                sample = values[start:start + window]
                if subject in train_subjects:
                    train_windows.append(sample)
                    train_labels.append(activity_index[activity])
                else:
                    test_windows.append(sample)
                    test_labels.append(activity_index[activity])
                    test_subject_ids.append(subject)
    return (
        np.stack(train_windows), np.stack(test_windows),
        np.asarray(train_labels, dtype=np.int64), np.asarray(test_labels, dtype=np.int64),
        np.asarray(test_subject_ids, dtype=np.int64),
    )


def load_wisdm(root: Path, window: int = 128, windows_per_activity: int = 12):
    train, test, *_ = load_wisdm_split(root, window, windows_per_activity)
    return train, test


def robust_standardize(train: np.ndarray, test: np.ndarray):
    # Training-derived clipping prevents a few known EEG artifacts from
    # dominating every error metric. The same deterministic bounds are used
    # by both encoder and baselines.
    flat = train.reshape(-1, train.shape[-1]).astype(np.float64)
    lo = np.quantile(flat, 0.001, axis=0)
    hi = np.quantile(flat, 0.999, axis=0)
    clipped_train = np.clip(train, lo, hi)
    clipped_test = np.clip(test, lo, hi)
    mean = clipped_train.reshape(-1, train.shape[-1]).mean(axis=0)
    std = clipped_train.reshape(-1, train.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0
    return (
        ((clipped_train - mean) / std).astype(np.float32),
        ((clipped_test - mean) / std).astype(np.float32),
        mean,
        std,
        lo,
        hi,
    )


def kmeans_prototypes(x: np.ndarray, k: int, iterations: int = 12, max_train: int = 2500):
    rng = np.random.default_rng(SEED)
    matrix = x.reshape(len(x), -1).astype(np.float32)
    if len(matrix) > max_train:
        matrix = matrix[rng.choice(len(matrix), max_train, replace=False)]
    centers = np.empty((k, matrix.shape[1]), dtype=np.float32)
    centers[0] = matrix[rng.integers(len(matrix))]
    closest = np.sum((matrix - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(closest.sum())
        if total <= 0:
            centers[i] = matrix[rng.integers(len(matrix))]
        else:
            centers[i] = matrix[rng.choice(len(matrix), p=closest / total)]
        closest = np.minimum(closest, np.sum((matrix - centers[i]) ** 2, axis=1))
    for _ in range(iterations):
        distances = (
            np.sum(matrix * matrix, axis=1, keepdims=True)
            + np.sum(centers * centers, axis=1)[None, :]
            - 2.0 * matrix @ centers.T
        )
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            members = matrix[labels == i]
            if len(members):
                new_centers[i] = members.mean(axis=0)
        shift = float(np.mean((new_centers - centers) ** 2))
        centers = new_centers
        if shift < 1e-8:
            break
    # Decoder stores the dictionary as float16, so evaluation uses exactly it.
    return centers.astype(np.float16).astype(np.float32)


def nearest_prototype(vector: np.ndarray, centers: np.ndarray):
    idx = int(np.argmin(np.sum((centers - vector[None, :]) ** 2, axis=1)))
    return idx, centers[idx]


def transform_record(vector: np.ndarray, target_nrmse: float):
    """Top-energy real-FFT record with float16 coefficients.

    Record layout: type byte, coefficient count uint16, then repeated
    (frequency index uint16, real float16, imaginary float16).
    """
    n = len(vector)
    coeff = np.fft.rfft(vector, norm="ortho")
    weights = np.full(len(coeff), 2.0, dtype=np.float64)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    energy = weights * (coeff.real ** 2 + coeff.imag ** 2)
    order = np.argsort(energy)[::-1]
    denom = max(float(np.std(vector)), 0.10)
    allowed = (target_nrmse * denom) ** 2 * n
    cumulative = np.cumsum(energy[order])
    required = max(0.0, float(energy.sum() - allowed))
    k = int(np.searchsorted(cumulative, required, side="left") + 1) if required > 0 else 0
    k = min(k, len(order))
    selected = order[:k]
    qreal = coeff.real[selected].astype(np.float16)
    qimag = coeff.imag[selected].astype(np.float16)
    sparse = np.zeros_like(coeff)
    sparse[selected] = qreal.astype(np.float64) + 1j * qimag.astype(np.float64)
    reconstruction = np.fft.irfft(sparse, n=n, norm="ortho").astype(np.float32)
    nrmse = float(np.sqrt(np.mean((vector - reconstruction) ** 2)) / denom)
    # Float16 rounding is normally negligible; if it crosses the bound, add
    # coefficients deterministically until the measured bound is met.
    while nrmse > target_nrmse * 1.005 and k < len(order):
        k = min(len(order), max(k + 1, int(np.ceil(k * 1.05))))
        selected = order[:k]
        qreal = coeff.real[selected].astype(np.float16)
        qimag = coeff.imag[selected].astype(np.float16)
        sparse.fill(0)
        sparse[selected] = qreal.astype(np.float64) + 1j * qimag.astype(np.float64)
        reconstruction = np.fft.irfft(sparse, n=n, norm="ortho").astype(np.float32)
        nrmse = float(np.sqrt(np.mean((vector - reconstruction) ** 2)) / denom)
    entries = np.empty(k, dtype=np.dtype([("index", "<u2"), ("real", "<f2"), ("imag", "<f2")]))
    entries["index"] = selected.astype(np.uint16)
    entries["real"] = qreal
    entries["imag"] = qimag
    payload = struct.pack("<BH", 1, k) + entries.tobytes()
    return payload, reconstruction, nrmse, k


def evaluate(dataset: str, train: np.ndarray, test: np.ndarray, k: int, epsilon: float):
    train_z, test_z, mean, std, lo, hi = robust_standardize(train, test)
    flat_train = train_z.reshape(len(train_z), -1)
    flat_test = test_z.reshape(len(test_z), -1)
    centers = kmeans_prototypes(flat_train, k)
    n = flat_test.shape[1]
    raw_bytes = flat_test.astype("<f4").nbytes
    metadata_bytes = 4 * 8 * test.shape[-1]  # mean, std, clipping bounds
    dictionary_bytes = centers.astype("<f2").nbytes

    raw_blob = flat_test.astype("<f4").tobytes()
    zlib_size = len(zlib.compress(raw_blob, level=9)) + metadata_bytes
    shaped = test_z.astype("<f4")
    delta = np.diff(shaped, axis=1, prepend=shaped[:, :1, :])
    delta_zlib_size = len(zlib.compress(delta.tobytes(), level=9)) + metadata_bytes

    direct_size = metadata_bytes
    proto_size = metadata_bytes + dictionary_bytes
    adaptive_size = metadata_bytes + dictionary_bytes
    direct_errors, proto_errors, adaptive_errors = [], [], []
    direct_ks, proto_ks, adaptive_ks = [], [], []
    reference_only = 0
    adaptive_proto = 0
    adaptive_direct = 0
    adaptive_raw = 0
    direct_raw = 0
    proto_raw = 0

    start = time.perf_counter()
    for vector in flat_test:
        direct_record = transform_record(vector, epsilon)
        d_payload, d_rec, d_err, d_k = direct_record
        d_size = len(d_payload)
        if d_size >= 1 + 4 * n:
            d_payload = b"\x00" + vector.astype("<f4").tobytes()
            d_size, d_rec, d_err, d_k = len(d_payload), vector, 0.0, n
            direct_raw += 1
        direct_size += d_size
        direct_errors.append(d_err)
        direct_ks.append(d_k)

        proto_idx, prototype = nearest_prototype(vector, centers)
        residual = vector - prototype
        denom = max(float(np.std(vector)), 0.10)
        ref_err = float(np.sqrt(np.mean(residual ** 2)) / denom)
        if ref_err <= epsilon:
            p_payload = struct.pack("<BH", 2, proto_idx)
            p_size, p_rec, p_err, p_k = len(p_payload), prototype, ref_err, 0
            reference_only += 1
        else:
            residual_denom = max(float(np.std(residual)), 0.10)
            residual_target = epsilon * denom / residual_denom
            r_payload, r_rec, _, p_k = transform_record(residual, residual_target)
            p_rec = prototype + r_rec
            p_err = float(np.sqrt(np.mean((vector - p_rec) ** 2)) / denom)
            p_payload = struct.pack("<BH", 3, proto_idx) + r_payload
            p_size = len(p_payload)
        if p_size >= 1 + 4 * n:
            p_payload = b"\x00" + vector.astype("<f4").tobytes()
            p_size, p_rec, p_err, p_k = len(p_payload), vector, 0.0, n
            proto_raw += 1
        proto_size += p_size
        proto_errors.append(p_err)
        proto_ks.append(p_k)

        # Novelty gate: use the smaller valid representation. This avoids a
        # false biological claim and makes the policy directly falsifiable.
        if p_size < d_size:
            a_size, a_rec, a_err, a_k = p_size, p_rec, p_err, p_k
            adaptive_proto += 1
        else:
            a_size, a_rec, a_err, a_k = d_size, d_rec, d_err, d_k
            adaptive_direct += 1
        if a_size >= 1 + 4 * n:
            adaptive_raw += 1
        adaptive_size += a_size
        adaptive_errors.append(a_err)
        adaptive_ks.append(a_k)
    elapsed = time.perf_counter() - start

    mib = raw_bytes / (1024 ** 2)
    base = {
        "dataset": dataset,
        "windows": len(test_z),
        "window_shape": f"{test.shape[1]}x{test.shape[2]}",
        "prototype_count": k,
        "target_nrmse": epsilon,
        "raw_bytes": raw_bytes,
    }

    def row(method, size, errors, ks=None, extra=None):
        errors = np.asarray(errors, dtype=float)
        result = dict(base)
        result.update({
            "method": method,
            "encoded_bytes": int(size),
            "compression_ratio": float(raw_bytes / size),
            "mean_nrmse": float(errors.mean()) if len(errors) else 0.0,
            "p95_nrmse": float(np.quantile(errors, 0.95)) if len(errors) else 0.0,
            "mean_coefficients": float(np.mean(ks)) if ks is not None else 0.0,
            "benchmark_mib_per_s": float(mib / elapsed) if method == "adaptive_prototype_residual" else "",
        })
        if extra:
            result.update(extra)
        return result

    return [
        row("raw_float32", raw_bytes + metadata_bytes, np.zeros(len(test_z))),
        row("zlib_float32_lossless", zlib_size, np.zeros(len(test_z))),
        row("delta_zlib_float32_lossless", delta_zlib_size, np.zeros(len(test_z))),
        row("direct_sparse_fft", direct_size, direct_errors, direct_ks,
            {"raw_fallback_rate": direct_raw / len(test_z)}),
        row("prototype_residual_always", proto_size, proto_errors, proto_ks,
            {"reference_only_rate": reference_only / len(test_z),
             "raw_fallback_rate": proto_raw / len(test_z)}),
        row("adaptive_prototype_residual", adaptive_size, adaptive_errors, adaptive_ks,
            {"prototype_route_rate": adaptive_proto / len(test_z),
             "direct_route_rate": adaptive_direct / len(test_z),
             "raw_fallback_rate": adaptive_raw / len(test_z)}),
    ]


def repeated_evaluate(dataset: str, train: np.ndarray, test: np.ndarray, k: int,
                      epsilon: float, repetitions: int):
    runs = [evaluate(dataset, train, test, k, epsilon) for _ in range(repetitions)]
    reference = runs[0]
    adaptive_index = next(
        index for index, row in enumerate(reference)
        if row["method"] == "adaptive_prototype_residual"
    )
    throughput = np.asarray(
        [run[adaptive_index]["benchmark_mib_per_s"] for run in runs], dtype=float
    )
    adaptive = reference[adaptive_index]
    adaptive["benchmark_mib_per_s"] = float(np.median(throughput))
    adaptive["benchmark_mib_per_s_q1"] = float(np.quantile(throughput, 0.25))
    adaptive["benchmark_mib_per_s_q3"] = float(np.quantile(throughput, 0.75))
    adaptive["runtime_repetitions"] = repetitions
    return reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--har-root", type=Path, required=True)
    parser.add_argument("--eeg-arff", type=Path, required=True)
    parser.add_argument("--wisdm-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evidence/results"))
    parser.add_argument("--prototypes", type=int, default=16)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--runtime-repetitions", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    har = load_har(args.har_root)
    eeg = load_eeg(args.eeg_arff)
    datasets = [("UCI_HAR", har), ("UCI_EEG_Eye_State", eeg)]
    if args.wisdm_root:
        datasets.append(("UCI_WISDM_Phone_Accelerometer", load_wisdm(args.wisdm_root)))
    all_rows = []
    for name, (train, test) in datasets:
        for epsilon in args.epsilons:
            print(f"Evaluating {name} at epsilon={epsilon:.3f}", flush=True)
            all_rows.extend(repeated_evaluate(
                name, train, test, args.prototypes, epsilon, args.runtime_repetitions
            ))

    fieldnames = sorted({key for row in all_rows for key in row})
    csv_path = args.output_dir / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    json_path = args.output_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
