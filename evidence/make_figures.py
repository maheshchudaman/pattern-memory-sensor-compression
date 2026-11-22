#!/usr/bin/env python3
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/paper3_review/mpl")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("evidence/results")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def real_dataset_figure():
    rows = read_csv(OUT / "benchmark_results.csv")
    methods = [
        ("zlib_float32_lossless", "Zlib lossless"),
        ("delta_zlib_float32_lossless", "Delta + zlib"),
        ("direct_sparse_fft", "Direct sparse FFT"),
        ("adaptive_prototype_residual", "Adaptive prototype residual"),
    ]
    datasets = ["UCI_HAR", "UCI_EEG_Eye_State", "UCI_WISDM_Phone_Accelerometer"]
    values = []
    for method, _ in methods:
        values.append([
            float(next(r["compression_ratio"] for r in rows
                       if r["dataset"] == dataset and r["method"] == method
                       and float(r["target_nrmse"]) == 0.05))
            for dataset in datasets
        ])
    x = np.arange(len(datasets))
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, ((_, label), series) in enumerate(zip(methods, values)):
        ax.bar(x + (i - 1.5) * width, series, width, label=label)
    ax.set_ylabel("Compression ratio (raw bytes / encoded bytes)")
    ax.set_xticks(x, ["UCI HAR", "UCI EEG Eye State", "UCI WISDM phone accel."])
    ax.set_title("Measured storage ratios at target NRMSE 0.05")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "real_dataset_compression.png", dpi=300)
    plt.close(fig)


def break_even_figure():
    rows = read_csv(OUT / "synthetic_break_even.csv")
    noises = sorted({float(r["noise_sigma"]) for r in rows})
    recurrences = sorted({float(r["recurrence"]) for r in rows})
    matrix = np.zeros((len(noises), len(recurrences)))
    for row in rows:
        i = noises.index(float(row["noise_sigma"]))
        j = recurrences.index(float(row["recurrence"]))
        matrix[i, j] = float(row["adaptive_gain_percent"])
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-10, vmax=50, aspect="auto")
    ax.set_xticks(range(len(recurrences)), [f"{v:.0%}" for v in recurrences])
    ax.set_yticks(range(len(noises)), [f"{v:.2f}" for v in noises])
    ax.set_xlabel("Recurring-window share")
    ax.set_ylabel("Noise standard deviation")
    ax.set_title("Prototype-residual gain over direct sparse FFT")
    for i in range(len(noises)):
        for j in range(len(recurrences)):
            label = ">1000%" if matrix[i, j] > 1000 else f"{matrix[i, j]:.1f}%"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Compression-ratio gain (%)")
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_break_even.png", dpi=300)
    plt.close(fig)


def codec_baseline_figure():
    codec_rows = read_csv(OUT / "codec_baselines.csv")
    learned_rows = read_csv(OUT / "learned_gate_results.csv")
    datasets = ["UCI_HAR", "UCI_EEG_Eye_State", "UCI_WISDM_Phone_Accelerometer"]
    methods = [
        ("gorilla_xor_float32", "Gorilla XOR", "lossless"),
        ("zstandard_19_float32", "Zstandard", "lossless"),
        ("official_sprintz_delta_rle_uint16", "Sprintz uint16", "quantized"),
        ("zfp_fixed_accuracy_abs_0.2", "ZFP abs. 0.2", "lossy"),
        ("learned_gate", "Proposed gate", "lossy"),
    ]
    values = np.zeros((len(methods), len(datasets)))
    for method_index, (method, _, _) in enumerate(methods):
        for dataset_index, dataset in enumerate(datasets):
            if method == "learned_gate":
                row = next(r for r in learned_rows if r["dataset"] == dataset)
                values[method_index, dataset_index] = float(row["learned_ratio"])
            else:
                row = next(r for r in codec_rows
                           if r["dataset"] == dataset and r["method"] == method)
                values[method_index, dataset_index] = float(row["compression_ratio"])

    x = np.arange(len(datasets))
    width = 0.16
    colors = {
        "lossless": "#4C78A8",
        "quantized": "#F2CF5B",
        "lossy": "#E45756",
    }
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for index, ((_, label, kind), series) in enumerate(zip(methods, values)):
        bars = ax.bar(x + (index - 2) * width, series, width,
                      label=label, color=colors[kind], alpha=0.72 + index * 0.05,
                      edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in series],
                     padding=2, fontsize=7, rotation=90)
    ax.set_ylabel("Compression ratio (raw bytes / total encoded bytes)")
    ax.set_xticks(x, ["UCI HAR", "UCI EEG Eye State", "UCI WISDM phone accel."])
    ax.set_ylim(0, max(values.ravel()) * 1.22)
    ax.set_title("Verified codec baselines and learned coding gate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.text(0.01, -0.17,
            "Lossless, uint16-quantized and lossy operating points are not quality-equivalent; "
            "see codec_baselines.csv for actual errors.",
            transform=ax.transAxes, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "codec_baseline_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    real_dataset_figure()
    break_even_figure()
    codec_baseline_figure()
