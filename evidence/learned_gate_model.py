#!/usr/bin/env python3
"""Train and evaluate a coding-only learned gate for pattern-memory storage.

The model predicts whether a window should use direct sparse-transform coding
or nearest-prototype residual coding. It is trained to minimize encoded bytes,
not to imitate a neuroscience construct. A small downstream ridge classifier
checks whether reconstructed windows preserve task decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from prototype_residual_benchmark import (
    SEED,
    kmeans_prototypes,
    load_har,
    nearest_prototype,
    robust_standardize,
    transform_record,
)


FEATURE_NAMES = [
    "prototype_nrmse",
    "residual_to_signal_std",
    "prototype_correlation",
    "normalized_mean_abs_residual",
    "signal_lag1_correlation",
    "residual_lag1_correlation",
    "derivative_std_ratio",
    "residual_zero_crossing_rate",
]


def load_har_with_labels(root: Path):
    train, test = load_har(root)
    base = root / "UCI HAR Dataset"
    y_train = np.loadtxt(base / "train" / "y_train.txt", dtype=np.int64) - 1
    y_test = np.loadtxt(base / "test" / "y_test.txt", dtype=np.int64) - 1
    test_subjects = np.loadtxt(base / "test" / "subject_test.txt", dtype=np.int64)
    return train, test, y_train, y_test, test_subjects


def load_eeg_with_labels(arff: Path, window=64, stride=32):
    values, labels = [], []
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
                row = [float(value) for value in stripped.split(",")]
                values.append(row[:-1])
                labels.append(int(row[-1]))
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    windows, window_labels = [], []
    for start in range(0, len(values) - window + 1, stride):
        stop = start + window
        windows.append(values[start:stop])
        counts = np.bincount(labels[start:stop], minlength=2)
        window_labels.append(int(np.argmax(counts)))
    windows = np.stack(windows)
    window_labels = np.asarray(window_labels)
    cut = int(0.60 * len(windows))
    return windows[:cut], windows[cut:], window_labels[:cut], window_labels[cut:], None


def safe_corr(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0


def lag1_corr(x):
    return safe_corr(x[:-1], x[1:]) if len(x) > 1 else 0.0


def routing_features(vector, prototype):
    residual = vector - prototype
    signal_std = max(float(np.std(vector)), 0.10)
    residual_std = float(np.std(residual))
    signal_diff_std = max(float(np.std(np.diff(vector))), 1e-8)
    residual_diff_std = float(np.std(np.diff(residual)))
    zero_cross = float(np.mean((residual[:-1] * residual[1:]) < 0)) if len(residual) > 1 else 0.0
    return np.asarray([
        float(np.sqrt(np.mean(residual ** 2)) / signal_std),
        residual_std / signal_std,
        safe_corr(vector, prototype),
        float(np.mean(np.abs(residual)) / signal_std),
        lag1_corr(vector),
        lag1_corr(residual),
        residual_diff_std / signal_diff_std,
        zero_cross,
    ], dtype=np.float64)


def make_candidate_records(matrix, centers, epsilon):
    features = np.empty((len(matrix), len(FEATURE_NAMES)), dtype=np.float64)
    direct_sizes = np.empty(len(matrix), dtype=np.int64)
    proto_sizes = np.empty(len(matrix), dtype=np.int64)
    direct_rec = np.empty_like(matrix, dtype=np.float32)
    proto_rec = np.empty_like(matrix, dtype=np.float32)
    for i, vector in enumerate(matrix):
        direct_payload, direct_reconstruction, _, _ = transform_record(vector, epsilon)
        prototype_idx, prototype = nearest_prototype(vector, centers)
        features[i] = routing_features(vector, prototype)
        residual = vector - prototype
        denom = max(float(np.std(vector)), 0.10)
        reference_error = float(np.sqrt(np.mean(residual ** 2)) / denom)
        if reference_error <= epsilon:
            proto_payload_size = 3
            proto_reconstruction = prototype
        else:
            residual_denom = max(float(np.std(residual)), 0.10)
            residual_target = epsilon * denom / residual_denom
            residual_payload, residual_reconstruction, _, _ = transform_record(residual, residual_target)
            proto_payload_size = 3 + len(residual_payload)
            proto_reconstruction = prototype + residual_reconstruction
        raw_size = 1 + 4 * len(vector)
        if len(direct_payload) >= raw_size:
            direct_sizes[i] = raw_size
            direct_rec[i] = vector
        else:
            direct_sizes[i] = len(direct_payload)
            direct_rec[i] = direct_reconstruction
        if proto_payload_size >= raw_size:
            proto_sizes[i] = raw_size
            proto_rec[i] = vector
        else:
            proto_sizes[i] = proto_payload_size
            proto_rec[i] = proto_reconstruction
    return features, direct_sizes, proto_sizes, direct_rec, proto_rec


def sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(x, y, iterations=1800, learning_rate=0.05, l2=1e-3):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    z = (x - mean) / std
    design = np.column_stack([np.ones(len(z)), z])
    weights = np.zeros(design.shape[1], dtype=np.float64)
    positive = max(int(y.sum()), 1)
    negative = max(int(len(y) - y.sum()), 1)
    sample_weight = np.where(y == 1, len(y) / (2 * positive), len(y) / (2 * negative))
    for _ in range(iterations):
        probabilities = sigmoid(design @ weights)
        gradient = design.T @ ((probabilities - y) * sample_weight) / len(y)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return weights, mean, std


def predict_probability(x, weights, mean, std):
    z = (x - mean) / std
    design = np.column_stack([np.ones(len(z)), z])
    return sigmoid(design @ weights)


def select_cost_threshold(probability, direct_sizes, proto_sizes):
    candidates = np.concatenate([np.linspace(0.0, 1.0, 201), np.asarray([1.01])])
    costs = []
    for threshold in candidates:
        choose_proto = probability >= threshold
        costs.append(np.where(choose_proto, proto_sizes, direct_sizes).sum())
    return float(candidates[int(np.argmin(costs))])


def confusion_metrics(truth, prediction):
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    accuracy = (tp + tn) / max(len(truth), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def downstream_features(windows):
    # Compact, deterministic time/frequency features; no external ML package.
    values = []
    for window in windows:
        channel_features = []
        for channel in window.T:
            spectrum = np.abs(np.fft.rfft(channel)) ** 2
            bands = np.array_split(spectrum, 4)
            band_energy = [float(np.sum(band)) for band in bands]
            total = max(sum(band_energy), 1e-12)
            channel_features.extend([
                float(np.mean(channel)), float(np.std(channel)),
                float(np.min(channel)), float(np.max(channel)),
                float(np.sqrt(np.mean(channel ** 2))),
                float(np.mean(np.abs(np.diff(channel)))),
                *[energy / total for energy in band_energy],
            ])
        values.append(channel_features)
    return np.asarray(values, dtype=np.float64)


def fit_ridge_classifier(x, y, ridge=1.0):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    z = (x - mean) / std
    z = np.column_stack([np.ones(len(z)), z])
    classes = np.unique(y)
    target = np.zeros((len(y), len(classes)), dtype=np.float64)
    for idx, label in enumerate(classes):
        target[:, idx] = (y == label).astype(np.float64)
    regularizer = ridge * np.eye(z.shape[1])
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(z.T @ z + regularizer, z.T @ target)
    return weights, mean, std, classes


def ridge_predict(x, model):
    weights, mean, std, classes = model
    z = (x - mean) / std
    z = np.column_stack([np.ones(len(z)), z])
    return classes[np.argmax(z @ weights, axis=1)]


def macro_f1(y_true, y_pred):
    scores = []
    for label in np.unique(y_true):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


def task_metrics(y_true, y_pred):
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": macro_f1(y_true, y_pred),
        "changed_decisions": int(np.sum(y_true != y_pred)),
    }


def subject_cluster_bootstrap(y_true, original_prediction, reconstructed_prediction,
                              subjects, repetitions=10000, seed=SEED):
    """Paired percentile intervals from resampling complete HAR subjects."""
    unique_subjects = np.unique(subjects)
    indices = {subject: np.flatnonzero(subjects == subject) for subject in unique_subjects}
    rng = np.random.default_rng(seed)
    accuracy_changes = np.empty(repetitions, dtype=np.float64)
    macro_f1_changes = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        selection = np.concatenate([indices[subject] for subject in sampled])
        truth = y_true[selection]
        original = original_prediction[selection]
        reconstructed = reconstructed_prediction[selection]
        accuracy_changes[repetition] = (
            np.mean(truth == reconstructed) - np.mean(truth == original)
        )
        macro_f1_changes[repetition] = macro_f1(truth, reconstructed) - macro_f1(truth, original)

    def summarize(samples):
        low, high = np.quantile(samples, [0.025, 0.975])
        return {
            "lower_95": float(low),
            "upper_95": float(high),
            "bootstrap_mean": float(np.mean(samples)),
        }

    return {
        "unit": "HAR_test_subject",
        "subjects": int(len(unique_subjects)),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "accuracy_change": summarize(accuracy_changes),
        "macro_f1_change": summarize(macro_f1_changes),
    }


def run_dataset(name, train, test, y_train, y_test, test_subjects,
                epsilon, prototypes, output_dir):
    train_z, test_z, mean, std, lo, hi = robust_standardize(train, test)
    train_matrix = train_z.reshape(len(train_z), -1)
    test_matrix = test_z.reshape(len(test_z), -1)
    centers = kmeans_prototypes(train_matrix, prototypes)

    train_features, train_direct_size, train_proto_size, _, _ = make_candidate_records(
        train_matrix, centers, epsilon
    )
    split = int(0.80 * len(train_features))
    gate_model = fit_logistic(train_features[:split], (train_proto_size[:split] < train_direct_size[:split]).astype(float))
    val_probability = predict_probability(train_features[split:], *gate_model)
    threshold = select_cost_threshold(val_probability, train_direct_size[split:], train_proto_size[split:])
    gate_parameter_bytes = (1 + 3 * len(FEATURE_NAMES)) * 8 + 8
    dictionary_bytes = centers.astype("<f2").nbytes
    val_proto_route = val_probability >= threshold
    val_learned_bytes = np.where(val_proto_route, train_proto_size[split:], train_direct_size[split:]).sum()
    prototype_enabled = bool(val_learned_bytes + dictionary_bytes + gate_parameter_bytes < train_direct_size[split:].sum())
    if not prototype_enabled:
        threshold = 1.01

    test_features, direct_size, proto_size, direct_rec, proto_rec = make_candidate_records(
        test_matrix, centers, epsilon
    )
    probability = predict_probability(test_features, *gate_model)
    learned_proto = (probability >= threshold) if prototype_enabled else np.zeros(len(probability), dtype=bool)
    oracle_proto = proto_size < direct_size
    learned_size = np.where(learned_proto, proto_size, direct_size)
    oracle_size = np.minimum(proto_size, direct_size)
    learned_rec = np.where(learned_proto[:, None], proto_rec, direct_rec)
    raw_bytes = test_matrix.astype("<f4").nbytes
    normalization_bytes = 4 * 8 * test.shape[-1]
    metadata_bytes = normalization_bytes + gate_parameter_bytes + (dictionary_bytes if prototype_enabled else 0)
    direct_total_bytes = direct_size.sum() + normalization_bytes
    oracle_pattern_total_bytes = oracle_size.sum() + normalization_bytes + dictionary_bytes
    oracle_total_bytes = min(direct_total_bytes, oracle_pattern_total_bytes)

    gate = confusion_metrics(oracle_proto.astype(int), learned_proto.astype(int))
    gate.update({
        "threshold": threshold,
        "prototype_enabled": prototype_enabled,
        "oracle_prototype_rate": float(np.mean(oracle_proto)),
        "learned_prototype_rate": float(np.mean(learned_proto)),
        "learned_compression_ratio": float(raw_bytes / (learned_size.sum() + metadata_bytes)),
        "oracle_compression_ratio": float(raw_bytes / oracle_total_bytes),
        "direct_compression_ratio": float(raw_bytes / direct_total_bytes),
        "byte_regret_percent": float(100 * ((learned_size.sum() + metadata_bytes) /
                                             oracle_total_bytes - 1)),
    })

    reconstructed_z = learned_rec.reshape(test_z.shape)
    if test_subjects is not None:
        task_train_features = downstream_features(train_z)
        task_test_original = downstream_features(test_z)
        task_test_reconstructed = downstream_features(reconstructed_z)
        classifier = fit_ridge_classifier(task_train_features, y_train, ridge=2.0)
        original_prediction = ridge_predict(task_test_original, classifier)
        reconstructed_prediction = ridge_predict(task_test_reconstructed, classifier)
        original_metrics = task_metrics(y_test, original_prediction)
        reconstructed_metrics = task_metrics(y_test, reconstructed_prediction)
        downstream = {
            "status": "evaluated_subject_independent_official_split",
            "original": original_metrics,
            "reconstructed": reconstructed_metrics,
            "accuracy_change": reconstructed_metrics["accuracy"] - original_metrics["accuracy"],
            "macro_f1_change": reconstructed_metrics["macro_f1"] - original_metrics["macro_f1"],
            "prediction_flip_rate": float(np.mean(original_prediction != reconstructed_prediction)),
            "subject_cluster_bootstrap_95_ci": subject_cluster_bootstrap(
                y_test, original_prediction, reconstructed_prediction, test_subjects
            ),
        }
    else:
        downstream = {
            "status": "not_evaluated",
            "reason": (
                "The chronological EEG window split has severe label shift and is retained "
                "for compression analysis only; no downstream performance claim is made."
            ),
        }

    weights, feature_mean, feature_std = gate_model
    model_path = output_dir / f"apmc_gate_{name}.npz"
    np.savez_compressed(
        model_path,
        version=np.asarray([1], dtype=np.int32),
        seed=np.asarray([SEED], dtype=np.int64),
        epsilon=np.asarray([epsilon], dtype=np.float64),
        window_shape=np.asarray(test.shape[1:], dtype=np.int32),
        prototypes=centers.astype(np.float16),
        normalization_mean=mean.astype(np.float32),
        normalization_std=std.astype(np.float32),
        clipping_low=lo.astype(np.float32),
        clipping_high=hi.astype(np.float32),
        gate_weights=weights,
        gate_feature_mean=feature_mean,
        gate_feature_std=feature_std,
        gate_threshold=np.asarray([threshold], dtype=np.float64),
        prototype_enabled=np.asarray([prototype_enabled], dtype=np.bool_),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    return {"dataset": name, "gate": gate, "downstream": downstream, "model_file": str(model_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--har-root", type=Path, required=True)
    parser.add_argument("--eeg-arff", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evidence/results"))
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--prototypes", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("UCI_HAR", load_har_with_labels(args.har_root)),
        ("UCI_EEG_Eye_State", load_eeg_with_labels(args.eeg_arff)),
    ]
    results = []
    for name, values in datasets:
        print(f"Training learned coding gate for {name}", flush=True)
        results.append(run_dataset(name, *values, args.epsilon, args.prototypes, args.output_dir))
    json_path = args.output_dir / "learned_gate_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "learned_gate_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "dataset", "gate_accuracy", "gate_f1", "threshold",
            "direct_ratio", "learned_ratio", "oracle_ratio", "byte_regret_percent",
            "downstream_status", "original_accuracy", "reconstructed_accuracy",
            "accuracy_change", "accuracy_change_ci95_low", "accuracy_change_ci95_high",
            "original_macro_f1", "reconstructed_macro_f1", "macro_f1_change",
            "macro_f1_change_ci95_low", "macro_f1_change_ci95_high",
            "prediction_flip_rate", "downstream_note", "model_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            gate = result["gate"]
            downstream = result["downstream"]
            original = downstream.get("original", {})
            reconstructed = downstream.get("reconstructed", {})
            bootstrap = downstream.get("subject_cluster_bootstrap_95_ci", {})
            accuracy_ci = bootstrap.get("accuracy_change", {})
            macro_f1_ci = bootstrap.get("macro_f1_change", {})
            writer.writerow({
                "dataset": result["dataset"],
                "gate_accuracy": gate["accuracy"],
                "gate_f1": gate["f1"],
                "threshold": gate["threshold"],
                "direct_ratio": gate["direct_compression_ratio"],
                "learned_ratio": gate["learned_compression_ratio"],
                "oracle_ratio": gate["oracle_compression_ratio"],
                "byte_regret_percent": gate["byte_regret_percent"],
                "downstream_status": downstream["status"],
                "original_accuracy": original.get("accuracy", ""),
                "reconstructed_accuracy": reconstructed.get("accuracy", ""),
                "accuracy_change": downstream.get("accuracy_change", ""),
                "accuracy_change_ci95_low": accuracy_ci.get("lower_95", ""),
                "accuracy_change_ci95_high": accuracy_ci.get("upper_95", ""),
                "original_macro_f1": original.get("macro_f1", ""),
                "reconstructed_macro_f1": reconstructed.get("macro_f1", ""),
                "macro_f1_change": downstream.get("macro_f1_change", ""),
                "macro_f1_change_ci95_low": macro_f1_ci.get("lower_95", ""),
                "macro_f1_change_ci95_high": macro_f1_ci.get("upper_95", ""),
                "prediction_flip_rate": downstream.get("prediction_flip_rate", ""),
                "downstream_note": downstream.get("reason", ""),
                "model_file": result["model_file"],
            })
    print(f"Wrote {json_path}, {csv_path}, and two model files")


if __name__ == "__main__":
    main()
