#!/usr/bin/env python3
"""Verified codec baselines for the pattern-memory evidence study.

All codecs receive the same training-normalized, clipped float32 test tensor.
Lossless codecs are verified bit-for-bit.  Sprintz operates on a documented
training-range uint16 quantization because the official codec accepts integer
time series.  ZFP uses its fixed-accuracy mode and is checked against the
requested absolute tolerance after decoding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from prototype_residual_benchmark import load_eeg, load_har, robust_standardize


NORMALIZATION_FLOATS_PER_CHANNEL = 4  # mean, std, clip low, clip high


class BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.accumulator = 0
        self.nbits = 0

    def write(self, value: int, width: int) -> None:
        if width == 0:
            return
        value &= (1 << width) - 1
        self.accumulator = (self.accumulator << width) | value
        self.nbits += width
        while self.nbits >= 8:
            shift = self.nbits - 8
            self.data.append((self.accumulator >> shift) & 0xFF)
            self.accumulator &= (1 << shift) - 1 if shift else 0
            self.nbits = shift

    def finish(self) -> bytes:
        if self.nbits:
            self.data.append((self.accumulator << (8 - self.nbits)) & 0xFF)
            self.accumulator = 0
            self.nbits = 0
        return bytes(self.data)


class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.position = 0

    def read(self, width: int) -> int:
        value = 0
        for _ in range(width):
            byte = self.data[self.position >> 3]
            bit = (byte >> (7 - (self.position & 7))) & 1
            value = (value << 1) | bit
            self.position += 1
        return value


def leading_zeros32(value: int) -> int:
    return 32 - value.bit_length() if value else 32


def trailing_zeros32(value: int) -> int:
    return (value & -value).bit_length() - 1 if value else 32


def gorilla_encode(array: np.ndarray) -> bytes:
    """Gorilla-style XOR value coding for uniformly sampled float32 channels."""
    tensor = np.ascontiguousarray(array, dtype="<f4")
    channels = tensor.shape[-1]
    matrix = tensor.reshape(-1, channels).view("<u4")
    writer = BitWriter()
    for channel in range(channels):
        values = matrix[:, channel]
        writer.write(int(values[0]), 32)
        previous = int(values[0])
        previous_leading = 0
        previous_trailing = 0
        have_window = False
        for raw in values[1:]:
            current = int(raw)
            xor = current ^ previous
            if xor == 0:
                writer.write(0, 1)
            else:
                writer.write(1, 1)
                leading = min(leading_zeros32(xor), 31)
                trailing = trailing_zeros32(xor)
                if have_window and leading >= previous_leading and trailing >= previous_trailing:
                    writer.write(0, 1)
                    width = 32 - previous_leading - previous_trailing
                    writer.write(xor >> previous_trailing, width)
                else:
                    writer.write(1, 1)
                    width = 32 - leading - trailing
                    writer.write(leading, 5)
                    writer.write(0 if width == 32 else width, 6)
                    writer.write(xor >> trailing, width)
                    previous_leading = leading
                    previous_trailing = trailing
                    have_window = True
            previous = current
    # Header has four dimension slots; absent slots are zero.
    dimensions = list(tensor.shape) + [0] * (4 - tensor.ndim)
    header = struct.pack("<4sB3xIIII", b"GRL1", tensor.ndim, *dimensions[:4])
    return header + writer.finish()


def gorilla_decode(payload: bytes) -> np.ndarray:
    magic, ndim, d0, d1, d2, d3 = struct.unpack("<4sB3xIIII", payload[:24])
    if magic != b"GRL1" or ndim < 1 or ndim > 4:
        raise ValueError("invalid Gorilla stream header")
    shape = (d0, d1, d2, d3)[:ndim]
    channels = shape[-1]
    rows = int(np.prod(shape[:-1]))
    matrix = np.empty((rows, channels), dtype="<u4")
    reader = BitReader(payload[24:])
    for channel in range(channels):
        previous = reader.read(32)
        matrix[0, channel] = previous
        previous_leading = 0
        previous_trailing = 0
        have_window = False
        for row in range(1, rows):
            if reader.read(1) == 0:
                current = previous
            else:
                if reader.read(1) == 0:
                    if not have_window:
                        raise ValueError("Gorilla stream reused a missing XOR window")
                    leading = previous_leading
                    trailing = previous_trailing
                    width = 32 - leading - trailing
                else:
                    leading = reader.read(5)
                    encoded_width = reader.read(6)
                    width = 32 if encoded_width == 0 else encoded_width
                    trailing = 32 - leading - width
                    previous_leading = leading
                    previous_trailing = trailing
                    have_window = True
                xor = reader.read(width) << trailing
                current = previous ^ xor
            matrix[row, channel] = current
            previous = current
    return matrix.reshape(shape).view("<f4")


def run_zstd(array: np.ndarray, executable: str) -> tuple[bytes, np.ndarray, float, float]:
    raw = np.ascontiguousarray(array, dtype="<f4").tobytes()
    start = time.perf_counter()
    encoded = subprocess.run(
        [executable, "-q", "-19", "-c"], input=raw, capture_output=True, check=True
    ).stdout
    encode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    decoded = subprocess.run(
        [executable, "-q", "-d", "-c"], input=encoded, capture_output=True, check=True
    ).stdout
    decode_seconds = time.perf_counter() - start
    if decoded != raw:
        raise AssertionError("Zstandard round-trip was not bit-exact")
    return encoded, np.frombuffer(decoded, dtype="<f4").reshape(array.shape), encode_seconds, decode_seconds


def quantize_uint16(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_flat = train.reshape(-1, train.shape[-1]).astype(np.float64)
    low = train_flat.min(axis=0)
    high = train_flat.max(axis=0)
    span = np.maximum(high - low, 1e-12)
    scaled = (test - low) * (65535.0 / span)
    quantized = np.rint(np.clip(scaled, 0, 65535)).astype("<u2")
    reconstructed = (quantized.astype(np.float64) * (span / 65535.0) + low).astype(np.float32)
    return quantized, reconstructed, np.stack([low, high]).astype("<f4")


def run_sprintz(train: np.ndarray, test: np.ndarray, executable: Path):
    quantized, reconstructed, quantization_metadata = quantize_uint16(train, test)
    matrix = np.ascontiguousarray(quantized.reshape(-1, quantized.shape[-1]), dtype="<u2")
    with tempfile.TemporaryDirectory(prefix="sprintz_") as directory:
        directory = Path(directory)
        raw_path = directory / "input.u16"
        compressed_path = directory / "stream.sprintz"
        decoded_path = directory / "decoded.u16"
        raw_path.write_bytes(matrix.tobytes())
        common = [str(matrix.shape[1]), str(matrix.size)]
        start = time.perf_counter()
        prefix = ["arch", "-x86_64"] if platform.system() == "Darwin" and platform.machine() == "arm64" else []
        subprocess.run(
            [*prefix, str(executable), "compress", str(raw_path),
             str(compressed_path), *common], check=True, capture_output=True
        )
        encode_seconds = time.perf_counter() - start
        start = time.perf_counter()
        subprocess.run(
            [*prefix, str(executable), "decompress", str(compressed_path),
             str(decoded_path), *common], check=True, capture_output=True
        )
        decode_seconds = time.perf_counter() - start
        decoded = np.frombuffer(decoded_path.read_bytes(), dtype="<u2").reshape(matrix.shape)
        if not np.array_equal(decoded, matrix):
            raise AssertionError("official Sprintz integer round-trip was not bit-exact")
        encoded = compressed_path.read_bytes()
    return encoded, reconstructed, quantization_metadata, encode_seconds, decode_seconds


def run_zfp(array: np.ndarray, tolerance: float, executable: str):
    tensor = np.ascontiguousarray(array, dtype="<f4")
    if tensor.ndim != 3:
        raise ValueError("ZFP runner expects [windows, samples, channels]")
    windows, samples, channels = tensor.shape
    with tempfile.TemporaryDirectory(prefix="zfp_") as directory:
        directory = Path(directory)
        raw_path = directory / "input.f32"
        compressed_path = directory / "stream.zfp"
        decoded_path = directory / "decoded.f32"
        raw_path.write_bytes(tensor.tobytes())
        start = time.perf_counter()
        subprocess.run(
            [executable, "-q", "-h", "-f", "-3", str(channels), str(samples),
             str(windows), "-a", str(tolerance), "-i", str(raw_path), "-z", str(compressed_path)],
            check=True, capture_output=True,
        )
        encode_seconds = time.perf_counter() - start
        start = time.perf_counter()
        subprocess.run(
            [executable, "-q", "-h", "-z", str(compressed_path), "-o", str(decoded_path)],
            check=True, capture_output=True,
        )
        decode_seconds = time.perf_counter() - start
        encoded = compressed_path.read_bytes()
        reconstructed = np.frombuffer(decoded_path.read_bytes(), dtype="<f4").reshape(tensor.shape).copy()
    max_abs_error = float(np.max(np.abs(tensor - reconstructed)))
    # ZFP's guarantee includes a small floating-point allowance.
    if max_abs_error > tolerance * 1.0001 + 1e-7:
        raise AssertionError(f"ZFP error {max_abs_error} exceeds tolerance {tolerance}")
    return encoded, reconstructed, encode_seconds, decode_seconds, max_abs_error


def window_nrmse(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    original_flat = original.reshape(len(original), -1)
    reconstructed_flat = reconstructed.reshape(len(reconstructed), -1)
    denominator = np.maximum(np.std(original_flat, axis=1), 0.10)
    return np.sqrt(np.mean((original_flat - reconstructed_flat) ** 2, axis=1)) / denominator


def result_row(dataset: str, method: str, original: np.ndarray, reconstructed: np.ndarray,
               payload_bytes: int, shared_metadata_bytes: int, extra_metadata_bytes: int,
               encode_seconds: float, decode_seconds: float, exact: bool, **extra):
    raw_bytes = original.astype("<f4", copy=False).nbytes
    total_bytes = payload_bytes + shared_metadata_bytes + extra_metadata_bytes
    errors = window_nrmse(original, reconstructed)
    row = {
        "dataset": dataset,
        "method": method,
        "representation": "training-normalized_clipped_float32",
        "payload_bytes": int(payload_bytes),
        "shared_metadata_bytes": int(shared_metadata_bytes),
        "method_metadata_bytes": int(extra_metadata_bytes),
        "encoded_bytes": int(total_bytes),
        "raw_bytes": int(raw_bytes),
        "compression_ratio": float(raw_bytes / total_bytes),
        "mean_nrmse": float(np.mean(errors)),
        "p95_nrmse": float(np.quantile(errors, 0.95)),
        "max_nrmse": float(np.max(errors)),
        "lossless_wrt_input_representation": bool(exact),
        "roundtrip_verified": True,
        "encode_seconds": float(encode_seconds),
        "decode_seconds": float(decode_seconds),
    }
    row.update(extra)
    return row


def benchmark_dataset(name: str, train: np.ndarray, test: np.ndarray, args) -> list[dict]:
    train_z, test_z, *_ = robust_standardize(train, test)
    channels = test_z.shape[-1]
    # Match the proposed codec: mean, standard deviation, and clipping bounds
    # are serialized as float64 values for every channel.
    shared_metadata = NORMALIZATION_FLOATS_PER_CHANNEL * channels * 8
    rows = []

    start = time.perf_counter()
    gorilla = gorilla_encode(test_z)
    gorilla_encode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    gorilla_reconstruction = gorilla_decode(gorilla)
    gorilla_decode_seconds = time.perf_counter() - start
    if not np.array_equal(gorilla_reconstruction.view("<u4"), test_z.view("<u4")):
        raise AssertionError("Gorilla round-trip was not bit-exact")
    rows.append(result_row(name, "gorilla_xor_float32", test_z, gorilla_reconstruction,
                           len(gorilla), shared_metadata, 0,
                           gorilla_encode_seconds, gorilla_decode_seconds, True,
                           timestamp_policy="omitted_uniform_sampling"))

    zstd, zstd_reconstruction, enc_s, dec_s = run_zstd(test_z, args.zstd)
    rows.append(result_row(name, "zstandard_19_float32", test_z, zstd_reconstruction,
                           len(zstd), shared_metadata, 0, enc_s, dec_s, True,
                           zstd_level=19))

    sprintz, sprintz_reconstruction, sprintz_meta, enc_s, dec_s = run_sprintz(
        train_z, test_z, args.sprintz
    )
    rows.append(result_row(name, "official_sprintz_delta_rle_uint16", test_z,
                           sprintz_reconstruction, len(sprintz), shared_metadata,
                           sprintz_meta.nbytes, enc_s, dec_s, False,
                           integer_roundtrip_exact=True,
                           quantization="per-channel training-range uint16"))

    for tolerance in args.tolerances:
        zfp, zfp_reconstruction, enc_s, dec_s, max_error = run_zfp(
            test_z, tolerance, args.zfp
        )
        rows.append(result_row(
            name, f"zfp_fixed_accuracy_abs_{tolerance:g}", test_z, zfp_reconstruction,
            len(zfp), shared_metadata, 0, enc_s, dec_s, False,
            requested_absolute_tolerance=tolerance,
            observed_max_absolute_error=max_error,
            zfp_header_embedded=True,
        ))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--har-root", type=Path, required=True)
    parser.add_argument("--eeg-arff", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evidence/results"))
    parser.add_argument(
        "--tolerances", type=float, nargs="+", default=[0.05, 0.20],
        help="ZFP fixed-accuracy operating points in normalized signal units",
    )
    parser.add_argument("--sprintz", type=Path, default=Path("evidence/sprintz_cli"))
    parser.add_argument("--zstd", default=shutil.which("zstd") or "/opt/homebrew/bin/zstd")
    parser.add_argument("--zfp", default=shutil.which("zfp") or "/opt/homebrew/bin/zfp")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("UCI_HAR", *load_har(args.har_root)),
        ("UCI_EEG_Eye_State", *load_eeg(args.eeg_arff)),
    ]
    rows = []
    for name, train, test in datasets:
        print(f"Running verified codec baselines for {name}", flush=True)
        rows.extend(benchmark_dataset(name, train, test, args))

    json_path = args.output_dir / "codec_baselines.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = args.output_dir / "codec_baselines.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    repository_root = Path(__file__).resolve().parent.parent

    def first_line(command):
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        lines = (completed.stdout or completed.stderr).splitlines()
        if not lines:
            raise RuntimeError(f"version command produced no output: {command}")
        return lines[0]

    sprintz_root = repository_root / "third_party" / "sprintz"
    git_commit = subprocess.run(
        ["git", "-C", str(sprintz_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if git_commit.returncode == 0:
        sprintz_commit = git_commit.stdout.strip()
    else:
        sprintz_commit = (repository_root / "third_party" / "SPRINTZ_COMMIT").read_text(
            encoding="utf-8"
        ).strip()
    sprintz_sha256 = hashlib.sha256(args.sprintz.read_bytes()).hexdigest()
    manifest = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "sprintz": {
            "repository": "https://github.com/dblalock/sprintz",
            "commit": sprintz_commit,
            "helper": str(args.sprintz),
            "helper_sha256": sprintz_sha256,
            "build": "x86_64 AVX2 BMI2; Apple silicon executes through macOS arch -x86_64",
        },
        "zstandard": {
            "executable": args.zstd,
            "version": first_line([args.zstd, "--version"]),
            "level": 19,
        },
        "zfp": {
            "executable": args.zfp,
            "version": first_line([args.zfp, "--help"]),
            "mode": "fixed accuracy with embedded header",
            "absolute_tolerances": args.tolerances,
        },
        "gorilla": {
            "implementation": "codec_baselines.py float32 XOR value stream",
            "timestamps": "omitted because sampling is uniform",
        },
        "eeg_policy": "compression analysis only; no downstream metric",
    }
    manifest_path = args.output_dir / "codec_environment.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}, {csv_path}, and {manifest_path}")


if __name__ == "__main__":
    main()
