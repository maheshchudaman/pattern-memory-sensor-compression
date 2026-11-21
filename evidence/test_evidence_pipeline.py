#!/usr/bin/env python3
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codec_baselines import (  # noqa: E402
    gorilla_decode,
    gorilla_encode,
    run_sprintz,
    run_zfp,
    run_zstd,
)
from learned_gate_model import subject_cluster_bootstrap  # noqa: E402


class CodecRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = np.random.default_rng(19).normal(size=(8, 16, 3)).astype(np.float32)
        cls.values[1:4, 3:9, 1] = cls.values[0, 0, 1]

    def test_gorilla_is_bit_exact(self):
        reconstructed = gorilla_decode(gorilla_encode(self.values))
        self.assertTrue(np.array_equal(self.values.view(np.uint32), reconstructed.view(np.uint32)))

    @unittest.skipUnless(shutil.which("zstd"), "zstd is not installed")
    def test_zstd_is_bit_exact(self):
        _, reconstructed, _, _ = run_zstd(self.values, shutil.which("zstd"))
        self.assertTrue(np.array_equal(self.values.view(np.uint32), reconstructed.view(np.uint32)))

    @unittest.skipUnless(shutil.which("zfp"), "zfp is not installed")
    def test_zfp_respects_fixed_accuracy(self):
        tolerance = 0.05
        _, reconstructed, _, _, _ = run_zfp(self.values, tolerance, shutil.which("zfp"))
        self.assertLessEqual(float(np.max(np.abs(self.values - reconstructed))), tolerance * 1.0001 + 1e-7)

    @unittest.skipUnless(Path("evidence/sprintz_cli").exists(), "Sprintz helper is not built")
    def test_official_sprintz_integer_roundtrip(self):
        payload, reconstructed, metadata, _, _ = run_sprintz(
            self.values, self.values, Path("evidence/sprintz_cli").resolve()
        )
        self.assertGreater(len(payload), 0)
        self.assertEqual(metadata.shape, (2, self.values.shape[-1]))
        self.assertLess(float(np.max(np.abs(self.values - reconstructed))), 1e-3)


class SubjectBootstrapTests(unittest.TestCase):
    def test_subject_resampling_is_deterministic_and_paired(self):
        truth = np.array([0, 0, 1, 1, 0, 1])
        original = np.array([0, 1, 1, 1, 0, 0])
        reconstructed = np.array([0, 0, 1, 0, 0, 1])
        subjects = np.array([1, 1, 2, 2, 3, 3])
        first = subject_cluster_bootstrap(truth, original, reconstructed, subjects, repetitions=200, seed=7)
        second = subject_cluster_bootstrap(truth, original, reconstructed, subjects, repetitions=200, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["subjects"], 3)


if __name__ == "__main__":
    unittest.main()
