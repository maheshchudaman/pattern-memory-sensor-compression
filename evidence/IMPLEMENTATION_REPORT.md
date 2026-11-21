# Verified Codec and Downstream-Evaluation Implementation Report

Run date: 2026-11-15  
Datasets: UCI HAR and UCI EEG Eye State

## Implemented changes

1. The official Sprintz repository is vendored at `third_party/sprintz` at
   commit `a056cdb67d049669875ab5487359aca99ae873ea`. Its delta+RLE C++ codec is
   called through an x86_64 AVX2/BMI2 helper. Each decoded uint16 stream is
   checked for exact equality. Float signals use per-channel, training-range
   uint16 quantization, whose metadata and reconstruction error are reported.
2. `codec_baselines.py` implements Gorilla-style lossless float32 XOR coding,
   including reuse of leading/trailing-zero windows. Uniform sampling makes a
   timestamp stream unnecessary. Decoded float bits are checked exactly.
3. Zstandard 1.5.7 is integrated at level 19. Decoded bytes are checked exactly.
4. ZFP 1.0.1 is integrated in fixed-accuracy mode with embedded headers at
   absolute tolerances 0.05 and 0.20. The maximum decoded absolute error is
   checked against the requested bound.
5. HAR now reports paired subject-cluster percentile intervals from 10,000
   resamples of the nine official test subjects. All windows from a sampled
   subject remain together.
6. EEG downstream classification is disabled. The chronological EEG split is
   retained only for compression analysis, and the machine-readable output
   records this policy and its label-shift reason.

## Verified compression results

All ratios include shared normalization metadata and any method-specific
metadata. Error is measured against the normalized, training-clipped float32
representation supplied to every codec.

| Dataset | Method | Ratio | Mean NRMSE | P95 NRMSE | Exact input round-trip |
|---|---:|---:|---:|---:|---:|
| UCI HAR | Gorilla XOR | 0.944 | 0 | 0 | Yes |
| UCI HAR | Zstandard-19 | 1.967 | 0 | 0 | Yes |
| UCI HAR | Sprintz uint16 | 2.601 | 0.000072 | 0.000185 | Integer stream only |
| UCI HAR | ZFP abs. 0.05 | 2.922 | 0.002432 | 0.006500 | No |
| UCI HAR | ZFP abs. 0.20 | 3.788 | 0.009612 | 0.025642 | No |
| UCI EEG | Gorilla XOR | 1.028 | 0 | 0 | Yes |
| UCI EEG | Zstandard-19 | 4.688 | 0 | 0 | Yes |
| UCI EEG | Sprintz uint16 | 2.315 | 0.000044 | 0.000060 | Integer stream only |
| UCI EEG | ZFP abs. 0.05 | 2.882 | 0.001713 | 0.002736 | No |
| UCI EEG | ZFP abs. 0.20 | 3.626 | 0.006845 | 0.010929 | No |

The learned gate ratios are 3.438 on HAR and 2.139 on EEG. It disables
prototype memory on both datasets because validation savings do not amortize
the dictionary and gate parameters.

## HAR downstream result

| Metric | Original | Reconstructed | Paired change | Subject-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Accuracy | 0.8853 | 0.8897 | +0.0044 | [-0.0017, 0.0116] |
| Macro F1 | 0.8870 | 0.8914 | +0.0045 | [-0.0020, 0.0120] |

Both intervals include zero. The defensible conclusion is that this experiment
does not detect a downstream degradation; it does not prove equivalence or an
improvement.

## Scientific consequence

The stronger baselines do not support a codec-superiority claim. ZFP at
absolute tolerance 0.20 exceeds the learned gate's HAR compression ratio while
also producing lower observed NRMSE, and lossless Zstandard substantially
exceeds the gate on EEG. The paper should therefore be framed as an empirical
study of when pattern memory does or does not amortize—not as a new
state-of-the-art compression codec.

## Reproduction

```bash
evidence/run_full_evidence.sh "/path/to/unpacked/har" "/path/to/EEG Eye State.arff"
```

The command rebuilds Sprintz, regenerates codec and downstream results, and
runs the five round-trip/error/bootstrap tests.
