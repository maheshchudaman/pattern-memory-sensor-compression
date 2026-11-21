# Pattern Memory Sensor Compression

This repository contains the executable evidence for the study **When Pattern Memory Pays Off in Sensor Time Series Compression**. It evaluates a prototype-residual codec as a costed deployment decision rather than claiming that pattern memory is universally superior to established codecs.

The repository reproduces three parts of the study:

1. a learned record-level router with a deployment-level amortization gate;
2. verified Gorilla-style XOR, Zstandard, Sprintz and ZFP comparisons; and
3. downstream HAR preservation with subject-cluster bootstrap intervals and a synthetic recurrence boundary.

## Main findings reproduced by the code

| Dataset | Selected comparison | Compression ratio | Interpretation |
|---|---:|---:|---|
| UCI HAR | ZFP fixed accuracy 0.20 | 3.788 | Largest tested HAR ratio at the reported operating points |
| UCI HAR | Deployed pattern-memory gate | 3.438 | Prototype memory disabled because shared cost was not amortized |
| EEG Eye State | Zstandard level 19 | 4.688 | Largest tested EEG ratio and lossless for the normalized float32 stream |
| EEG Eye State | Deployed pattern-memory gate | 2.139 | Prototype memory disabled because shared cost was not amortized |

For HAR, the reconstructed-versus-original accuracy change is `+0.0044`, with a subject-cluster 95% interval of `[-0.0017, 0.0116]`. The interval includes zero and is not an equivalence test.

## Repository structure

```text
evidence/
  learned_gate_model.py            learned routing and economic gate
  prototype_residual_benchmark.py  direct and prototype-residual records
  codec_baselines.py               Gorilla, Zstandard, Sprintz and ZFP
  synthetic_break_even.py          controlled recurrence experiment
  test_evidence_pipeline.py        round-trip and bootstrap tests
  run_full_evidence.sh             one-command reproduction entry point
  results/                         machine-readable reported results
third_party/sprintz/                pinned upstream Sprintz source
DATA.md                             dataset acquisition and layout
REPRODUCIBILITY.md                  environment and verification details
```

## Requirements

- Python 3.11 or newer
- NumPy 2.3.2
- Matplotlib 3.10.3 for figure regeneration
- Zstandard command-line tool 1.5.7
- ZFP command-line tool 1.0.1
- A C++17 compiler
- An x86-64 processor with AVX2 and BMI2 for the tested Sprintz path

Apple-silicon macOS requires Rosetta 2 because the verified Sprintz helper is compiled for x86-64. Linux x86-64 and Intel macOS build the helper natively.

Install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install `zstd` and `zfp` through the operating-system package manager. Confirm that both commands are on `PATH` before running the complete benchmark.

## Quick verification

The unit tests do not download datasets. Tests requiring an external executable are skipped when that executable is unavailable.

```bash
./evidence/build_sprintz.sh
python -m unittest -v evidence/test_evidence_pipeline.py
```

The five verified checks cover:

- bit-exact Gorilla float32 reconstruction;
- bit-exact Zstandard reconstruction;
- exact Sprintz uint16 reconstruction;
- ZFP fixed-accuracy tolerance compliance; and
- deterministic paired subject-cluster bootstrap sampling.

## Full experiment

Download the datasets as described in [DATA.md](DATA.md), then run:

```bash
./evidence/run_full_evidence.sh \
  "/path/to/unpacked/har" \
  "/path/to/EEG Eye State.arff"
```

The command rebuilds the Sprintz helper, regenerates codec and gate results, and runs all verification tests. Outputs are written to `evidence/results/`.

## Important interpretation limits

- Sprintz is exact for the per-channel training-range uint16 representation, not for the original float32 values. Its quantization error and metadata are reported.
- ZFP and the proposed codec use different formal error constraints. The repository therefore reports observed error for every operating point.
- The EEG chronological split has substantial label shift. EEG is used only for compression analysis; no EEG downstream-classification claim is made.
- The HAR uncertainty calculation contains nine independent test subjects. Its interval is descriptive and does not prove equivalence.
- Timing fields depend on the machine and implementation. The study does not claim throughput leadership.

## Data and privacy

The datasets are not included. They must be obtained from the UCI Machine Learning Repository under their respective terms. The repository contains no participant identifiers beyond the subject grouping already supplied with UCI HAR.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). Update the repository URL, release tag and DOI after publication and Zenodo archiving.

## Licensing

The project code is currently distributed for review with all rights reserved; see [LICENSE](LICENSE). The vendored Sprintz source retains its upstream license in `third_party/sprintz/LICENSE`. Choose an open-source license before public release if reuse beyond review is intended.

