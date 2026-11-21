# Adaptive Pattern Memory Coding Model

This folder contains an executable research model. It is not a theoretical
brain analogy. The model learns whether a prototype dictionary can reduce
encoded bytes after its storage cost is included.

## Model behavior

1. Standardize multichannel windows using training-only statistics.
2. Learn a deterministic prototype dictionary.
3. Measure direct sparse-transform and prototype-residual records on training
   windows.
4. Train a lightweight logistic gate from time-domain recurrence and residual
   features.
5. Apply a global economic gate. If validation savings cannot amortize the
   dictionary and gate parameters, disable prototype memory for the deployment.
6. Reconstruct test windows and evaluate downstream classification decisions
   on the official subject-independent HAR split.

The current real-dataset models disable prototype memory because the validation
streams do not recover the dictionary cost. This is an intended model decision,
not a failed run.

## Reproduce

Download the official UCI Human Activity Recognition Using Smartphones and EEG
Eye State datasets, then run:

```bash
python3 evidence/learned_gate_model.py \
  --har-root "/path/to/unpacked/har" \
  --eeg-arff "/path/to/EEG Eye State.arff" \
  --output-dir evidence/results
```

Build and run the codec baselines. The repository already contains the pinned
Sprintz source revision:

```bash
evidence/build_sprintz.sh
python3 evidence/codec_baselines.py \
  --har-root "/path/to/unpacked/har" \
  --eeg-arff "/path/to/EEG Eye State.arff" \
  --output-dir evidence/results
```

Or rebuild both analyses and run all verification tests with one command:

```bash
evidence/run_full_evidence.sh "/path/to/unpacked/har" "/path/to/EEG Eye State.arff"
```

The tested Sprintz path is x86-64/AVX2/BMI2-specific. On Apple silicon the build
script creates an x86-64 helper that runs through macOS translation. On Linux
x86-64 and Intel macOS it builds natively. Zstandard and ZFP
must be available on `PATH`; the verified run used Zstandard 1.5.7 and ZFP
1.0.1. Gorilla-style XOR coding is implemented directly in
`codec_baselines.py` for uniformly sampled float32 channels (timestamps are
therefore omitted).

## Outputs

- `apmc_gate_UCI_HAR.npz`: saved HAR coding-gate model.
- `apmc_gate_UCI_EEG_Eye_State.npz`: saved EEG coding-gate model.
- `learned_gate_results.csv`: compact evidence table.
- `learned_gate_results.json`: complete gate and downstream metrics.
- `codec_baselines.csv`: verified Sprintz, Gorilla, Zstandard and ZFP results.
- `codec_baselines.json`: complete codec settings, error and timing fields.
- `codec_environment.json`: exact codec versions, Sprintz commit and helper hash.
- `IMPLEMENTATION_REPORT.md`: verified numerical summary and manuscript-safe interpretation.
- `codec_baseline_comparison.png`: 300-dpi baseline comparison figure.

## Interpretation limits

- HAR reports paired 95% percentile intervals from 10,000 resamples of all
  windows belonging to each of the nine test subjects.
- The EEG Eye State chronological split has severe label shift. The executable
  pipeline now suppresses EEG downstream metrics and uses EEG only for
  compression evidence.
- Sprintz is lossless with respect to the per-channel training-range uint16
  representation, not the source float32 values. Its quantization error and
  metadata are included in the output.
- ZFP fixed-accuracy and the pattern-memory method use different formal error
  constraints (absolute error versus per-window NRMSE); report both actual
  error columns and do not imply exact error-budget equivalence.
- Add a third longer public stream and repeated throughput measurements.
