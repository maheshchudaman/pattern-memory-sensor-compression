# Reproducibility Record

## Verified environment

The reported evidence was generated with:

- Python 3.13.5
- NumPy 2.3.2
- Matplotlib 3.10.3
- Zstandard CLI 1.5.7 at level 19
- ZFP CLI 1.0.1 in fixed-accuracy mode
- Sprintz upstream commit `a056cdb67d049669875ab5487359aca99ae873ea`
- random seed `20261115`

The machine-readable environment record is stored in `evidence/results/codec_environment.json`.

## Reproduction levels

### Level 1 Unit verification

```bash
./evidence/build_sprintz.sh
python -m unittest -v evidence/test_evidence_pipeline.py
```

This validates codec round trips, ZFP tolerance enforcement and deterministic clustered bootstrapping without rerunning the full datasets.

### Level 2 Full evidence regeneration

```bash
./evidence/run_full_evidence.sh "/path/to/unpacked/har" "/path/to/EEG Eye State.arff"
```

This rebuilds the Sprintz helper, reruns the external codec baselines, trains both deployment gates and executes the tests.

### Level 3 Figure regeneration

After Level 2 completes:

```bash
python evidence/synthetic_break_even.py
python evidence/make_figures.py
```

## Determinism

Randomized operations use seed `20261115`. External codec output is deterministic for a fixed executable, dataset and command configuration. Runtime measurements are not expected to be identical across machines.

## Complete-size accounting

Compression ratios divide the normalized float32 byte count by the complete serialized size. The denominator includes shared normalization values and method-specific metadata. The proposed method also counts gate parameters and the dictionary whenever prototype memory is enabled.

## Independent verification checklist

1. Confirm the Sprintz commit or vendored commit marker.
2. Record Python, NumPy, Zstandard and ZFP versions.
3. Run all five tests and retain the console log.
4. Regenerate JSON and CSV files from clean output directories.
5. Compare the regenerated values with the committed `evidence/results` files.
6. Report hardware separately when interpreting timing fields.

