#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <HAR-unpacked-root> <EEG-Eye-State.arff>" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
har_root=$1
eeg_arff=$2

cd "$repo_root"
python_executable=${PYTHON:-python3}
evidence/build_sprintz.sh
"$python_executable" evidence/codec_baselines.py \
  --har-root "$har_root" \
  --eeg-arff "$eeg_arff" \
  --output-dir evidence/results
"$python_executable" evidence/learned_gate_model.py \
  --har-root "$har_root" \
  --eeg-arff "$eeg_arff" \
  --output-dir evidence/results
"$python_executable" -m unittest -v evidence/test_evidence_pipeline.py
