# Dataset Setup

The repository does not redistribute either dataset.

## UCI Human Activity Recognition Using Smartphones

- UCI record DOI: `10.24432/C54S4K`
- Expected content: the directory `UCI HAR Dataset` with the original `train` and `test` folders.

Pass the parent directory containing `UCI HAR Dataset` to `--har-root` or as the first argument to `run_full_evidence.sh`.

Expected files include:

```text
UCI HAR Dataset/
  train/Inertial Signals/
  train/y_train.txt
  test/Inertial Signals/
  test/y_test.txt
  test/subject_test.txt
```

The experiment uses the official subject-independent train and test partition, nine inertial channels, 7,352 training windows and 2,947 test windows.

## UCI EEG Eye State

- UCI record DOI: `10.24432/C57G7J`
- Expected file: the original ARFF dataset file.

Pass the ARFF path to `--eeg-arff` or as the second argument to `run_full_evidence.sh`.

The experiment creates 64-sample windows with stride 32 and uses the first 60% chronologically for fitting. Because the resulting chronological segments have substantial label shift, EEG is evaluated only for compression.

## Integrity checks

The loaders fail if required files are absent or malformed. Dataset files should remain outside version control. Record the archive checksum and retrieval date in any independent reproduction report.

