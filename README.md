# NIDaaS Deduplication Experiment

This repository contains a minimal, reproducible experiment for evaluating
stream deduplication strategies on the CIC-IDS2017 CSV feature dataset before a
fixed Random Forest anomaly detector.

The experiment implemented here is:

```text
CIC-IDS2017 CSV files -> cleaning -> optional duplicate injection -> fingerprinting -> deduplication -> RF anomaly detection
```

The goal is to measure how different deduplication strategies affect event
volume, benign/attack drops, throughput, and fixed RF detection metrics.

## Platform Notes

This repository is developed and primarily documented for macOS/Linux
environments using Unix-like shells. Windows PowerShell setup guidance is
included for convenience, but Windows has not been fully tested in this project.

Keep one copy of this README for all platforms. After the virtual environment is
activated, the Python module commands are the same across platforms.

## Quick Reproduction

After placing the CIC-IDS2017 CSV files in `data/`, the minimum reproduction
sequence is:

macOS / Linux (Unix-like shell):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m src.train_rf
python3 -m src.runner
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train_rf
python -m src.runner
```

There are no standalone executable binaries in this repository. The project is
run through the Python module commands documented below.

## Experiment Scope

This repository evaluates only the cleaning/deduplication + RF anomaly detection
branch.

Snort is not part of this experiment. It is not used by the code, runner,
metrics, dataset preparation, or README commands in this repository. The
experiments here should not be interpreted as measuring Snort performance or
Snort acceleration.

The downstream RF detector is intentionally kept fixed while deduplication is
varied. This keeps deduplication as the main changing variable.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- data/                         # Place CIC-IDS2017 CSV files here
|-- artifacts/                    # RF model artifacts written/read here
|   |-- rf_model.joblib
|   `-- rf_features.joblib
`-- src/
    |-- load_clean.py             # CIC-IDS2017 CSV loading and cleaning
    |-- fingerprint.py            # Event fingerprint variants
    |-- duplicate_injector.py     # Controlled exact replay duplicate injection
    |-- microbatch.py             # Micro-batch iterator
    |-- metrics.py                # Runtime aggregation helpers
    |-- no_dedupe.py              # NoDedupe baseline
    |-- exact_map.py              # Bounded exact-map deduplication
    |-- bloom.py                  # Bloom-only approximate deduplication
    |-- bloom_exact.py            # Bloom front filter + ExactMap confirmation
    |-- train_rf.py               # Offline Random Forest training
    |-- runner.py                 # Main experiment runner
    |-- ids/
    |   `-- rf_detector.py        # Saved RF inference wrapper
    `-- experiments/
        `-- bloom_fairness.py     # Isolated Bloom parameter sweep side experiment
```

## Environment Setup

The code was developed with Python 3.14.2. Use Python 3.11 or newer.

From a fresh clone:

macOS / Linux (Unix-like shell):

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All commands below assume they are run from the repository root after the virtual
environment has been activated.

## Dataset Placement

Place the CIC-IDS2017 CSV files in the repository `data/` directory:

```text
data/
|-- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
|-- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
|-- Friday-WorkingHours-Morning.pcap_ISCX.csv
|-- Monday-WorkingHours.pcap_ISCX.csv
|-- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
|-- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
|-- Tuesday-WorkingHours.pcap_ISCX.csv
`-- Wednesday-workingHours.pcap_ISCX.csv
```

The loader reads every `*.csv` file in `data/` in sorted filename order. With
the expected eight CIC-IDS2017 CSV files present, the cleaned merged dataset is
approximately 2.83 million rows.

The loader keeps only the columns needed for fingerprinting and RF inference,
normalizes column names, parses timestamps, removes invalid rows, and creates a
binary label:

```text
binary_label = 0 for BENIGN
binary_label = 1 for all non-BENIGN labels
```

## RF Model Training

The main runner uses saved RF artifacts from `artifacts/`:

```text
artifacts/rf_model.joblib
artifacts/rf_features.joblib
```

If these files are missing, train the RF model first:

```bash
python3 -m src.train_rf
```

This script:

- loads and cleans the CIC-IDS2017 CSV files from `data/`
- trains a `RandomForestClassifier`
- prints baseline precision, recall, f1, and a classification report
- writes the model and feature list into `artifacts/`

Note: the current RF training script uses a random row split. This is sufficient
for the current deduplication experiment scaffold, but a stricter day/file
holdout split should be used for final evaluation claims.

## Fingerprint Selection

The fixed default fingerprint for the current experiments is:

```text
packet_counts
```

This fingerprint combines the basic flow identity fields with:

```text
Total Fwd Packets
Total Backward Packets
```

It was selected after fingerprint sensitivity testing because it was a better
practical tradeoff than:

- `basic`, which was too coarse and over-collapsed attack traffic
- `duration`, which was too strict and removed almost nothing

Other fingerprint modes remain available for sensitivity checks, but normal
deduplication experiments should keep `packet_counts` fixed.

## Deduplication Modes

The main runner supports four deduplication modes:

```text
no_dedupe
exact_map
bloom
bloom_exact
```

`no_dedupe`

Passes every event through unchanged. This is the no-deduplication baseline.

`exact_map`

Uses a bounded recent set of fingerprints. If a fingerprint was seen recently,
the event is dropped. This is exact within the configured recent state window.

`bloom`

Uses a Bloom filter as an approximate deduplicator. Because Bloom filters can
produce false positives, this mode may drop some non-duplicate records.

`bloom_exact`

Uses Bloom as a front filter and ExactMap for confirmation. Bloom identifies
"maybe seen" fingerprints, then ExactMap confirms before dropping. This avoids
Bloom false-positive drops while preserving a Bloom-filtered path for comparison.

## Main Experiment Commands

Run the default baseline. This command runs the `exact_map` deduplication
baseline with the fixed `packet_counts` fingerprint:

```bash
python3 -m src.runner
```

Default settings:

```text
fingerprint mode: packet_counts
dedupe mode: exact_map
batch size: 5000
ExactMap max recent state: 50000
Bloom bits: 50000000
Bloom hashes: 4
duplicate rate: 0
```

Compare all deduplication modes with the fixed `packet_counts` fingerprint:

```bash
python3 -m src.runner --dedupe-mode all
```

Run one dedupe mode explicitly:

```bash
python3 -m src.runner --dedupe-mode no_dedupe
python3 -m src.runner --dedupe-mode exact_map
python3 -m src.runner --dedupe-mode bloom
python3 -m src.runner --dedupe-mode bloom_exact
```

Run Bloom with explicit parameters:

```bash
python3 -m src.runner --dedupe-mode bloom --bloom-bits 50000000 --bloom-hashes 4
```

Run Bloom+ExactMap with explicit parameters:

```bash
python3 -m src.runner --dedupe-mode bloom_exact --bloom-bits 50000000 --bloom-hashes 4
```

Run the fingerprint sensitivity sweep only if needed:

```bash
python3 -m src.runner --fingerprint-mode all
```

## Controlled Duplicate Injection

The main runner can inject controlled exact replay duplicates before
fingerprinting and deduplication.

Supported duplicate rates:

```text
0, 5, 10, 20 percent
```

Example comparison under 10 percent exact replay pressure:

```bash
python3 -m src.runner --dedupe-mode all --duplicate-rate 10
```

Run the full duplicate-rate sweep:

```bash
for rate in 0 5 10 20; do
  python3 -m src.runner --dedupe-mode all --duplicate-rate "$rate"
done
```

Injection details:

- exact replay only
- deterministic by default with `--duplicate-seed 42`
- duplicate rows are inserted immediately after sampled source rows
- injection happens after loading/cleaning and before fingerprinting

## Isolated Bloom Fairness Experiment

The side experiment in `src/experiments/bloom_fairness.py` exists only for Bloom
parameter fairness checks. It does not change the main runner or main experiment
defaults.

Run the full default Bloom fairness sweep:

```bash
python3 -m src.experiments.bloom_fairness
```

Default sweep:

```text
bloom_bits:   10000000, 25000000, 50000000, 100000000
bloom_hashes: 2, 3, 4, 5
```

Run with controlled duplicate pressure:

```bash
python3 -m src.experiments.bloom_fairness --duplicate-rate 10
```

For faster sanity checks, cap the number of cleaned rows before duplicate
injection:

```bash
python3 -m src.experiments.bloom_fairness \
  --max-rows 50000 \
  --duplicate-rate 10 \
  --bloom-bits 10000000 25000000 50000000 \
  --bloom-hashes 2 3 4
```

The side experiment always uses the fixed `packet_counts` fingerprint and the
same saved RF detector as the main runner.

## Expected Outputs

The scripts print progress and summary tables to stdout.

`src.train_rf` prints:

- total loaded rows
- class distribution
- train/test row counts
- RF precision, recall, and f1
- classification report
- saved artifact paths

`src.runner` prints:

- loaded files and cleaned row counts
- merged row count and class distribution
- duplicate injection settings
- fingerprint mode
- dedupe mode
- total input/output/dropped rows
- throughput and average batch time
- final dedupe state size
- dropped benign and attack rows
- RF precision, recall, and f1
- comparison table when multiple modes are run

`src.experiments.bloom_fairness` prints:

- loaded row counts
- optional row limit information
- duplicate injection settings
- fixed fingerprint mode
- ExactMap reference results
- Bloom and Bloom+ExactMap sweep results
- Bloom memory estimate in bytes

## Current Implementation Notes

- `ExactMap` is the default deduplication method in the main experiment.
- `Bloom` and `Bloom+ExactMap` are included as comparative deduplication methods.
- The isolated Bloom fairness experiment is provided for parameter sensitivity
  checks and does not change the main runner defaults.
- Controlled duplicate injection is included for exact replay-pressure
  evaluation.
- Observations from this repository are implementation-specific and should be
  reported with the dataset split, fingerprint mode, duplicate rate, Bloom
  parameters, and ExactMap state size.

## Troubleshooting

`ModuleNotFoundError` for packages such as `pandas`, `sklearn`, or `joblib`

Activate the virtual environment and install dependencies:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`ModuleNotFoundError: No module named 'src'`

Run scripts as modules from the repository root:

```bash
python3 -m src.runner
python3 -m src.train_rf
```

Do not run `python3 src/runner.py` from inside another directory.

`ValueError: No CSV files found in: data`

Create `data/` in the repository root and place the CIC-IDS2017 CSV files there.
The runner expects `data/*.csv`.

Missing RF artifact files

If `artifacts/rf_model.joblib` or `artifacts/rf_features.joblib` is missing, run:

```bash
python3 -m src.train_rf
```

Encoding warning for the WebAttacks CSV

The loader first tries UTF-8 and then falls back to latin1 when needed. Seeing a
warning like this is expected for the WebAttacks file:

```text
[warn] utf-8 failed for data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv, retrying with latin1
```

Long runtime

The full CIC-IDS2017 dataset has approximately 2.83 million cleaned rows. Bloom
fairness sweeps can be expensive because each parameter combination reruns the
dedupe and RF path. Use `--max-rows` in the side experiment for quick sanity
checks.

## Reproducibility Checklist

Before submitting or reviewing results:

- Confirm Python version: `python3 --version`
- Create and activate `.venv`
- Install `requirements.txt`
- Place the eight CIC-IDS2017 CSV files in `data/`
- Train RF artifacts if they are not already present:
  `python3 -m src.train_rf`
- Keep `packet_counts` as the fixed fingerprint for dedupe comparisons
- Record dedupe mode, duplicate rate, Bloom bits, Bloom hashes, and ExactMap
  `max_recent`
- Run commands from the repository root
- Save stdout tables for reported experimental results
