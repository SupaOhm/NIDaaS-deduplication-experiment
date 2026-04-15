# NIDaaS Deduplication Experiment

Minimal experimental framework for evaluating stream deduplication strategies before downstream IDS-style detection.

## Current scope
- Load and clean CIC-IDS2017 CSV files
- Merge multiple daily files
- Generate stable fingerprints
- Simulate micro-batch processing
- Run NoDedupe baseline
- Train and reuse a Random Forest downstream detector

## Project structure
- `src/load_clean.py` - CSV loading and cleaning
- `src/fingerprint.py` - event fingerprint generation
- `src/microbatch.py` - batch iterator
- `src/no_dedupe.py` - NoDedupe baseline
- `src/metrics.py` - runtime metrics
- `src/train_rf.py` - offline RF training
- `src/rf_infer.py` - RF inference wrapper
- `src/runner.py` - main experiment runner

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt