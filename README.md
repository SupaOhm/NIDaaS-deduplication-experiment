# NIDaaS Deduplication Experiment

Minimal experimental framework for evaluating stream deduplication strategies before downstream IDS-style detection.

## Current scope
- Load and clean CIC-IDS2017 CSV files
- Merge multiple daily files
- Generate stable fingerprints
- Simulate micro-batch processing
- Run NoDedupe baseline
- Run ExactMap deduplication before a fixed Random Forest detector

## Project structure
- `src/load_clean.py` - CSV loading and cleaning
- `src/fingerprint.py` - event fingerprint generation
- `src/microbatch.py` - batch iterator
- `src/no_dedupe.py` - NoDedupe baseline
- `src/exact_map.py` - bounded ExactMap dedupe baseline
- `src/metrics.py` - runtime metrics
- `src/train_rf.py` - offline RF training
- `src/runner.py` - main experiment runner

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Current status
Working:
- multi-file CIC-IDS2017 loading and cleaning
- merged dataset creation
- fingerprint generation with selectable modes
- micro-batch simulation
- NoDedupe baseline
- ExactMap deduplication
- offline Random Forest training and saved-model inference

Current default:
- fingerprint mode: `packet_counts`
- dedupe mode: ExactMap
- downstream detector: saved Random Forest artifacts

Planned next:
- keep `packet_counts` fixed while evaluating Bloom and Bloom+ExactMap
- add duplicate injection
- evaluate Bloom and Bloom+ExactMap dedup variants

## Runner commands
```bash
# Default next-stage baseline: packet_counts + ExactMap + RF
.venv/bin/python -m src.runner

# Fingerprint sensitivity sweep
.venv/bin/python -m src.runner --fingerprint-mode all

# Original coarse-vs-strict comparison
.venv/bin/python -m src.runner --fingerprint-mode ab
```

Note:
- current RF training uses a random row split and is only a temporary downstream detector setup
- final evaluation should use a stricter split strategy such as day/file holdout
