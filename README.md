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
- `src/bloom.py` - Bloom-only approximate dedupe baseline
- `src/bloom_exact.py` - Bloom front filter with ExactMap confirmation
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
- Bloom-only deduplication
- Bloom+ExactMap hybrid deduplication
- offline Random Forest training and saved-model inference

Current default:
- fingerprint mode: `packet_counts`
- dedupe mode: ExactMap
- downstream detector: saved Random Forest artifacts

Planned next:
- add duplicate injection
- tune Bloom and Bloom+ExactMap parameters

## Runner commands
```bash
# Default next-stage baseline: packet_counts + ExactMap + RF
.venv/bin/python -m src.runner

# Compare NoDedupe, ExactMap, Bloom, and Bloom+ExactMap with packet_counts fixed
.venv/bin/python -m src.runner --dedupe-mode all

# Run Bloom-only with explicit parameters
.venv/bin/python -m src.runner --dedupe-mode bloom --bloom-bits 50000000 --bloom-hashes 4

# Run Bloom+ExactMap with explicit parameters
.venv/bin/python -m src.runner --dedupe-mode bloom_exact --bloom-bits 50000000 --bloom-hashes 4

# Fingerprint sensitivity sweep
.venv/bin/python -m src.runner --fingerprint-mode all

# Original coarse-vs-strict comparison
.venv/bin/python -m src.runner --fingerprint-mode ab
```

Note:
- current RF training uses a random row split and is only a temporary downstream detector setup
- final evaluation should use a stricter split strategy such as day/file holdout
