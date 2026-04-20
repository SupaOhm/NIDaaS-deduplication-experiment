from __future__ import annotations

import pandas as pd

from src.bloom import BloomDedupe
from src.exact_map import ExactMapDedupe


class BloomExactDedupe:
    def __init__(
        self,
        bit_size: int = 50_000_000,
        num_hashes: int = 4,
        max_recent: int = 50000,
    ) -> None:
        self.bloom = BloomDedupe(bit_size=bit_size, num_hashes=num_hashes)
        self.exact = ExactMapDedupe(max_recent=max_recent)
        self.bloom_maybe_seen = 0
        self.bloom_false_positives = 0

    def reset(self) -> None:
        self.bloom.reset()
        self.exact.reset()
        self.bloom_maybe_seen = 0
        self.bloom_false_positives = 0

    def process_batch(
        self,
        batch: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        keep_mask: list[bool] = []
        dropped = 0

        for fp in batch["fingerprint"]:
            if self.bloom.may_contain(fp):
                self.bloom_maybe_seen += 1
                if self.exact.contains(fp):
                    keep_mask.append(False)
                    dropped += 1
                    continue

                self.bloom_false_positives += 1

            keep_mask.append(True)
            self.bloom.add(fp)
            self.exact.insert(fp)

        kept = batch.loc[keep_mask].copy()
        dropped_df = batch.loc[[not x for x in keep_mask]].copy()

        input_events = len(batch)
        output_events = len(kept)

        return kept, dropped_df, {
            "input_events": input_events,
            "output_events": output_events,
            "dropped_duplicates": dropped,
            "drop_rate": dropped / input_events if input_events > 0 else 0.0,
            "state_size": len(self.exact.seen),
            "bit_size": self.bloom.bit_size,
            "num_hashes": self.bloom.num_hashes,
            "bloom_bits_set": self.bloom.bits_set,
            "bloom_inserted": self.bloom.inserted_count,
            "bloom_maybe_seen": self.bloom_maybe_seen,
            "bloom_false_positives": self.bloom_false_positives,
        }
