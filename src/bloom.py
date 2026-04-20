from __future__ import annotations

import hashlib
import pandas as pd


class BloomDedupe:
    def __init__(self, bit_size: int = 50_000_000, num_hashes: int = 4) -> None:
        if bit_size <= 0:
            raise ValueError("bit_size must be positive")
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")

        self.bit_size = bit_size
        self.num_hashes = num_hashes
        self.bits = bytearray((bit_size + 7) // 8)
        self.bits_set = 0
        self.inserted_count = 0

    def reset(self) -> None:
        self.bits = bytearray(len(self.bits))
        self.bits_set = 0
        self.inserted_count = 0

    def _positions(self, fp: str) -> list[int]:
        digest = hashlib.blake2b(fp.encode("utf-8"), digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:], "little") or 1
        return [
            (h1 + i * h2) % self.bit_size
            for i in range(self.num_hashes)
        ]

    def _has_bit(self, position: int) -> bool:
        byte_index = position // 8
        bit_mask = 1 << (position % 8)
        return bool(self.bits[byte_index] & bit_mask)

    def _set_bit(self, position: int) -> None:
        byte_index = position // 8
        bit_mask = 1 << (position % 8)
        if not self.bits[byte_index] & bit_mask:
            self.bits_set += 1
        self.bits[byte_index] |= bit_mask

    def may_contain(self, fp: str) -> bool:
        return all(self._has_bit(position) for position in self._positions(fp))

    def add(self, fp: str) -> None:
        for position in self._positions(fp):
            self._set_bit(position)
        self.inserted_count += 1

    def process_batch(
        self,
        batch: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        keep_mask: list[bool] = []
        dropped = 0

        for fp in batch["fingerprint"]:
            if self.may_contain(fp):
                keep_mask.append(False)
                dropped += 1
            else:
                keep_mask.append(True)
                self.add(fp)

        kept = batch.loc[keep_mask].copy()
        dropped_df = batch.loc[[not x for x in keep_mask]].copy()

        input_events = len(batch)
        output_events = len(kept)

        return kept, dropped_df, {
            "input_events": input_events,
            "output_events": output_events,
            "dropped_duplicates": dropped,
            "drop_rate": dropped / input_events if input_events > 0 else 0.0,
            "state_size": self.bits_set,
            "inserted_count": self.inserted_count,
            "bit_size": self.bit_size,
            "num_hashes": self.num_hashes,
        }
