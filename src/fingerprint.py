from __future__ import annotations

import hashlib
import pandas as pd


def _hash_parts(parts: list[str]) -> str:
    raw = "|".join(parts)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()


def _basic_parts(row: pd.Series) -> list[str]:
    return [
        str(row["Source IP"]),
        str(int(row["Source Port"])),
        str(row["Destination IP"]),
        str(int(row["Destination Port"])),
        str(int(row["Protocol"])),
        str(row["Timestamp_rounded_1s"]),
    ]


def _duration_bucket(duration: int) -> str:
    if duration <= 0:
        return "0"

    # Power-of-two buckets keep duration useful without requiring exact equality.
    return str(duration.bit_length() - 1)


def build_fingerprint_basic(row: pd.Series) -> str:
    return _hash_parts(_basic_parts(row))


def build_fingerprint_with_duration(row: pd.Series) -> str:
    return _hash_parts(_basic_parts(row) + [
        str(int(row["Flow Duration"])),
    ])


def build_fingerprint_with_duration_bucket(row: pd.Series) -> str:
    return _hash_parts(_basic_parts(row) + [
        _duration_bucket(int(row["Flow Duration"])),
    ])


def build_fingerprint_with_packet_counts(row: pd.Series) -> str:
    return _hash_parts(_basic_parts(row) + [
        str(int(row["Total Fwd Packets"])),
        str(int(row["Total Backward Packets"])),
    ])


def build_fingerprint(row: pd.Series) -> str:
    return build_fingerprint_with_packet_counts(row)
