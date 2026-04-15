from __future__ import annotations

import hashlib
import pandas as pd


def build_fingerprint(row: pd.Series) -> str:
    raw = "|".join([
        str(row["Source IP"]),
        str(int(row["Source Port"])),
        str(row["Destination IP"]),
        str(int(row["Destination Port"])),
        str(int(row["Protocol"])),
        str(row["Timestamp_rounded_1s"]),
    ])
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()