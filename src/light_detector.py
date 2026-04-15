from __future__ import annotations
import pandas as pd

def run_light_detector(batch: pd.DataFrame) -> pd.DataFrame:
    out = batch.copy()

    pred = (
        (out["SYN Flag Count"] > 2) |
        (out["Total Fwd Packets"] > 20) |
        (out["Average Packet Size"] > 800)
    )

    out["pred_label"] = pred.astype(int)
    return out