from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


MODEL_PATH = Path("artifacts/rf_model.joblib")
FEATURES_PATH = Path("artifacts/rf_features.joblib")


class RFDetector:
    def __init__(self) -> None:
        self.model = joblib.load(MODEL_PATH)
        self.features = joblib.load(FEATURES_PATH)

    def predict_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        out = batch.copy()
        x = out[self.features].copy()
        out["pred_label"] = self.model.predict(x)
        return out