from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.load_clean import load_and_clean_folder


RF_FEATURES = [
    "Protocol",
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "SYN Flag Count",
    "ACK Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "Average Packet Size",
]

TARGET = "binary_label"


def main() -> None:
    df = load_and_clean_folder("data")

    print("Total rows:", len(df))
    print("Class distribution:")
    print(df[TARGET].value_counts())

    X = df[RF_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n=== RF Baseline ===")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"f1:        {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "rf_model.joblib"
    meta_path = artifact_dir / "rf_features.joblib"

    joblib.dump(model, model_path)
    joblib.dump(RF_FEATURES, meta_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved feature list to: {meta_path}")


if __name__ == "__main__":
    main()