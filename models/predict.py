"""Predict UFC fight outcomes using a trained Random Forest model.

This script requires only the fighter names and the event date. All other
statistics are looked up from the historical dataset so the user does not need
to supply feature values manually.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "complete_ufc_data.csv"

# Columns that are not used as features for the model
TARGET_COLUMNS = ["betting_outcome", "outcome", "method", "round"]
DROP_COLUMNS = [
    "fighter1_dob",
    "fighter2_dob",
    "event_name",
    "weight_class",
    "favourite",
    "underdog",
    "events_extract_ts",
    "odds_extract_ts",
    "fighter_extract_ts",
]


def load_dataset() -> pd.DataFrame:
    """Load and preprocess the dataset to match model training."""

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["event_date", "fighter1_dob", "fighter2_dob"],
    )
    df["fighter1_age"] = (df["event_date"] - df["fighter1_dob"]).dt.days / 365.25
    df["fighter2_age"] = (df["event_date"] - df["fighter2_dob"]).dt.days / 365.25
    df["event_date"] = df["event_date"].map(pd.Timestamp.toordinal)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.drop(columns=DROP_COLUMNS)
    return df


def build_features(fighter1: str, fighter2: str, event_date: str) -> dict:
    """Return model-ready features for the given fighters and event date."""

    df = load_dataset()
    event_date_ord = pd.Timestamp(event_date).toordinal()
    row = df[
        (df["fighter1"] == fighter1)
        & (df["fighter2"] == fighter2)
        & (df["event_date"] == event_date_ord)
    ]
    if row.empty:
        raise ValueError(
            f"No data found for {fighter1} vs {fighter2} on {event_date}."
        )
    row = row.drop(columns=TARGET_COLUMNS)
    return row.squeeze().to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict UFC fight outcomes using a trained Random Forest model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the saved model (.joblib file).",
    )
    parser.add_argument("--fighter1", required=True, help="Name of the first fighter.")
    parser.add_argument("--fighter2", required=True, help="Name of the second fighter.")
    parser.add_argument(
        "--event-date",
        required=True,
        help="Fight date in YYYY-MM-DD format",
    )
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    features = build_features(args.fighter1, args.fighter2, args.event_date)
    df = pd.DataFrame([features])
    prediction = pipeline.predict(df)[0]
    print(prediction)


if __name__ == "__main__":
    main()

