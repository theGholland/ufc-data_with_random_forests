"""Predict UFC fight outcomes using a trained Random Forest model.

The original script looked up a pre-computed row in the historical dataset for
the exact fight and date. This limited predictions to bouts that had already
taken place.  The updated version infers each fighter's statistics based on
their most recent fight *before* the requested event date, allowing predictions
for hypothetical or upcoming matchups.

Run with ``--changes`` to view a changelog of modifications made in this fork.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "complete_ufc_data.csv"

# Columns present in the dataset that are not model features
TARGET_COLUMNS = ["betting_outcome", "outcome", "method", "round"]


def load_dataset() -> pd.DataFrame:
    """Load the historical dataset with raw fighter statistics."""

    return pd.read_csv(
        DATA_PATH,
        parse_dates=["event_date", "fighter1_dob", "fighter2_dob"],
    )


def _recent_stats(
    df: pd.DataFrame, fighter: str, event_dt: pd.Timestamp, prefix: str
) -> dict:
    """Return the fighter's stats from their most recent fight before ``event_dt``.

    Parameters
    ----------
    df:
        Historical fights dataset with datetime columns.
    fighter:
        Name of the fighter to look up.
    event_dt:
        Timestamp of the upcoming event.
    prefix:
        Prefix to apply to the returned feature names (``"fighter1_"`` or
        ``"fighter2_"``).
    """

    mask = (
        ((df["fighter1"] == fighter) | (df["fighter2"] == fighter))
        & (df["event_date"] < event_dt)
    )
    prior_fights = df.loc[mask]
    if prior_fights.empty:
        raise ValueError(f"No historical data found for {fighter} before {event_dt.date()}")

    last_row = prior_fights.loc[prior_fights["event_date"].idxmax()]
    source_prefix = "fighter1_" if last_row["fighter1"] == fighter else "fighter2_"
    cols = [c for c in df.columns if c.startswith(source_prefix)]
    stats = {prefix + c[len(source_prefix):]: last_row[c] for c in cols}

    # Compute age at the upcoming event date
    dob_key = prefix + "dob"
    dob = stats.pop(dob_key)
    stats[prefix + "age"] = (event_dt - dob).days / 365.25
    return stats


def build_features(fighter1: str, fighter2: str, event_date: str) -> dict:
    """Return model-ready features for the given fighters and event date."""

    df = load_dataset()
    event_dt = pd.Timestamp(event_date)

    features: dict[str, object] = {
        "fighter1": fighter1,
        "fighter2": fighter2,
        "event_date": event_dt.toordinal(),
        # Odds may not be known for future fights; impute with NaN
        "favourite_odds": np.nan,
        "underdog_odds": np.nan,
    }
    features.update(_recent_stats(df, fighter1, event_dt, "fighter1_"))
    features.update(_recent_stats(df, fighter2, event_dt, "fighter2_"))

    # Remove any columns that are not features (e.g., target labels)
    for col in TARGET_COLUMNS:
        features.pop(col, None)
    return features

def main() -> None:
    changes_only = "--changes" in sys.argv
    parser = argparse.ArgumentParser(
        description="Predict UFC fight outcomes using a trained Random Forest model.",
        epilog="Use --changes to see fork-specific modifications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=not changes_only,
        help="Path to the saved model (.joblib file).",
    )
    parser.add_argument("--fighter1", required=not changes_only, help="Name of the first fighter.")
    parser.add_argument("--fighter2", required=not changes_only, help="Name of the second fighter.")
    parser.add_argument(
        "--event-date",
        required=not changes_only,
        help="Fight date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--changes",
        action="store_true",
        help="Show changelog for this fork and exit.",
    )
    args = parser.parse_args()

    if args.changes:
        changelog_path = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
        if changelog_path.exists():
            print(changelog_path.read_text())
        else:
            print("No changelog available.")
        return

    pipeline = joblib.load(args.model)
    features = build_features(args.fighter1, args.fighter2, args.event_date)
    df = pd.DataFrame([features])
    prediction = pipeline.predict(df)[0]
    print(prediction)


if __name__ == "__main__":
    main()

