"""
Features for modelling which require additional calculation steps

TODO
- Add fighter record prior to fight as feature

"""

import pandas as pd
import numpy as np

from ufc import constants


def randomize_fighter_order(df: pd.DataFrame, random_state: int | None = None) -> pd.DataFrame:
    """Randomly switch fighter1 and fighter2 related columns.

    This helps prevent models from learning that ``fighter1`` tends to be the
    winner simply because of ordering in the dataset. For a random half of the
    rows the fighters, their statistics and odds information are swapped and the
    outcome columns updated accordingly.
    """

    df = df.copy()
    rng = np.random.default_rng(random_state)
    swap_mask = rng.random(len(df)) < 0.5

    if not swap_mask.any():
        return df

    # swap fighter names
    df.loc[swap_mask, ["fighter1", "fighter2"]] = df.loc[swap_mask, ["fighter2", "fighter1"]].values

    # swap fighter stats
    fighter1_cols = [c for c in df.columns if c.startswith("fighter1_")]
    # Deriving partner column names avoids mismatches if column order changes
    for c1 in fighter1_cols:
        c2 = c1.replace("fighter1_", "fighter2_")
        if c2 in df.columns:
            df.loc[swap_mask, [c1, c2]] = df.loc[swap_mask, [c2, c1]].values

    # swap betting odds information
    df.loc[swap_mask, ["favourite", "underdog"]] = df.loc[swap_mask, ["underdog", "favourite"]].values
    df.loc[swap_mask, ["favourite_odds", "underdog_odds"]] = df.loc[swap_mask, ["underdog_odds", "favourite_odds"]].values

    # update winner and betting outcome columns
    def _swap_outcome(x: str) -> str:
        return "fighter2" if x == "fighter1" else ("fighter1" if x == "fighter2" else x)

    def _swap_bet_outcome(x: str) -> str:
        return "underdog" if x == "favourite" else ("favourite" if x == "underdog" else x)

    df.loc[swap_mask, "outcome"] = df.loc[swap_mask, "outcome"].apply(_swap_outcome)
    df.loc[swap_mask, "betting_outcome"] = df.loc[swap_mask, "betting_outcome"].apply(_swap_bet_outcome)

    return df


def derive_features(
    df: pd.DataFrame,
    randomise_fighters: bool = True,
    random_state: int | None = None,
) -> pd.DataFrame:

    if randomise_fighters:
        df = randomize_fighter_order(df, random_state=random_state)

    # Ensure datetime columns retain their type after any swapping operations
    # ``pd.to_datetime`` can raise ``ValueError`` if a column contains a mix of
    # datetime-like objects and integers (behaviour observed on older Pandas
    # versions).  Coercing invalid or mixed values to ``NaT`` avoids this while
    # keeping valid timestamps intact.
    for col in ["event_date", "fighter1_dob", "fighter2_dob"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Compute age of each fighter at the time of the event in fractional years
    df["fighter1_age"] = (
        df["event_date"] - df["fighter1_dob"]
    ).dt.days / 365.25
    df["fighter2_age"] = (
        df["event_date"] - df["fighter2_dob"]
    ).dt.days / 365.25

    compare_attributes = [
        'height',
        "age",
        'reach',
        'sig_strikes_landed_pm',
        'sig_strikes_accuracy',
        'sig_strikes_absorbed_pm',
        'sig_strikes_defended',
        'takedown_avg_per15m',
        'takedown_accuracy',
        'takedown_defence',
        'submission_avg_attempted_per15m'
    ]

    # compute delta between fighter1, fighter2 attributes
    for attribute in compare_attributes:
        df[f"delta_{attribute}"] = df[f"fighter1_{attribute}"] - df[f"fighter2_{attribute}"]

    # compute ratio between fighter1, fighter2 attributes safely
    for attribute in compare_attributes:
        numerator = df[f"fighter1_{attribute}"].astype(float)
        denominator = df[f"fighter2_{attribute}"].astype(float)
        df[f"ratio_{attribute}"] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0,
        )

    # TODO - add win/loss for last X fights

    return df
