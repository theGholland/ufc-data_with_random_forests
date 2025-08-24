import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'complete_ufc_data.csv'
MODEL_DIR = Path(__file__).resolve().parent
TARGET_COLUMNS = ['betting_outcome', 'outcome', 'method', 'round']


def load_dataset(
    *,
    apply_feature_engineering: bool = False,
    random_state: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=['event_date', 'fighter1_dob', 'fighter2_dob'],
    )

    if apply_feature_engineering:
        from ufc.preprocessing.feature_engineering import derive_features

        # operate on a copy so the original data remains intact
        df = derive_features(df.copy(), random_state=random_state)
    else:
        # Compute age of each fighter at the time of the event
        df['fighter1_age'] = (df['event_date'] - df['fighter1_dob']).dt.days / 365.25
        df['fighter2_age'] = (df['event_date'] - df['fighter2_dob']).dt.days / 365.25

    # Convert event_date to ordinal for numerical processing
    df['event_date'] = df['event_date'].map(pd.Timestamp.toordinal)
    # Replace infinite values which can appear in the odds columns
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop unused or leakage columns
    df = df.drop(
        columns=[
            'fighter1_dob',
            'fighter2_dob',
            'event_name',
            'weight_class',
            'favourite',
            'underdog',
            'events_extract_ts',
            'odds_extract_ts',
            'fighter_extract_ts',
        ]
    )
    return df

def build_pipeline(
    categorical_features,
    numeric_features,
    *,
    n_estimators: int = 50,
    max_depth: int = 10,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
):
    return Pipeline(
        steps=[
            (
                'preprocess',
                ColumnTransformer(
                    transformers=[
                        (
                            'cat',
                            Pipeline(
                                steps=[
                                    ('imputer', SimpleImputer(strategy='most_frequent')),
                                    (
                                        'encoder',
                                        OrdinalEncoder(
                                            handle_unknown='use_encoded_value',
                                            unknown_value=-1,
                                        ),
                                    ),
                                ]
                            ),
                            categorical_features,
                        ),
                        (
                            'num',
                            Pipeline(
                                steps=[('imputer', SimpleImputer(strategy='mean'))]
                            ),
                            numeric_features,
                        ),
                    ]
                ),
            ),
            (
                'model',
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                ),
            ),
        ]
    )

def train_and_save(
    target: str,
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    random_state: int,
    apply_feature_engineering: bool,
) -> None:
    df = load_dataset(
        apply_feature_engineering=apply_feature_engineering,
        random_state=random_state,
    )
    df = df.dropna(subset=[target])
    feature_columns = [c for c in df.columns if c not in TARGET_COLUMNS]
    categorical_features = [
        'fighter1',
        'fighter2',
        'fighter1_stance',
        'fighter2_stance',
    ]
    numeric_features = [c for c in feature_columns if c not in categorical_features]

    X = df[feature_columns]
    y = df[target]

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    pipeline = build_pipeline(
        categorical_features,
        numeric_features,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    print(
        f"{target} train accuracy: {train_score:.3f}, test accuracy: {test_score:.3f}"
    )

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f'{target}_random_forest.joblib'
    joblib.dump(pipeline, model_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Random Forest models for multiple UFC prediction targets"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=50, help="Number of trees in the forest"
    )
    parser.add_argument(
        "--max-depth", type=int, default=10, help="Maximum depth of the trees"
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="Minimum number of samples required to split an internal node",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum number of samples required to be at a leaf node",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--feature-engineering",
        action="store_true",
        help="Apply additional feature engineering during training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for target in TARGET_COLUMNS:
        train_and_save(
            target,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            apply_feature_engineering=args.feature_engineering,
        )

if __name__ == '__main__':
    main()
