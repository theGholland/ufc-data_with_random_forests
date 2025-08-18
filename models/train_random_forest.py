import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import joblib
import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'complete_ufc_data.csv'
MODEL_DIR = Path(__file__).resolve().parent
TARGET_COLUMNS = ['betting_outcome', 'outcome', 'method', 'round']

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=['event_date', 'fighter1_dob', 'fighter2_dob'],
    )
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

def build_pipeline(categorical_features, numeric_features):
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
                                    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
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
            ('model', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
        ]
    )

def train_and_save(target: str) -> None:
    df = load_dataset()
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

    pipeline = build_pipeline(categorical_features, numeric_features)
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f'{target} accuracy: {score:.3f}')

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f'{target}_random_forest.joblib'
    joblib.dump(pipeline, model_path)

def main():
    for target in TARGET_COLUMNS:
        train_and_save(target)

if __name__ == '__main__':
    main()
