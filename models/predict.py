import argparse
from pathlib import Path
import joblib
import pandas as pd


def parse_features(feature_list):
    features = {}
    for item in feature_list:
        if '=' not in item:
            raise ValueError(f"Invalid feature format: {item}. Expected key=value.")
        key, value = item.split('=', 1)
        try:
            value = float(value)
            if value.is_integer():
                value = int(value)
        except ValueError:
            pass
        features[key] = value
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Predict UFC fight outcomes using a trained Random Forest model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the saved model (.joblib file).",
    )
    parser.add_argument(
        "--feature",
        action="append",
        required=True,
        help="Feature in the form key=value. Can be provided multiple times.",
    )
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    features = parse_features(args.feature)
    df = pd.DataFrame([features])
    prediction = pipeline.predict(df)[0]
    print(prediction)


if __name__ == "__main__":
    main()
