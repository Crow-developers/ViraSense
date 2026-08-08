from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dataset():
    data_path = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"
    return pd.read_csv(data_path)


def engineer_features(df):
    # Convert published_at to datetime
    df["published_at"] = pd.to_datetime(
        df["published_at"],
        errors="coerce"
    )

    # Extract date/time features
    df["publish_year"] = df["published_at"].dt.year
    df["publish_month"] = df["published_at"].dt.month
    df["publish_day"] = df["published_at"].dt.day
    df["publish_hour"] = df["published_at"].dt.hour
    df["publish_dayofweek"] = df["published_at"].dt.dayofweek

    # Remove leakage column
    if "view_count" in df.columns:
        df = df.drop(columns=["view_count"])

    # Normalize numeric features
    numeric_columns = [
        "duration",
        "subscriber_count"
    ]

    numeric_columns = [
        column for column in numeric_columns
        if column in df.columns
    ]

    if numeric_columns:
        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(
            df[numeric_columns]
        )

    return df


def save_dataset(df):
    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "featured_dataset.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    df = load_dataset()

    print("Original dataset shape:", df.shape)

    df = engineer_features(df)

    print("Feature-engineered dataset shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())

    save_dataset(df)

    print("Feature-engineered dataset saved successfully!")


if __name__ == "__main__":
    main()
