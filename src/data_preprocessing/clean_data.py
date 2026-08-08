from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dataset():
    data_path = PROJECT_ROOT / "data" / "raw" / "train.csv"
    return pd.read_csv(data_path)
def clean_dataset(df):
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows without essential information
    df = df.dropna(subset=["video_id", "title"])

    # Fill missing numeric values with the median
    numeric_columns = [
        "duration",
        "view_count",
        "subscriber_count",
        "virality_score"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = df[column].fillna(df[column].median())

    # Fill missing thumbnail URLs with an empty string
    if "thumbnail_url" in df.columns:
        df["thumbnail_url"] = df["thumbnail_url"].fillna("")

    return df


def save_dataset(df):
    output_path = Path("data/processed/cleaned_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    df = load_dataset()

    print("Original dataset shape:", df.shape)

    df = clean_dataset(df)

    print("Cleaned dataset shape:", df.shape)

    save_dataset(df)

    print("Cleaned dataset saved successfully!")


if __name__ == "__main__":
    main()