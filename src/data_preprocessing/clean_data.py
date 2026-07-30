from pathlib import Path
import pandas as pd


def load_dataset():
    """Load the raw dataset from the data/raw folder."""
    data_path = (
        R"C:\Users\ASUS TOF GAMING\PycharmProjects\ViraSense\data\raw\train.csv"
                  )
    df = pd.read_csv(data_path)
    return df


def main():
    # Load dataset
    df = load_dataset()

    print("=" * 50)
    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    # Save processed dataset
    output_path = Path(
        r"C:\Users\ASUS TOF GAMING\PycharmProjects\ViraSense\data\processed\cleaned_dataset.csv"
    )

    # إنشاء الفولدر تلقائياً إذا مو موجود
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print("\nProcessed dataset saved successfully!")


if __name__== "__main__":
    main()
