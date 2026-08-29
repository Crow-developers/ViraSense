import pandas as pd
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Paths
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
THUMBNAILS_DIR = PROJECT_ROOT / "data" / "thumbnails"

# Create thumbnails folder if it doesn't exist
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

# Read dataset
df = pd.read_csv(DATA_PATH)

print(f"Found {len(df)} videos")

# Download thumbnails
for index, row in df.iterrows():
    video_id = str(row["video_id"])
    thumbnail_url = row["thumbnail_url"]

    if pd.isna(thumbnail_url):
        print(f"Skipping {video_id}: no thumbnail URL")
        continue

    output_path = THUMBNAILS_DIR / f"{video_id}.jpg"

    if output_path.exists():
        print(f"Already exists: {video_id}")
        continue

    try:
        response = requests.get(thumbnail_url, timeout=10)
        response.raise_for_status()

        with open(output_path, "wb") as file:
            file.write(response.content)

        print(f"Downloaded: {video_id}")

    except requests.RequestException as error:
        print(f"Failed: {video_id} -> {error}")

print("Thumbnail downloading completed!")