# ViraSense 
### Pre-Publish Video Virality Prediction using Multimodal Deep Learning

ViraSense is a multimodal deep learning system that predicts a video's **virality score** *before* it is published — using only information available at upload time: the **thumbnail image**, the **title**, and **channel/video metadata** (subscriber count, duration, publish schedule).

Unlike simple view-count predictors, ViraSense avoids data leakage by never using post-publish engagement metrics (views, likes) as input features. Instead, it fuses three complementary signals:

- 🖼️ **Visual branch** — EfficientNetB0 (transfer learning) extracts features from the video thumbnail
- 📝 **Text branch** — Embedding-based model extracts semantic signals from the video title
- 🔢 **Tabular branch** — MLP processes numeric/categorical metadata (subscribers, duration, publish time)

These branches are combined in a **fusion layer** that outputs the predicted virality score (regression task, evaluated with RMSE).

---

## 🎯 Project Goal
Help content creators and marketing teams **score draft content before publishing**, so they can prioritize the highest-potential videos and iterate on weaker ones (title/thumbnail) before going live.

---

## 🗂️ Project Structure

```
ViraSense/
├── data/
│   ├── raw/                  # Original dataset (train.xlsx)
│   ├── processed/            # Cleaned & feature-engineered dataset
│   └── thumbnails/           # Downloaded thumbnail images
├── notebooks/
│   └── EDA.ipynb             # Exploratory Data Analysis
├── src/
│   ├── data_preprocessing/
│   │   ├── clean_data.py             # Cleaning, deduplication, missing values
│   │   ├── feature_engineering.py    # Date/time features, normalization
│   │   └── download_thumbnails.py    # Fetch & validate thumbnail images
│   ├── models/
│   │   ├── cnn_image_model.py        # EfficientNetB0 image branch
│   │   ├── text_model.py             # Title embedding branch
│   │   ├── tabular_model.py          # Metadata MLP branch
│   │   └── fusion_model.py           # Multimodal fusion + regression head
│   ├── training/
│   │   ├── config.py                 # Hyperparameters & paths
│   │   └── train.py                  # Training loop
│   └── evaluation/
│       ├── evaluate.py               # Model evaluation on test set
│       └── metrics.py                # RMSE and other metrics
├── models_saved/             # Trained model checkpoints (.h5 / .pt)
├── results/
│   ├── figures/              # Training curves, prediction plots
│   └── logs/                 # Training logs
├── docs/
│   └── team_roles.md         # Team task breakdown
├── requirements.txt
└── README.md
```

---

## 🧩 Pipeline Overview

1. **Preprocessing** — clean `train.xlsx`, engineer features, exclude leakage columns (`view_count`), download & validate thumbnails, save `cleaned_dataset.csv`
2. **Feature branches** — build image (CNN), text, and tabular sub-models independently
3. **Fusion & training** — merge branch outputs, train end-to-end, track RMSE
4. **Evaluation** — validate on held-out set, generate prediction plots and error analysis

---

## 🛠️ Tech Stack
- Python 3.10+
- TensorFlow / Keras (EfficientNetB0)
- Pandas, NumPy, Scikit-learn
- Sentence embeddings (for title text) or TF-IDF baseline
- Matplotlib / Seaborn (visualization)

---

## 👥 Team
This project is developed collaboratively by a 7-person team. See [`docs/team_roles.md`](docs/team_roles.md) for the detailed task breakdown per member.

---

## 📦 Setup
```bash
git clone <repo-url>
cd ViraSense
pip install -r requirements.txt
```

---


