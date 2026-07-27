# 👥 Team Roles & Task Distribution - ViraSense

This document outlines the division of responsibilities and tasks among the development team members for the **ViraSense** project.

---

### 🧹 1. Data Engineering (Preprocessing Lead)
**Assigned Members:** Member 1 & Member 2

* Clean `train.xlsx`: Handle duplicates and missing values.
* Exclude `view_count` from features to prevent data leakage (keep it only for evaluation/comparison).
* Extract features from `published_date` (day of the week, month, weekend indicator).
* Download and validate thumbnail images.
* Save the final dataset as `cleaned_dataset.csv`.

---

### 🖼️ 2. Computer Vision Engineer (Image Branch)
**Assigned Members:** Member 3 & Member 4

* Build the **EfficientNetB0** branch (Transfer Learning) to extract embeddings from video thumbnails.
* Preprocess images (resizing, normalization) according to EfficientNet requirements.

---

### 📝 3. NLP & Tabular Engineer
**Assigned Members:** Member 5 & Member 6

* **Text branch:** Convert video titles into embeddings (using TF-IDF or Sentence Transformers).
* **Numeric branch:** Build an MLP on subscribers, duration, and date features.
* Conduct comparative experiments (text-only vs. tabular-only).

---

### 🚀 4. Integration & Evaluation Lead
**Assigned Member:** Member 7

* Build the **Fusion Model** to combine the three branches and add a regression layer to predict the `virality_score`.
* Build the complete training pipeline (`train.py`) and compute RMSE.
* Generate the final evaluation report (Predicted vs. Actual plots).

---