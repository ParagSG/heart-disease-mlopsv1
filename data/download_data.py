"""
Download Heart Disease UCI Dataset from UCI ML Repository.
Run: python data/download_data.py
"""

import os
import urllib.request
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "heart.csv")


def download():
    print(f"Downloading dataset from UCI repository...")
    urllib.request.urlretrieve(URL, "heart_raw.csv")

    df = pd.read_csv("heart_raw.csv", header=None, names=COLUMNS, na_values="?")

    # Binarise target: 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)

    df.to_csv(OUTPUT_PATH, index=False)
    os.remove("heart_raw.csv")

    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")


if __name__ == "__main__":
    download()
