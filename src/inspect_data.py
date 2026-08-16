import pandas as pd
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent
DATA_PATH=BASE_DIR/ "data" / "twitter_training.csv"

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=["id","topic","sentiment", "text"]
    )

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nLabel distribution:")
print(df["topic"].value_counts().head(10))
