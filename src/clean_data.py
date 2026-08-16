import pandas as pd
from pathlib import Path

# Locate dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "twitter_training.csv"

# Load dataset
df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=["id", "topic", "sentiment", "text"]
)

print("Original shape:", df.shape)

# 1. Remove rows with missing text
df = df.dropna(subset=["text"])

# 2. Keep only the three sentiment classes
df = df[df["sentiment"].isin(["Positive", "Negative", "Neutral"])]

# 3. Remove duplicate tweets
df = df.drop_duplicates(subset=["text"])

# 4. Keep only the columns we need
df = df[["text", "sentiment"]]

# Reset index
df = df.reset_index(drop=True)

print("\nCleaned shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

# Save cleaned dataset
OUTPUT_PATH = BASE_DIR / "data" / "cleaned_twitter.csv"
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved to:")
print(OUTPUT_PATH)