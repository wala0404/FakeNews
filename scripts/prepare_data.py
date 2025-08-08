import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare():
    raw_path = "data/raw/news.csv"  # Example file
    df = pd.read_csv(raw_path)
    df = df.dropna(subset=["text", "label"])
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)
    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

if __name__ == "__main__":
    prepare()

