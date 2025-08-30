import os
import pandas as pd
from sklearn.model_selection import train_test_split


AR_PATH = r"C:\Users\lolo-\OneDrive\Desktop\dataset EN&AR\dataset EN&AR\FakeNewsDataAR.csv"
EN_PATH = r"C:\Users\lolo-\OneDrive\Desktop\dataset EN&AR\dataset EN&AR\FakeNewsDataEN.csv"

os.makedirs("data/processed", exist_ok=True)


TEXT_CANDIDATES  = ["text", "content", "body", "article", "news", "headline", "title"]
LABEL_CANDIDATES = ["label", "labels", "target", "class", "verdict", "is_fake", "fake", "truth"]

def read_any_csv(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def normalize_df(df: pd.DataFrame, path_hint: str) -> pd.DataFrame:

    text_col  = pick_col(df.columns, TEXT_CANDIDATES)
    label_col = pick_col(df.columns, LABEL_CANDIDATES)

    if text_col is None or label_col is None:
        print(f"\n[!] Could not find suitable columns in: {path_hint}")
        print("Available columns:", list(df.columns))
        raise SystemExit("Please tell me the names of the text and label columns, or update the candidate lists above.")

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    if df["label"].dtype == object:
        mapping = {
           "0": 0,
             "1": 1
        }
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(mapping)

    return df

def split_and_save(df: pd.DataFrame, out_prefix: str):

    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val, test   = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=42)
    train.to_csv(f"data/processed/{out_prefix}_train.csv", index=False)
    val.to_csv(  f"data/processed/{out_prefix}_val.csv",   index=False)
    test.to_csv( f"data/processed/{out_prefix}_test.csv",  index=False)
    print(f"✓ {out_prefix} -> train:{train.shape}  val:{val.shape}  test:{test.shape}")

def main():

    ar = read_any_csv(AR_PATH)
    en = read_any_csv(EN_PATH)


    print("AR cols:", list(ar.columns))
    print("EN cols:", list(en.columns))


    ar = normalize_df(ar, "AR")
    en = normalize_df(en, "EN")


    ar["lang"] = "ar"; en["lang"] = "en"


    merged = pd.concat([ar, en], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


    merged.to_csv("data/processed/multilingual_all.csv", index=False)


    split_and_save(merged, "ml")

    print(" Ready: data/processed/ml_train.csv / ml_val.csv / ml_test.csv")

if __name__ == "__main__":
    main()
