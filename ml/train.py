import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
from app.ml.features import preprocess_text

def train():
    train_df = pd.read_csv("data/processed/train.csv")
    X = train_df["text"].apply(preprocess_text)
    y = train_df["label"].map({"REAL": 1, "FAKE": 0, "Real": 1, "Fake": 0})
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    joblib.dump(vectorizer, "models/vectorizer.joblib")

if __name__ == "__main__":
    train()
from fastapi import APIRouter, HTTPException
from app.schemas import NewsClassifyRequest, NewsClassifyResponse
from app.ml.infer import classify_news

router = APIRouter()

@router.post("/classify", response_model=NewsClassifyResponse)
def classify(request: NewsClassifyRequest):
    try:
        label, score = classify_news(request.text)
        return NewsClassifyResponse(label=label, score=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

