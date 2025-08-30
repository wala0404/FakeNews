# server.py
import os, torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import pytesseract

import numpy as np, joblib

USE_META = int(os.getenv("USE_META", "0"))
META_PIPE, FEAT_ORDER = None, []

if USE_META:
    try:
        _meta = joblib.load(os.getenv("META_PATH", "models/hybrid_meta/meta.joblib"))
        META_PIPE = _meta["pipe"]
        FEAT_ORDER = _meta["feature_order"]
        print(f"[meta] loaded with {len(FEAT_ORDER)} features")
        from features_hybrid import extract_features
    except Exception as e:
        print("[meta] not loaded:", e)
        USE_META = 0











MODEL_DIR = os.getenv("MODEL_DIR", "models/mbert-fake-news-bf16/best")
THRESHOLD_REAL = float(os.getenv("THRESHOLD_REAL", "0.57"))
UNSURE_BAND = float(os.getenv("UNSURE_BAND", "0.10"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Fake News API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

_tok = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()

class ClassifyIn(BaseModel):
    text: str

class ClassifyOut(BaseModel):
    label: str
    score: float
    probs: dict
    threshold_real: float
    unsure_band: float
    model_dir: str

@app.get("/api")
def info():
    return {"ok": True, "device": DEVICE, "model_dir": MODEL_DIR,
            "threshold_real": THRESHOLD_REAL, "unsure_band": UNSURE_BAND}



@torch.inference_mode()
@app.post("/api/classify", response_model=ClassifyOut)
def classify(payload: ClassifyIn):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")


    enc = _tok(payload.text, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    logits = _model(**enc).logits[0].float()
    probs_bert = torch.softmax(logits, dim=-1).cpu().tolist()   # [FAKE, REAL]
    p_fake_bert, p_real_bert = float(probs_bert[0]), float(probs_bert[1])


    p_real_final = p_real_bert


    if USE_META and META_PIPE is not None:
        try:

            f = extract_features(payload.text, _tok)
            vec = [float(f.get(k, 0.0)) for k in FEAT_ORDER if k != "bert_prob_real"]
            vec.append(p_real_bert)
            x = np.array([vec], dtype=np.float32)

            p_real_meta = float(META_PIPE.predict_proba(x)[0][1])


            if abs(p_real_bert - 0.5) <= 0.20:
                p_real_final = 0.7 * p_real_meta + 0.3 * p_real_bert

        except Exception as e:

            print("[meta] failed during inference:", e)
            p_real_final = p_real_bert

    p_fake_final = 1.0 - p_real_final


    if abs(p_real_final - 0.5) < UNSURE_BAND:
        label, score = "UNSURE", max(p_fake_final, p_real_final)
    else:
        label = "REAL" if p_real_final >= THRESHOLD_REAL else "FAKE"
        score = p_real_final if label == "REAL" else p_fake_final

    return ClassifyOut(
        label=label,
        score=score,
        probs={"FAKE": p_fake_final, "REAL": p_real_final},
        threshold_real=THRESHOLD_REAL,
        unsure_band=UNSURE_BAND,
        model_dir=MODEL_DIR
    )


@app.post("/api/ocr")
def ocr(file: UploadFile = File(...), lang: str = Form("ara")):
    img = Image.open(file.file)
    text = pytesseract.image_to_string(img, lang=lang)
    return {"text": text}
