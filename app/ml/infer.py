import joblib
import os
from app.ml.features import preprocess_text
from tempfile import NamedTemporaryFile

def classify_news(text: str):
    model_path = os.path.join(os.path.dirname(__file__), '../../models/model.joblib')
    model = joblib.load(model_path)
    vectorizer = joblib.load(model_path.replace('model.joblib', 'vectorizer.joblib'))
    X = vectorizer.transform([preprocess_text(text)])
    proba = model.predict_proba(X)[0][1]
    label = "REAL" if proba >= 0.5 else "FAKE"
    return label, float(proba)

def ocr_image(file):
    # Lazy imports to avoid requiring OCR deps for classification-only usage
    from PIL import Image
    import pytesseract
    with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    image = Image.open(tmp_path)
    text = pytesseract.image_to_string(image)
    os.remove(tmp_path)
    return text

