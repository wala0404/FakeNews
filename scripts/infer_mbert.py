
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, os, sys

MODEL_DIR = os.getenv("MODEL_DIR", "models/mbert-fake-news/best")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

# Move to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

id2label = {0: "Fake", 1: "Real"}

def predict(text: str):
    if not text.strip():
        return {"label": "Unknown", "score": 0.0}
    enc = tok(text, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].detach().cpu().numpy()
    pred = probs.argmax()
    return {"label": id2label[pred], "score": float(probs[pred])}

if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "عاجل: دواء سحري يشفي كل الأمراض بيوم"
    result = predict(text)
    print(result)