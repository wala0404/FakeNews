from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

best_dir = "models/mbert-fake-news-bf16/best"
tok = AutoTokenizer.from_pretrained(best_dir)
model = AutoModelForSequenceClassification.from_pretrained(best_dir)

clf = pipeline("text-classification", model=model, tokenizer=tok, device_map="auto")

print(clf("السرعة عامل مسبب ومضاعف للحوادث"))
print(clf("عاجل: علاج نهائي للسرطان خلال أسبوع وفق منشور فيسبوك!"))
print(clf("The quick brown fox jumps over the lazy dog."))
print(clf("Breaking: Cure for cancer found in a Facebook post!"))
print(clf(" "))


