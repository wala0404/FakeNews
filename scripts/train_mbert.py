import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # عطّل تكامل TensorFlow
os.environ["TRANSFORMERS_NO_FLAX"] = "1" # (اختياري) عطّل Flax كمان
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
import evaluate

MODEL_NAME = "distilbert-base-multilingual-cased"   # خفيف وسريع
DATA_DIR   = "data/processed"
OUT_DIR    = "models/mbert-fake-news"

data_files = {
    "train": f"{DATA_DIR}/train.csv",
    "validation": f"{DATA_DIR}/val.csv",
    "test": f"{DATA_DIR}/test.csv",
}
raw = load_dataset("csv", data_files=data_files)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(b):
    return tok(b["text"], truncation=True)

tokenized = raw.map(tokenize, batched=True, remove_columns=[c for c in raw["train"].column_names if c not in ["text","label"]])

acc = evaluate.load("accuracy")
f1  = evaluate.load("f1")
def compute_metrics(p):
    logits, labels = p
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

args = TrainingArguments(
    output_dir=OUT_DIR,
    eval_strategy="steps",  # Changed from evaluation_strategy
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tok,
    data_collator=DataCollatorWithPadding(tokenizer=tok),
    compute_metrics=compute_metrics,
)

trainer.train()
print("Test:", trainer.evaluate(tokenized["test"]))
best_dir = os.path.join(OUT_DIR, "best")
trainer.save_model(best_dir)
tok.save_pretrained(best_dir)
print("Saved to:", best_dir)
