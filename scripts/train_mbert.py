# -*- coding: utf-8 -*-


import os, sys, glob
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback
)
import evaluate

MODEL_NAME = "distilbert-base-multilingual-cased"
DATA_DIR   = "data/processed"
OUT_DIR    = "models/mbert-fake-news-bf16"

DATA_FILES = {
    "train":      f"{DATA_DIR}/ml_train.csv",
    "validation": f"{DATA_DIR}/ml_val.csv",
    "test":       f"{DATA_DIR}/ml_test.csv",
}

ID2LABEL = {0: "FAKE", 1: "REAL"}
LABEL2ID = {"FAKE": 0, "REAL": 1}

def latest_checkpoint(dir_path: str) -> Optional[str]:
    if not os.path.isdir(dir_path): return None
    ckpts = sorted(glob.glob(os.path.join(dir_path, "checkpoint-*")), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None

def print_env_info():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Device Name:", torch.cuda.get_device_name(0))

def main():
    print_env_info()


    raw = load_dataset("csv", data_files=DATA_FILES)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)


    def map_labels(ex):
        ex["labels"] = int(ex["label"])
        return ex
    raw = raw.map(map_labels)


    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=128)
    tokenized = raw.map(
        tokenize, batched=True,
        remove_columns=[c for c in raw["train"].column_names if c not in ["labels", "text"]],
    )


    print("Train size:", len(tokenized["train"]))
    print("Val size:", len(tokenized["validation"]))
    print("Test size:", len(tokenized["test"]))
    print("Sample labels:", tokenized["train"][:10]["labels"])


    acc = evaluate.load("accuracy")
    f1  = evaluate.load("f1")
    prec = evaluate.load("precision")
    rec  = evaluate.load("recall")
    def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "precision_REAL": prec.compute(predictions=preds, references=labels, average="binary", pos_label=1)["precision"],
            "recall_REAL":    rec.compute(predictions=preds, references=labels, average="binary", pos_label=1)["recall"],
            "f1_REAL":        f1.compute(predictions=preds, references=labels, average="binary", pos_label=1)["f1"],
            "f1_macro":       f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        problem_type="single_label_classification",
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    has_cuda = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=OUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        learning_rate=2e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,

        num_train_epochs=2,
        weight_decay=0.01,

        logging_steps=200,
        logging_first_step=True,

        bf16=has_cuda,
        fp16=False,
        optim="adamw_torch_fused" if has_cuda else "adamw_torch",
        torch_compile=False,
        dataloader_num_workers=4,
        skip_memory_metrics=True,

        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )


    print("Starting training...")
    train_result = trainer.train()
    print("Training finished.")
    print("Train metrics:", train_result.metrics)

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(tokenized["test"])
    print("Test metrics:", test_metrics)


    best_dir = os.path.join(OUT_DIR, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tok.save_pretrained(best_dir)
    print("Saved best model to:", best_dir)

if __name__ == "__main__":
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TRANSFORMERS_NO_FLAX"] = "1"
    main()
