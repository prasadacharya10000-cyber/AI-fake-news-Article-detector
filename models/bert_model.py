"""
models/bert_model.py
Fine-tuning BERT and RoBERTa for fake news binary classification.
Both models share the same trainer — just swap the model name.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class TransformerNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = 256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ─────────────────────────────────────────────
# Generic Transformer Trainer
# ─────────────────────────────────────────────

def train_transformer(
    X_train, y_train,
    X_val, y_val,
    model_name: str = "bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_len: int = 256,
    device: str = None,
):
    """
    Fine-tune a HuggingFace sequence classification model.
    Works for BERT, RoBERTa, DistilBERT, etc.

    Args:
        model_name: HuggingFace model hub name.
                    'bert-base-uncased'  → BERT
                    'roberta-base'       → RoBERTa
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    short_name = model_name.split("/")[-1].upper()
    print(f"[{short_name}] Loading model & tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    train_ds = TransformerNewsDataset(X_train, y_train, tokenizer, max_len)
    val_ds   = TransformerNewsDataset(X_val,   y_val,   tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()

        # ── Validate ──
        model.eval()
        val_loss, all_preds, all_labels = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average="weighted")
        history["train_loss"].append(total_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)
        print(f"  Epoch {epoch}/{epochs} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Loss: {history['val_loss'][-1]:.4f} | "
              f"Val Acc: {acc:.4f} | F1: {f1:.4f}")

    return model, tokenizer, history


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def predict_transformer(
    text: str,
    model,
    tokenizer,
    max_len: int = 256,
    device: str = "cpu",
):
    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = torch.softmax(logits, dim=1)[0]

    label = int(probs.argmax())
    return {
        "label":      "FAKE" if label == 1 else "REAL",
        "confidence": float(probs[label]),
        "fake_prob":  float(probs[1]),
        "real_prob":  float(probs[0]),
    }
