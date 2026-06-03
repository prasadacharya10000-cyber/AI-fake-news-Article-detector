"""
models/lstm_model.py
Bidirectional LSTM model for fake news classification.
Uses GloVe-style random embeddings (or pretrained if available).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import re


# ─────────────────────────────────────────────
# Vocabulary Builder
# ─────────────────────────────────────────────

class Vocabulary:
    def __init__(self, max_vocab: int = 20000):
        self.max_vocab = max_vocab
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        for word, _ in counter.most_common(self.max_vocab - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self

    def encode(self, text: str, max_len: int = 256):
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        # Pad
        ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────

class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocabulary, max_len: int = 256):
        self.encodings = [vocab.encode(t, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ─────────────────────────────────────────────
# Bidirectional LSTM
# ─────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))           # (B, L, E)
        lstm_out, _ = self.lstm(embedded)                    # (B, L, H*2)

        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, L, 1)
        context = (attn_weights * lstm_out).sum(dim=1)       # (B, H*2)

        logits = self.classifier(context)
        return logits


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_lstm(
    X_train, y_train,
    X_val, y_val,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_len: int = 256,
    device: str = None,
    class_weights: list = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LSTM] Training on {device}")

    # Build vocabulary
    vocab = Vocabulary(max_vocab=20000).build(X_train)

    # Datasets & loaders
    train_ds = NewsDataset(X_train, y_train, vocab, max_len)
    val_ds   = NewsDataset(X_val,   y_val,   vocab, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # Model
    model = BiLSTMClassifier(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights = torch.tensor(class_weights if class_weights else [1.0, 1.0]).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item()
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)

        acc = correct / total
        history["train_loss"].append(total_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(acc)
        print(f"  Epoch {epoch}/{epochs} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Loss: {history['val_loss'][-1]:.4f} | "
              f"Val Acc: {acc:.4f}")

    return model, vocab, history


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def predict_lstm(text: str, model: BiLSTMClassifier, vocab: Vocabulary,
                 max_len: int = 256, device: str = "cpu"):
    model.eval()
    ids = torch.tensor([vocab.encode(text, max_len)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(ids)
        probs = torch.softmax(logits, dim=1)[0]
    label = int(probs.argmax())
    return {
        "label": "FAKE" if label == 1 else "REAL",
        "confidence": float(probs[label]),
        "fake_prob": float(probs[1]),
        "real_prob": float(probs[0]),
    }
