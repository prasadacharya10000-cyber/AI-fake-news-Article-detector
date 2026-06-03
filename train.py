"""
train.py — TruthLens Training Pipeline
───────────────────────────────────────
Usage:
    python train.py                    # synthetic data, all 3 models
    python train.py --real             # Kaggle CSVs (data/Fake.csv + data/True.csv)
    python train.py --model lstm       # only LSTM
    python train.py --model bert       # only BERT
    python train.py --model roberta    # only RoBERTa
    python train.py --epochs 5 --batch 32
"""

import os, sys, argparse, pickle
import torch, numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data.prepare_data   import load_kaggle_dataset, generate_synthetic_dataset, get_splits
from models.lstm_model   import train_lstm,        predict_lstm
from models.bert_model   import train_transformer, predict_transformer
from utils.evaluate      import compute_metrics, plot_confusion_matrix, \
                                plot_training_history, model_comparison_table

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)


def preds_lstm(m, vocab, texts, device):
    return [1 if predict_lstm(t, m, vocab, device=device)["label"] == "FAKE" else 0 for t in texts]

def preds_transformer(m, tok, texts, device):
    return [1 if predict_transformer(t, m, tok, device=device)["label"] == "FAKE" else 0 for t in texts]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real",   action="store_true")
    p.add_argument("--model",  default="all", help="lstm|bert|roberta|all")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch",  type=int, default=16)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'═'*52}\n  TruthLens — Training Pipeline\n  Device : {device}\n  Models : {args.model}\n{'═'*52}\n")

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    if args.real:
        df = load_kaggle_dataset(
            os.path.join(DATA_DIR, "Fake.csv"),
            os.path.join(DATA_DIR, "True.csv"),
        )
    else:
        print("⚠  Using synthetic data. Pass --real to use Kaggle CSVs.\n")
        df = generate_synthetic_dataset(2000)

    train_models = ["lstm", "bert", "roberta"] if args.model == "all" else [args.model]
    histories, results = {}, {}

    # ── LSTM ─────────────────────────────────────────────────
    if "lstm" in train_models:
        print("─"*40 + "\nTraining LSTM…")
        Xtr, Xv, Xte, ytr, yv, yte = get_splits(df, "combined_lstm")
        print(f"  Train {len(Xtr)} | Val {len(Xv)} | Test {len(Xte)}")
        m, vocab, hist = train_lstm(
            Xtr, ytr, Xv, yv,
            epochs=args.epochs, batch_size=args.batch * 4, device=device,
            class_weights=[1.0, 0.91]
        )
        histories["LSTM"] = hist
        preds = preds_lstm(m, vocab, Xte, device)
        results["LSTM"] = compute_metrics(yte, preds)
        torch.save(m.state_dict(), os.path.join(SAVE_DIR, "lstm_weights.pt"))
        with open(os.path.join(SAVE_DIR, "vocab.pkl"), "wb") as f:
            pickle.dump(vocab, f)
        plot_confusion_matrix(yte, preds, "LSTM", save_path=os.path.join(SAVE_DIR, "lstm_cm.png"))
        print(f"  ✓ LSTM Test Accuracy: {results['LSTM']['accuracy']:.4f}")

    # ── BERT ─────────────────────────────────────────────────
    if "bert" in train_models:
        print("\n" + "─"*40 + "\nTraining BERT…")
        Xtr, Xv, Xte, ytr, yv, yte = get_splits(df, "combined_transformer")
        m, tok, hist = train_transformer(
            Xtr, ytr, Xv, yv,
            model_name="bert-base-uncased",
            epochs=args.epochs, batch_size=args.batch, device=device
        )
        histories["BERT"] = hist
        preds = preds_transformer(m, tok, Xte, device)
        results["BERT"] = compute_metrics(yte, preds)
        m.save_pretrained(os.path.join(SAVE_DIR, "bert"))
        tok.save_pretrained(os.path.join(SAVE_DIR, "bert"))
        plot_confusion_matrix(yte, preds, "BERT", save_path=os.path.join(SAVE_DIR, "bert_cm.png"))
        print(f"  ✓ BERT Test Accuracy: {results['BERT']['accuracy']:.4f}")

    # ── RoBERTa ───────────────────────────────────────────────
    if "roberta" in train_models:
        print("\n" + "─"*40 + "\nTraining RoBERTa…")
        Xtr, Xv, Xte, ytr, yv, yte = get_splits(df, "combined_transformer")
        m, tok, hist = train_transformer(
            Xtr, ytr, Xv, yv,
            model_name="roberta-base",
            epochs=args.epochs, batch_size=args.batch, device=device
        )
        histories["RoBERTa"] = hist
        preds = preds_transformer(m, tok, Xte, device)
        results["RoBERTa"] = compute_metrics(yte, preds)
        m.save_pretrained(os.path.join(SAVE_DIR, "roberta"))
        tok.save_pretrained(os.path.join(SAVE_DIR, "roberta"))
        plot_confusion_matrix(yte, preds, "RoBERTa", save_path=os.path.join(SAVE_DIR, "roberta_cm.png"))
        print(f"  ✓ RoBERTa Test Accuracy: {results['RoBERTa']['accuracy']:.4f}")

    print("\n" + "═"*52 + "\n  MODEL COMPARISON")
    model_comparison_table(results)
    if len(histories) > 1:
        plot_training_history(histories, save_path=os.path.join(SAVE_DIR, "training_history.png"))
    print("Training complete ✓\n")


if __name__ == "__main__":
    main()
