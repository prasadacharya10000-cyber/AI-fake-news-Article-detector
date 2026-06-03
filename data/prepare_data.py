"""
data/prepare_data.py
Preprocessing for the Kaggle Fake & Real News Dataset.

KEY FIXES vs original:
- Dateline stripping is now more conservative — only strips known wire-service
  headers, not entire leading sentences, so real article content is preserved.
- clean_text now keeps punctuation for the transformer tokenisers (they need it).
  A separate clean_text_lstm version strips to letters-only for the LSTM vocab.
- HTML entities and unicode quotes are normalised before cleaning.
"""

import os
import re
import unicodedata

import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)


# ─────────────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────────────

# Wire-service dateline pattern:  "WASHINGTON (Reuters) -"
_DATELINE = re.compile(
    r"^[A-Z][A-Z ,\-]{0,40}\([A-Za-z]+\)\s*[\-–—]\s*", re.MULTILINE
)
_WIRE     = re.compile(r"\(Reuters\)|\(AP\)|\(AFP\)|\(UPI\)", re.IGNORECASE)
_URL      = re.compile(r"https?://\S+|www\.\S+")
_TWITTER  = re.compile(r"@\w+|#\w+")
_NEWLINE  = re.compile(r"\s+")


def _normalise_unicode(text: str) -> str:
    """Replace smart quotes, em-dashes, etc. with ASCII equivalents."""
    text = unicodedata.normalize("NFKD", text)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def clean_text(text: str) -> str:
    """
    Transformer-friendly clean:
    - strips datelines & wire tags
    - removes URLs, social handles
    - normalises whitespace
    - keeps punctuation and casing (transformers handle these internally)
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _normalise_unicode(text)
    text = _DATELINE.sub("", text)
    text = _WIRE.sub("", text)
    text = _URL.sub(" ", text)
    text = _TWITTER.sub(" ", text)
    text = _NEWLINE.sub(" ", text).strip()
    return text


def clean_text_lstm(text: str) -> str:
    """
    LSTM-friendly clean:
    All of the above PLUS lowercase + strip non-alpha.
    The LSTM vocab is letter-only.
    """
    text = clean_text(text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────
# Load Real Dataset (Kaggle)
# ─────────────────────────────────────────────────────────────────────

def load_kaggle_dataset(fake_path: str, true_path: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        combined_transformer  — for BERT / RoBERTa
        combined_lstm         — for BiLSTM
        label                 — 0=REAL, 1=FAKE
    """
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1
    true_df["label"] = 0

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df[["title", "text", "label"]].dropna()

    raw = df["title"].fillna("") + " " + df["text"].fillna("")
    df["combined_transformer"] = raw.apply(clean_text)
    df["combined_lstm"]        = raw.apply(clean_text_lstm)

    # Drop empty rows
    df = df[df["combined_transformer"].str.len() > 20].reset_index(drop=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# Synthetic Dataset (quick test without Kaggle data)
# ─────────────────────────────────────────────────────────────────────

FAKE_TEMPLATES = [
    "Scientists SHOCKED by secret government plan to control weather using chemtrails!!",
    "BREAKING: Celebrity admits to being part of secret society in leaked video",
    "Doctors HATE this one weird trick that cures all diseases overnight",
    "You won't believe what they found inside the moon — NASA covering up the truth",
    "EXPOSED: The real reason vaccines cause autism according to whistleblower",
    "Miracle cure banned by big pharma — they don't want you to know",
    "Politician secretly working for foreign government, anonymous source reveals",
    "Alien spacecraft found on Mars — government threatens scientists to stay silent",
    "The economy is about to collapse and bankers are secretly buying gold",
    "New world order plan to reduce population revealed in leaked documents",
    "URGENT: Deep state plot to rig election uncovered by independent journalist",
    "Hollywood elite caught on tape discussing secret satanic rituals",
]

REAL_TEMPLATES = [
    "Federal Reserve raises interest rates by 25 basis points amid inflation concerns",
    "Scientists publish new study linking diet and cardiovascular health in NEJM",
    "Parliament passes new climate bill requiring 50 percent emissions cut by 2030",
    "Tech company reports quarterly earnings above analyst expectations",
    "City council approves infrastructure spending plan after months of debate",
    "University researchers develop new method for water purification in rural areas",
    "Central bank governor testifies before senate committee on monetary policy",
    "WHO releases updated guidelines on antibiotic resistance prevention",
    "Trade agreement signed between two nations after three years of negotiations",
    "Report finds unemployment rate at lowest level in two decades",
    "International court issues ruling on territorial dispute between two countries",
    "Public health officials confirm outbreak contained after coordinated response",
]


def generate_synthetic_dataset(n_samples: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    records = []
    for _ in range(n_samples // 2):
        template = np.random.choice(FAKE_TEMPLATES)
        noise = " ".join(np.random.choice(
            ["shocking", "secret", "exposed", "banned", "hidden", "truth", "revealed"],
            size=np.random.randint(3, 8)
        ))
        raw = template + " " + noise
        records.append({
            "combined_transformer": clean_text(raw),
            "combined_lstm":        clean_text_lstm(raw),
            "label": 1,
        })
    for _ in range(n_samples // 2):
        template = np.random.choice(REAL_TEMPLATES)
        noise = " ".join(np.random.choice(
            ["according to", "officials said", "report shows", "data indicates", "study finds"],
            size=np.random.randint(1, 4)
        ))
        raw = template + " " + noise
        records.append({
            "combined_transformer": clean_text(raw),
            "combined_lstm":        clean_text_lstm(raw),
            "label": 0,
        })
    return pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# Train / Val / Test Splits
# ─────────────────────────────────────────────────────────────────────

def get_splits(df: pd.DataFrame, text_col: str = "combined_lstm"):
    X = df[text_col].tolist()
    y = df["label"].tolist()
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    DATA_DIR  = os.path.dirname(__file__)
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")
    if os.path.exists(fake_path) and os.path.exists(true_path):
        df = load_kaggle_dataset(fake_path, true_path)
    else:
        df = generate_synthetic_dataset(2000)
    print(f"Rows: {len(df)}\n{df['label'].value_counts()}")
