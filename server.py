"""
server.py — TruthLens Flask Backend
─────────────────────────────────────
Serves the beautiful frontend at http://localhost:5000
and exposes AI inference via REST API.

Run:   python server.py
Train: python train.py --real        (needs data/Fake.csv + data/True.csv)
       python train.py               (synthetic data, fast)
"""

import os, sys, pickle, threading, webbrowser, time, json, logging
from datetime import datetime

import torch
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

_cache = {}  # model cache


# ─── helpers ─────────────────────────────────────────────────────────

def _load_lstm():
    if "lstm" in _cache:
        return _cache["lstm"]
    try:
        from models.lstm_model import BiLSTMClassifier
        vp = os.path.join(SAVE_DIR, "vocab.pkl")
        wp = os.path.join(SAVE_DIR, "lstm_weights.pt")
        if not os.path.exists(vp) or not os.path.exists(wp):
            return None, None
        with open(vp, "rb") as f:
            vocab = pickle.load(f)
        model = BiLSTMClassifier(vocab_size=len(vocab))
        model.load_state_dict(torch.load(wp, map_location=DEVICE))
        model.to(DEVICE).eval()
        _cache["lstm"] = (model, vocab)
        log.info("LSTM loaded ✓")
        return model, vocab
    except Exception as e:
        log.warning(f"LSTM load failed: {e}")
        return None, None


def _load_transformer(key: str):
    if key in _cache:
        return _cache[key]
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        path = os.path.join(SAVE_DIR, key)
        if not os.path.exists(path):
            return None, None
        tok   = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(DEVICE).eval()
        _cache[key] = (model, tok)
        log.info(f"{key.upper()} loaded ✓")
        return model, tok
    except Exception as e:
        log.warning(f"{key} load failed: {e}")
        return None, None


def _model_ready(key: str) -> bool:
    k = key.lower()
    if k == "lstm":
        return (os.path.exists(os.path.join(SAVE_DIR, "vocab.pkl")) and
                os.path.exists(os.path.join(SAVE_DIR, "lstm_weights.pt")))
    return os.path.exists(os.path.join(SAVE_DIR, k))


# ─── routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "online":  True,
        "device":  DEVICE,
        "lstm":    _model_ready("lstm"),
        "bert":    _model_ready("bert"),
        "roberta": _model_ready("roberta"),
        "time":    datetime.now().isoformat(),
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    mdl  = (data.get("model") or "LSTM").upper()

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) < 10:
        return jsonify({"error": "Text too short — please paste a full headline or article"}), 400

    t0 = time.time()

    # Apply appropriate cleaning at inference time
    from data.prepare_data import clean_text, clean_text_lstm

    try:
        if mdl == "LSTM":
            m, vocab = _load_lstm()
            if m is None:
                return jsonify({"error": "LSTM not trained yet. Run: python train.py --model lstm"}), 503
            from models.lstm_model import predict_lstm
            cleaned = clean_text_lstm(text)
            result = predict_lstm(cleaned, m, vocab, device=DEVICE)

        elif mdl == "BERT":
            m, tok = _load_transformer("bert")
            if m is None:
                return jsonify({"error": "BERT not trained yet. Run: python train.py --model bert"}), 503
            from models.bert_model import predict_transformer
            cleaned = clean_text(text)
            result = predict_transformer(cleaned, m, tok, device=DEVICE)

        elif mdl == "ROBERTA":
            m, tok = _load_transformer("roberta")
            if m is None:
                return jsonify({"error": "RoBERTa not trained yet. Run: python train.py --model roberta"}), 503
            from models.bert_model import predict_transformer
            cleaned = clean_text(text)
            result = predict_transformer(cleaned, m, tok, device=DEVICE)

        else:
            return jsonify({"error": f"Unknown model '{mdl}'"}), 400

        result["ms"] = round((time.time() - t0) * 1000)
        return jsonify(result)

    except Exception as e:
        log.error(f"Inference error: {e}", exc_info=True)
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


# ─── startup ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    url  = f"http://localhost:{port}"

    print(f"""
╔══════════════════════════════════════════╗
║   🔍  TruthLens  AI Fake News Detector   ║
╠══════════════════════════════════════════╣
║  URL    : {url:<31}║
║  Device : {DEVICE:<31}║
╠══════════════════════════════════════════╣
║  Train  : python train.py --real         ║
║  Fast   : python train.py                ║
╚══════════════════════════════════════════╝
""")

    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
