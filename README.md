# TruthLens — AI Fake News Detector
### Beautiful Web UI · Flask Backend · LSTM · BERT · RoBERTa

---

## Quick Start (3 steps)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
# Fast (2 min) — synthetic data, LSTM only — good for testing
python train.py --model lstm

# Full (recommended) — real Kaggle data, all 3 models
python train.py --real

# Individual models on real data
python train.py --real --model lstm
python train.py --real --model bert
python train.py --real --model roberta
```
> **Note:** `data/Fake.csv` and `data/True.csv` are already included.

### 3. Start the server
```bash
python server.py
```
Your browser will open automatically at **http://localhost:5000** 🎉

---

## Project Structure
```
truthlens/
├── server.py           ← Flask backend (START HERE)
├── train.py            ← Training pipeline
├── index.html          ← Beautiful frontend (served by Flask)
├── requirements.txt
│
├── models/
│   ├── lstm_model.py   ← BiLSTM with attention
│   └── bert_model.py   ← BERT + RoBERTa fine-tuning
│
├── data/
│   ├── prepare_data.py ← Data loading & cleaning (FIXED)
│   ├── Fake.csv        ← Kaggle fake news dataset
│   └── True.csv        ← Kaggle real news dataset
│
├── utils/
│   └── evaluate.py     ← Metrics, confusion matrix, plots
│
└── saved_models/       ← Created after training
    ├── lstm_weights.pt
    ├── vocab.pkl
    ├── bert/
    └── roberta/
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the frontend |
| `/api/status` | GET | Backend health + which models are trained |
| `/api/analyze` | POST | Run inference |

### Example API call
```python
import requests
r = requests.post('http://localhost:5000/api/analyze', json={
    'text': 'Scientists SHOCKED by secret government plan...',
    'model': 'LSTM'   # or 'BERT' or 'RoBERTa'
})
print(r.json())
# {'label': 'FAKE', 'confidence': 0.92, 'fake_prob': 0.92, 'real_prob': 0.08, 'ms': 12}
```

---

## Bug Fixes (vs original)

| Issue | Fix |
|---|---|
| Dateline stripping too aggressive | Conservative regex, only strips known wire-service patterns |
| LSTM trained on punctuation-stripped text but inference didn't strip | `clean_text_lstm()` applied at both train and inference time |
| Transformer got lowercase/stripped text | `clean_text()` preserves case and punctuation for BERT/RoBERTa tokenisers |
| Both models used same `clean_text` | Separate `combined_lstm` and `combined_transformer` columns in DataFrame |

---

## Models

| Model | Accuracy | Speed | Params |
|---|---|---|---|
| BiLSTM | ~92% | ⚡ Fast | ~5M |
| BERT | ~97% | ⚙️ Medium | 110M |
| RoBERTa | ~98% | 🐢 Slower | 125M |
