# TruthLens — AI Fake News Detector

A web application that detects fake news using three ML models — BiLSTM, BERT, and RoBERTa — with a Flask backend and a clean browser UI.

## Models

| Model | Accuracy | Speed |
|-------|----------|-------|
| BiLSTM | ~92% | Fast |
| BERT | ~97% | Medium |
| RoBERTa | ~98% | Slower |

## Features
- Three selectable ML models for inference
- Separate NLP preprocessing pipelines for LSTM vs transformer models
- Confidence scores and probability breakdown for each prediction
- REST API for programmatic access
- Auto-opens in browser on server start

## Tech Stack
Python · Flask · PyTorch · Transformers (HuggingFace) · scikit-learn · NLTK

## Getting Started

### Installation
```bash
git clone https://github.com/prasadacharya10000-cyber/AI-fake-news-Article-detector.git
cd AI-fake-news-Article-detector
pip install -r requirements.txt
```

### Train Models
```bash
# Fast test (synthetic data, LSTM only)
python train.py --model lstm

# Full training on real Kaggle data (recommended)
python train.py --real
```
> Note: `data/Fake.csv` and `data/True.csv` are included in the repo.

### Run
```bash
python server.py
```
Opens automatically at http://localhost:5000

## API
```python
import requests
r = requests.post('http://localhost:5000/api/analyze', json={
    'text': 'Your news article text here...',
    'model': 'LSTM'  # or 'BERT' or 'RoBERTa'
})
print(r.json())
# {'label': 'FAKE', 'confidence': 0.92, 'fake_prob': 0.92, 'real_prob': 0.08}
```
