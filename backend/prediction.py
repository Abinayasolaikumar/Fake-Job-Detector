import os
import sys
from typing import Dict, Tuple
from pathlib import Path

import gdown
import torch

# Ensure project root (which contains the `model` package) is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.model_architecture import HybridBertCnnLstmAttention
from model.bert_embeddings import BertEmbeddingGenerator
from model.preprocess import clean_text
from explainability import extract_suspicious_patterns


MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "fake_recruitment_detector.pth")
DEFAULT_MODEL_URL = "https://drive.google.com/uc?id=1cE0BzOZsffaK5kvCCK8Tv_qi-ZN-MrVZ"


def ensure_model_file() -> bool:
    if os.path.exists(MODEL_PATH):
        return True

    model_url = os.getenv("MODEL_URL", DEFAULT_MODEL_URL).strip()
    if not model_url:
        return False

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        gdown.download(model_url, MODEL_PATH, quiet=True, fuzzy=True)
    except Exception:
        return False

    return os.path.exists(MODEL_PATH)


class PredictionEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep inference deterministic: if the trained weights are missing,
        # do not run with random weights.
        if not ensure_model_file():
            self.ready = False
            self.bert_model_name = "bert-base-uncased"
            self.bert_gen = None
            self.model = None
            return

        self.ready = True

        # Optional phishing-tuned backbone (if present). If not, use standard BERT.
        bertphish_path = Path(PROJECT_ROOT) / "model" / "bertphish-transformers-default-v1"
        self.bert_model_name = str(bertphish_path) if bertphish_path.exists() else "bert-base-uncased"

        self.bert_gen = BertEmbeddingGenerator(model_name=self.bert_model_name)
        self.model = HybridBertCnnLstmAttention(bert_model_name=self.bert_model_name)

        state = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, email_text: str) -> Tuple[float, float]:
        if not getattr(self, "ready", False):
            return heuristic_probabilities(email_text)

        cleaned = clean_text(email_text)
        encodings = self.bert_gen.encode_batch([cleaned])
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = outputs["probs"].cpu().numpy()[0]

        # label 0 = Genuine, label 1 = Fake
        genuine_prob = float(probs[0])
        fraud_prob = float(probs[1])
        return fraud_prob, genuine_prob


_ENGINE = None


def _get_engine() -> PredictionEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PredictionEngine()
    return _ENGINE


def risk_level_from_probability(fraud_probability: float) -> str:
    pct = fraud_probability * 100
    if pct < 40:
        return "Low"
    elif pct < 70:
        return "Medium"
    return "High"


def heuristic_probabilities(email_text: str) -> Tuple[float, float]:
    """
    Lightweight fallback when model weights are unavailable.
    Keeps the UI functional by using explainability rules as a proxy score.
    """
    explain = extract_suspicious_patterns(email_text=email_text, sender_email="")
    score = 0.2
    score += 0.6 * explain.get("risk_score_bonus", 0.0)
    if explain.get("high_risk"):
        score = max(score, 0.85)

    reasons_count = len(explain.get("reasons", []))
    if reasons_count > 1:
        score += min((reasons_count - 1) * 0.05, 0.15)

    fraud_prob = max(0.05, min(score, 0.95))
    genuine_prob = 1.0 - fraud_prob
    return fraud_prob, genuine_prob


def analyze_email(email_text: str, subject: str = "N/A", sender_email: str = "") -> Dict:
    engine = _get_engine()
    fraud_prob, genuine_prob = engine.predict(email_text)

    explain = extract_suspicious_patterns(email_text=email_text, sender_email=sender_email)

    adjusted_fraud_prob = min(fraud_prob + explain.get("risk_score_bonus", 0.0), 0.999)
    adjusted_genuine_prob = max(1.0 - adjusted_fraud_prob, 0.001)

    # High-risk alert override (PPT: alert notification for high-risk emails)
    if explain.get("high_risk"):
        adjusted_fraud_prob = max(adjusted_fraud_prob, 0.9)
        adjusted_genuine_prob = min(adjusted_genuine_prob, 0.1)

    risk = risk_level_from_probability(adjusted_fraud_prob)
    prediction_label = "Fake" if adjusted_fraud_prob >= 0.5 else "Genuine"

    alerts = []
    if risk == "High":
        alerts.append("High-risk recruitment fraud indicators detected. Do not send money or personal documents.")

    return {
        "subject": subject,
        "prediction": prediction_label,
        "risk_level": risk,
        "fraud_probability": round(adjusted_fraud_prob * 100, 2),
        "genuine_probability": round(adjusted_genuine_prob * 100, 2),
        "reasons": explain.get("reasons", []),
        "suspicious_keywords": explain.get("suspicious_keywords", []),
        "alert": risk == "High",
        "alerts": alerts,
    }
