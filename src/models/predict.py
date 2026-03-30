import joblib
import numpy as np

from src.utils.config import BASELINE_MODEL_PATH, TFIDF_VECTORIZER_PATH

def predict_baseline(text):
    model = joblib.load(BASELINE_MODEL_PATH)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

    text_tfidf = vectorizer.transform([text])
    pred = model.predict(text_tfidf)[0]

    return "Positive" if pred == 1 else "Negative"