import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

import joblib

from src.utils.config import (
    PROCESSED_DATA_PATH,
    BASELINE_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
    TEST_SIZE,
    RANDOM_STATE
)

def train_baseline():
    print("Loading processed data...")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df["clean_text"]
    y = df["label"]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Applying TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating baseline model...")

    y_pred = model.predict(X_test_tfidf)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model + vectorizer
    os.makedirs(os.path.dirname(BASELINE_MODEL_PATH), exist_ok=True)

    joblib.dump(model, BASELINE_MODEL_PATH)
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)

    print("Baseline model saved successfully.")

if __name__ == "__main__":
    train_baseline()