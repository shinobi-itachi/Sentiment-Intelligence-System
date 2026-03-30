import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

from src.utils.config import (
    PROCESSED_DATA_PATH,
    GRU_MODEL_PATH,
    MAX_WORDS,
    MAX_LEN,
    EMBEDDING_DIM,
    TEST_SIZE,
    RANDOM_STATE
)

def train_gru():
    print("Loading processed data...")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df["clean_text"]
    y = df["label"]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    print("Padding sequences...")
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

    print("Building GRU model...")

    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM))
    model.add(GRU(64))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    print(model.summary())

    print("Training model...")
    model.fit(
        X_train_pad,
        y_train,
        epochs=3,
        batch_size=64,
        validation_split=0.2
    )

    print("Evaluating model...")

    y_pred_prob = model.predict(X_test_pad)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    os.makedirs(os.path.dirname(GRU_MODEL_PATH), exist_ok=True)
    model.save(GRU_MODEL_PATH)

    print("GRU model saved successfully.")

if __name__ == "__main__":
    train_gru()