import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# DATA PATHS
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "imdb_reviews.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_reviews.csv")

# MODEL PATHS
BASELINE_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "baseline", "logreg_model.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(BASE_DIR, "saved_models", "baseline", "tfidf_vectorizer.pkl")

LSTM_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "lstm", "lstm_model.h5")
GRU_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "gru", "gru_model.h5")

# PARAMETERS
MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128

TEST_SIZE = 0.2
RANDOM_STATE = 42