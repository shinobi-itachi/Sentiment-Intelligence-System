import re
import string
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    negation_words = {"not", "no", "nor", "never"}
    filtered_words = []

    for word in words:
        if word in negation_words or word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            filtered_words.append(lemma)

    return " ".join(filtered_words)

def preprocess_dataframe(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    df = df[[text_col, label_col]].copy()
    df.dropna(inplace=True)

    df[text_col] = df[text_col].astype(str)
    df["clean_text"] = df[text_col].apply(clean_text)

    df = df[df["clean_text"].str.strip() != ""]

    label_map = {"negative": 0, "positive": 1}
    df["label"] = df[label_col].map(label_map)

    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    return df[["clean_text", "label"]]

if __name__ == "__main__":
    from src.utils.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

    print("Loading data...")

    df = pd.read_csv(RAW_DATA_PATH)

    print("Preprocessing started...")

    processed_df = preprocess_dataframe(df, text_col="review", label_col="sentiment")

    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Preprocessing completed.")
    print(processed_df.head())
    print(processed_df["label"].value_counts())