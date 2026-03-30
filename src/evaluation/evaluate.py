import pandas as pd

def compare_models():
    results = {
        "Model": ["TF-IDF + Logistic", "LSTM", "GRU", "BERT"],
        "Accuracy": [0.84, 0.878, 0.876, 0.83],
        "F1 Score": [0.83, 0.878, 0.871, 0.833]
    }

    df = pd.DataFrame(results)

    print("\n=== MODEL COMPARISON ===\n")
    print(df)

if __name__ == "__main__":
    compare_models()