# 🧠 Sentiment Intelligence System

## 🚀 Overview
This project is an end-to-end NLP system for sentiment analysis, comparing multiple approaches from classical machine learning to deep learning and transformer models.

The system takes text input and predicts whether the sentiment is Positive or Negative.

---

## 🏗️ Architecture
- Data Preprocessing Pipeline  
- Feature Engineering (TF-IDF, Tokenization)  
- Model Training  
- Model Evaluation & Comparison  
- Flask Deployment  

---

## 📊 Models Implemented

Baseline:
- TF-IDF + Logistic Regression  

Deep Learning:
- LSTM  
- GRU  

Transformer:
- BERT  

---

## 📈 Model Performance

TF-IDF + Logistic Regression → Accuracy: 89.4%, F1: 0.89  
LSTM → Accuracy: 87.8%, F1: 0.87  
GRU → Accuracy: 87.6%, F1: 0.87  
BERT → Accuracy: 83%, F1: 0.83  

---

## 🧠 Key Insights

- Traditional ML (TF-IDF + Logistic Regression) outperformed deep learning models in this setup  
- LSTM captured sequence information effectively  
- GRU provided similar performance with fewer parameters  
- BERT underperformed due to limited training data and compute constraints  

---

## 🌐 Deployment

Flask web application for real-time sentiment prediction.

Run locally:

python -m app.app

Then open in browser:
http://127.0.0.1:5000/

---

## 📥 Dataset
Please follow the link
https://ai.stanford.edu/~amaas/data/sentiment/

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- TensorFlow / Keras  
- HuggingFace Transformers  
- Flask  

---

## 📂 Project Structure

src/
  preprocessing/
  features/
  models/
  evaluation/

app/
data/
saved_models/

---

## 🎯 Key Learnings

- Importance of baseline models  
- Trade-offs between ML, DL, and Transformers  
- End-to-end ML system design  
- Model deployment using Flask  

---

## 👨‍💻 Author

Rohit Kamble
