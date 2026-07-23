import re
import joblib
import contractions
import numpy as np
import pandas as pd
import scipy.sparse as sp

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.util import mark_negation

# ---------------------------------
# Load saved model
# ---------------------------------

artifacts = joblib.load("./results/sentiment_model_artifacts.joblib")

model = artifacts["model"]
vectorizer = artifacts["vectorizer"]

sia = SentimentIntensityAnalyzer()

# ---------------------------------
# Same preprocessing function
# ---------------------------------

def preprocess_text(text):

    if pd.isna(text):
        text = ""

    text = str(text).lower()
    text = contractions.fix(text)

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)

    tokens = word_tokenize(text)

    tokens = mark_negation(tokens)

    stop_words = set(stopwords.words("english"))

    negations = {
        "not","no","nor","ain't","aren't","couldn't",
        "didn't","doesn't","hadn't","hasn't","haven't",
        "isn't","mightn't","mustn't","needn't",
        "shan't","shouldn't","wasn't","weren't",
        "won't","wouldn't"
    }

    stop_words = stop_words - negations

    lemmatizer = WordNetLemmatizer()

    cleaned = []

    for token in tokens:

        is_neg = token.endswith("_NEG")

        base = token[:-4] if is_neg else token

        base = re.sub(r"[^a-z0-9]", "", base)

        if base and base not in stop_words:

            base = lemmatizer.lemmatize(base)

            if is_neg:
                cleaned.append(base + "_NEG")
            else:
                cleaned.append(base)

    return " ".join(cleaned)

# ---------------------------------
# Prediction function
# ---------------------------------

def predict_sentiment(review):

    processed = preprocess_text(review)

    X_text = vectorizer.transform([processed])

    scores = sia.polarity_scores(review)

    X_vader = np.array([
        [
            scores["neg"] * 10,
            scores["neu"] * 10,
            scores["pos"] * 10,
            scores["compound"] * 10
        ]
    ])

    X = sp.hstack([X_text, X_vader])

    prediction = model.predict(X)[0]

    labels = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    return labels[prediction]

# ---------------------------------
# Test
# ---------------------------------

review = "It doesn't work."

print("Review:", review)
print("Prediction:", predict_sentiment(review))