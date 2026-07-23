import re
import pandas as pd
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.util import mark_negation
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import scipy.sparse as sp

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def preprocess_text(text: str) -> str:
    if pd.isna(text):
        text = ""

    text = str(text).lower()
    text = contractions.fix(text)

    # Preserve emotion-signal punctuation as special tokens
    text = re.sub(r'!+',    ' __EXCLAIM__ ', text)
    text = re.sub(r'\?+',   ' __QUESTION__ ', text)
    text = re.sub(r'\.{2,}', ' __ELLIPSIS__ ', text)

    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    tokens = word_tokenize(text)
    tokens = mark_negation(tokens)

    stop_words = set(stopwords.words('english'))
    negations = {
        "not", "no", "nor", "ain't", "aren't", "couldn't",
        "didn't", "doesn't", "hadn't", "hasn't", "haven't",
        "isn't", "mightn't", "mustn't", "needn't",
        "shan't", "shouldn't", "wasn't", "weren't",
        "won't", "wouldn't"
    }
    stop_words = stop_words - negations

    lemmatizer = WordNetLemmatizer()
    cleaned = []
    for token in tokens:
        if token.startswith('__') and token.endswith('__'):
            cleaned.append(token)
            continue
        is_neg = token.endswith('_NEG')
        base   = token[:-4] if is_neg else token
        base   = re.sub(r'[^a-z0-9]', '', base)
        if base and base not in stop_words:
            lemmatized = lemmatizer.lemmatize(base)
            cleaned.append(lemmatized + ('_NEG' if is_neg else ''))

    return ' '.join(cleaned)

def run_emotion_inference(artifact, text):
    model = artifact['model']
    word_tfidf = artifact['word_tfidf']
    char_tfidf = artifact['char_tfidf']
    id2label = artifact['id2label']

    processed = preprocess_text(text)
    X_word = word_tfidf.transform([processed])
    X_char = char_tfidf.transform([processed])
    X = sp.hstack([X_word, X_char])
    
    label_id = model.predict(X)[0]
    return id2label[label_id]

def run_sentiment_inference(artifact, text):
    model = artifact["model"]
    vectorizer = artifact["vectorizer"]

    processed = preprocess_text(text)
    X_text = vectorizer.transform([processed])
    scores = sia.polarity_scores(text)
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
