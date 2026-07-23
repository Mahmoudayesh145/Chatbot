import re
import joblib
import contractions
import pandas as pd
from scipy.sparse import hstack

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.util import mark_negation

# -------------------------------------------------------
# Load saved artifacts
# -------------------------------------------------------
ARTIFACT_PATH = './emotion_model/emotion_model_artifacts.joblib'

artifacts  = joblib.load(ARTIFACT_PATH)
model      = artifacts['model']
word_tfidf = artifacts['word_tfidf']
char_tfidf = artifacts['char_tfidf']
id2label   = artifacts['id2label']
label2id   = artifacts['label2id']

# -------------------------------------------------------
# Preprocessing  (identical to model.py)
# -------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Identical to model.py preprocessing — must stay in sync."""
    if pd.isna(text):
        text = ''

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
    tokens = mark_negation(tokens)  # "not happy" → "happy_NEG"

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
            cleaned.append(token)   # pass special tokens through unchanged
            continue
        is_neg = token.endswith('_NEG')
        base   = token[:-4] if is_neg else token
        base   = re.sub(r'[^a-z0-9]', '', base)
        if base and base not in stop_words:
            lemmatized = lemmatizer.lemmatize(base)
            cleaned.append(lemmatized + ('_NEG' if is_neg else ''))

    return ' '.join(cleaned)

# -------------------------------------------------------
# Prediction function
# -------------------------------------------------------

def predict_emotion(text: str) -> dict:
    """Predict emotion from raw text.

    Returns:
        dict with 'emotion' (one of 15 coarse groups:
        admiration, anger, curiosity, disapproval, disgust, fear,
        gratitude, hope, joy, love, neutral, pride, sadness, shame, surprise)
        and 'label_id'.
    """
    processed = preprocess_text(text)
    X_word = word_tfidf.transform([processed])
    X_char = char_tfidf.transform([processed])   # char n-grams on processed text
    X = hstack([X_word, X_char])
    label_id = model.predict(X)[0]
    return {'emotion': id2label[label_id], 'label_id': int(label_id)}


# -------------------------------------------------------
# Quick demo
# -------------------------------------------------------
if __name__ == '__main__':
    print(f'\nModel loaded from: {ARTIFACT_PATH}')
    print(f'Test accuracy : {artifacts.get("test_accuracy", "N/A"):.4f}')
    print(f'Emotions      : {artifacts["emotions"]}\n')

    test_cases = [
        "I am so furious right now, this is absolutely unacceptable!",
        "Today was the best day of my life, I am so happy and grateful!",
        "I am terrified, I don't know what is going to happen next.",
        "I can't believe they did that, I'm really angry.",
        "This made me feel so joyful and excited, I love it!",
        "I'm scared and anxious about the results.",
        "I feel like crying, everything is falling apart.",
    ]

    print(f'{"Text":<50} | Emotion')
    print('-' * 60)
    for text in test_cases:
        result = predict_emotion(text)
        short  = text[:48] + '..' if len(text) > 50 else text
        print(f'{short:<50} | {result["emotion"]}')
    print('-' * 60)
