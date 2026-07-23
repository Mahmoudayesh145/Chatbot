import os
import re
import joblib

import contractions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.util import mark_negation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# -------------------------------------------------------
# Config
# -------------------------------------------------------
DATA_PATH  = './data/Emotion_classify_Data.csv'
OUTPUT_DIR = './emotion_model'
ARTIFACT   = os.path.join(OUTPUT_DIR, 'emotion_model_artifacts.joblib')
EMOTIONS   = ['anger', 'joy', 'fear']   # label order

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------
# NLTK downloads
# -------------------------------------------------------
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet',
            'averaged_perceptron_tagger']:
    nltk.download(pkg, quiet=True)

# -------------------------------------------------------
# Preprocessing
# -------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Clean and normalize text for emotion classification."""
    if pd.isna(text):
        text = ''

    text = str(text).lower()
    text = contractions.fix(text)            # i'm → i am
    text = re.sub(r'<.*?>', ' ', text)       # remove HTML tags
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  # remove URLs
    text = re.sub(r'\d+', ' ', text)         # remove digits

    tokens = word_tokenize(text)
    tokens = mark_negation(tokens)           # e.g. "not happy" → "happy_NEG"

    stop_words = set(stopwords.words('english'))
    # keep negation words — they carry emotional signal
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
        is_neg = token.endswith('_NEG')
        base   = token[:-4] if is_neg else token
        base   = re.sub(r'[^a-z0-9]', '', base)

        if base and base not in stop_words:
            lemmatized = lemmatizer.lemmatize(base)
            cleaned.append(lemmatized + ('_NEG' if is_neg else ''))

    return ' '.join(cleaned)

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
print('\n' + '='*60)
print('  EMOTION DETECTION — TRAINING')
print('='*60)

print(f'\n[1/6] Loading dataset ...')
df = pd.read_csv(DATA_PATH)
print(f'      Shape  : {df.shape}')
print(f'      Classes:\n{df["Emotion"].value_counts().to_string()}')

# Encode labels
label2id = {e: i for i, e in enumerate(EMOTIONS)}
id2label = {i: e for i, e in enumerate(EMOTIONS)}
df['label'] = df['Emotion'].map(label2id)

# -------------------------------------------------------
# Preprocess (with caching)
# -------------------------------------------------------
CACHE_PATH = './data/emotion_preprocessed.csv'

print(f'\n[2/6] Preprocessing text ...')
if os.path.exists(CACHE_PATH):
    print('      Cache found — loading...')
    df['processed_text'] = pd.read_csv(CACHE_PATH)['processed_text']
else:
    print('      Running preprocessing...')
    df['processed_text'] = [
        preprocess_text(t)
        for t in tqdm(df['Comment'], desc='Preprocessing', unit='row')
    ]
    df[['Comment', 'Emotion', 'processed_text']].to_csv(CACHE_PATH, index=False)
    print(f'      Cache saved -> {CACHE_PATH}')

df['processed_text'] = df['processed_text'].fillna('')
print(f'      Sample: "{df["processed_text"].iloc[0][:80]}"')

# -------------------------------------------------------
# Train / Val / Test split  (60 / 20 / 20, stratified)
# -------------------------------------------------------
print('\n[3/6] Splitting dataset ...')
X = df['processed_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)
print(f'      Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}')

# -------------------------------------------------------
# TF-IDF
# -------------------------------------------------------
print('\n[4/6] Building TF-IDF features ...')
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents='unicode'
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec   = vectorizer.transform(X_val)
X_test_vec  = vectorizer.transform(X_test)

print(f'      Vocabulary size : {len(vectorizer.vocabulary_)}')
print(f'      Feature matrix  : {X_train_vec.shape}')

# -------------------------------------------------------
# Train LinearSVC
# -------------------------------------------------------
print('\n[5/6] Training LinearSVC ...')
model = LinearSVC(C=1.0, max_iter=3000, random_state=42)
model.fit(X_train_vec, y_train)
print('      Done.')

# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
print('\n[6/6] Evaluation')
print('-' * 60)

y_train_pred = model.predict(X_train_vec)
y_val_pred   = model.predict(X_val_vec)
y_pred       = model.predict(X_test_vec)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc   = accuracy_score(y_val,   y_val_pred)
test_acc  = accuracy_score(y_test,  y_pred)

print(f'  Train accuracy : {train_acc:.4f}')
print(f'  Val   accuracy : {val_acc:.4f}')
print(f'  Test  accuracy : {test_acc:.4f}')
print()
print(classification_report(y_test, y_pred, target_names=EMOTIONS))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Emotion Detection - Confusion Matrix')
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
plt.close()
print(f'  Confusion matrix -> {cm_path}')

# -------------------------------------------------------
# Save artifacts
# -------------------------------------------------------
print(f'\n  Saving artifacts -> {ARTIFACT}')
joblib.dump(
    {
        'model':        model,
        'vectorizer':   vectorizer,
        'label2id':     label2id,
        'id2label':     id2label,
        'emotions':     EMOTIONS,
        'test_accuracy': float(test_acc),
    },
    ARTIFACT
)

print('\n' + '='*60)
print(f'  DONE — Test Accuracy: {test_acc * 100:.2f}%')
print('='*60 + '\n')
