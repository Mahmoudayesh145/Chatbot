import os
import re
import string

import matplotlib.pyplot as plt
import nltk
import joblib
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.util import mark_negation
from sklearn.feature_extraction.text import TfidfVectorizer
import contractions
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm



pd.set_option('display.max_columns', None)
sia = SentimentIntensityAnalyzer()
MODEL_ARTIFACT_PATH = './data/sentiment_model_artifacts.joblib'



def preprocess_text(text):
	if pd.isna(text):
		text = ''

	text = str(text).lower()
	text = contractions.fix(text)
	text = re.sub(r'<.*?>', ' ', text)
	text = re.sub(r'http\S+|www\.\S+', ' ', text)
	text = re.sub(r'\d+', ' ', text)
	
	tokens = word_tokenize(text)
	tokens = mark_negation(tokens)
	
	stop_words = set(stopwords.words('english'))
	negations = {"not", "no", "nor", "ain't", "aren't", "couldn't", "didn't", "doesn't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't"}
	stop_words = stop_words - negations
	lemmatizer = WordNetLemmatizer()
	
	cleaned_tokens = []
	for token in tokens:
		is_neg = token.endswith('_NEG')
		base_token = token[:-4] if is_neg else token
		base_token = re.sub(r'[^a-z0-9]', '', base_token)
		if base_token and base_token not in stop_words:
			lemmatized = lemmatizer.lemmatize(base_token)
			cleaned_tokens.append(lemmatized + ('_NEG' if is_neg else ''))
			
	return ' '.join(cleaned_tokens)
def sentiment_label(rating):
    if rating <= 2:
        return 0      # Negative
    elif rating == 3:
        return 1      # Neutral
    else:
        return 2      # Positive
def sentiment_label_text(rating):
    if rating == 0:
        return 'Negative'
    elif rating == 1:
        return 'Neutral'
    else:
        return 'Positive'


print('Loading raw dataset...')
df = pd.read_csv('./data/amazon_reviews_with_sentiment.csv')
print(f'Raw dataset loaded with shape: {df.shape}')
df = df.dropna(subset=['reviewerName'])
print(f'After dropping missing reviewerName rows: {df.shape}')
df['reviewText'] = df['reviewText'].fillna('')
print('Filled missing reviewText values with empty strings')


if(not os.path.exists('./data/amazon_reviews_with_sentiment_preprocessed_v2.csv')):
    print('Preprocessed file not found. Starting preprocessing step...')
    df['processed_reviewText'] = [preprocess_text(text) for text in tqdm(df['reviewText'], total=len(df))]
    print('Sample preprocessed review:')
    print(df['processed_reviewText'].iloc[5])
    print()
    df.to_csv('./data/amazon_reviews_with_sentiment_preprocessed_v2.csv', index=False)
    print('Saved preprocessed dataset to disk')
else:
    print('Preprocessed file already exists. Skipping preprocessing step...')

print('Loading preprocessed dataset...')
df=pd.read_csv('./data/amazon_reviews_with_sentiment_preprocessed_v2.csv')
print(f'Preprocessed dataset loaded with shape: {df.shape}')
df['processed_reviewText'] = df['processed_reviewText'].fillna('')

sentiment_source = 'processed_reviewText'
print(f'Using text column: {sentiment_source}')
print(df['overall'].value_counts())
print()

print(df['overall'].describe())
print()


print('Creating labels from overall ratings...')
df['label'] = df['overall'].apply(sentiment_label)
print(f'Label distribution: {df["label"].value_counts().to_dict()}')


print('Extracting VADER sentiment features from raw text (x10 weight)...')
df['neg'] = df['reviewText'].apply(lambda x: sia.polarity_scores(str(x))['neg']) * 10
df['neu'] = df['reviewText'].apply(lambda x: sia.polarity_scores(str(x))['neu']) * 10
df['pos'] = df['reviewText'].apply(lambda x: sia.polarity_scores(str(x))['pos']) * 10
df['compound'] = df['reviewText'].apply(lambda x: sia.polarity_scores(str(x))['compound']) * 10

X_text = df["processed_reviewText"]
X_vader = df[['neg', 'neu', 'pos', 'compound']].values
y = df["label"]

print('Splitting dataset into train, validation, and test sets...')

X_text_train, X_text_test, X_vader_train, X_vader_test, y_train, y_test = train_test_split(
    X_text, X_vader, y, test_size=0.2, random_state=42, stratify=y
)

X_text_train, X_text_val, X_vader_train, X_vader_val, y_train, y_val = train_test_split(
    X_text_train, X_vader_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)

print(f'Train size: {len(X_text_train)}, Validation size: {len(X_text_val)}, Test size: {len(X_text_test)}')

print('Building TF-IDF features...')
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,3),
    min_df=3,
    max_df=0.90,
    sublinear_tf=True,
    strip_accents='unicode'
)
X_train_vec = vectorizer.fit_transform(X_text_train)
X_val_vec = vectorizer.transform(X_text_val)
X_test_vec = vectorizer.transform(X_text_test)

print('Combining TF-IDF with VADER features...')
X_train_vec = sp.hstack([X_train_vec, X_vader_train])
X_val_vec = sp.hstack([X_val_vec, X_vader_val])
X_test_vec = sp.hstack([X_test_vec, X_vader_test])
print(f'TFIDF vocabulary size: {len(vectorizer.vocabulary_)}')

print('Applying SMOTE to balance the training data...')
smote = SMOTE(random_state=42)
X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)
print(f'Resampled train size: {X_train_vec.shape[0]}')

print('Training LinearSVC model...')
model = LinearSVC(
    max_iter=2000,
    C=0.05,
    random_state=42
)
model.fit(X_train_vec, y_train)
print('Model training completed')

print('Generating predictions...')
y_train_pred = model.predict(X_train_vec)
y_val_pred = model.predict(X_val_vec)
y_pred = model.predict(X_test_vec)

print('Accuracy for Train Set:', accuracy_score(y_train, y_train_pred))
print('Accuracy for Validation Set:', accuracy_score(y_val, y_val_pred))
print('Accuracy for Test Set:', accuracy_score(y_test, y_pred))

print('Printing final classification report...')
print(classification_report(y_test, y_pred))

print(f'Saving trained model artifacts to {MODEL_ARTIFACT_PATH}...')
joblib.dump(
    {
        'model': model,
        'vectorizer': vectorizer,
        'use_vader_features': True,
        'feature_columns': ['neg', 'neu', 'pos', 'compound'],
    },
    MODEL_ARTIFACT_PATH
)
print('Saved model artifacts successfully')

