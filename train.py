# train.py — Streamlit Cloud will run this to create the model
from datasets import load_dataset
import pandas as pd
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Loading dataset...")
dataset = load_dataset("ag_news")
train_df = pd.DataFrame(dataset['train'])
train_df['clean_text'] = train_df['text'].apply(clean_text)

X_train = train_df['clean_text']
y_train = train_df['label']

print("Training model...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words='english'
    )),
    ('clf', LogisticRegression(
        max_iter=1000, C=5,
        solver='lbfgs',
        multi_class='auto'
    ))
])
pipeline.fit(X_train, y_train)

os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/news_classifier.pkl')
print("Model saved ✅")
