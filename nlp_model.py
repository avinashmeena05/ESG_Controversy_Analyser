import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import re

LABEL_MAP = {'E':'E','S':'S','G':'G','E_POS':'NONE','S_POS':'NONE','G_POS':'NONE'}

def normalize_labels(df):
    df = df.copy()
    df['label'] = df['label'].map(LABEL_MAP).fillna('NONE')
    return df

def clean_text(t):
    t = re.sub(r'\s+',' ', str(t)).strip()
    return t

def train(train_path: str, out_dir: str):
    df = pd.read_csv(train_path)
    df = normalize_labels(df)
    df['text'] = df['text'].map(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42, stratify=df['label'])
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LinearSVC())
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    joblib.dump(pipe, f"{out_dir}/esg_text_clf.joblib")
    print(f"Model saved to {out_dir}/esg_text_clf.joblib")

def predict_texts(model_path: str, texts):
    pipe = joblib.load(model_path)
    return pipe.predict(texts)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_path', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    train(args.train_path, args.out_dir)
