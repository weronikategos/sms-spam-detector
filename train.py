"""Trains and compares SMS spam detection models and saves the best one to a file."""
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# SMS Spam Collection (UCI), copy from a scikit-learn tutorial repository
DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATA_PATH = Path("data/sms.tsv")
MODEL_PATH = Path("model.joblib")


def load_data():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        print("Downloading data...")
        urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    df["is_spam"] = (df["label"] == "spam").astype(int)
    return df


def build_models():
    vectorizer = lambda: TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)
    return {
        "Naive Bayes": Pipeline([("tfidf", vectorizer()), ("clf", MultinomialNB())]),
        "Logistic Regression": Pipeline([("tfidf", vectorizer()),
                                         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Linear SVM": Pipeline([("tfidf", vectorizer()), ("clf", LinearSVC(class_weight="balanced"))]),
    }


def main():
    df = load_data()
    print(f"Messages: {len(df)}, spam: {df['is_spam'].sum()} ({df['is_spam'].mean():.1%})\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["is_spam"], test_size=0.2, stratify=df["is_spam"], random_state=42
    )

    # The model is chosen with cross-validation on the training set,
    # so the test set is used only for the final evaluation
    scores = {}
    for name, model in build_models().items():
        scores[name] = cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()
        print(f"{name:20s} F1 (spam, 5-fold CV) = {scores[name]:.3f}")

    best_name = max(scores, key=scores.get)
    best = build_models()[best_name].fit(X_train, y_train)
    y_pred = best.predict(X_test)
    print(f"\nBest model: {best_name}, test set F1: {f1_score(y_test, y_pred):.3f}\n")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=3))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["ham", "spam"])
    plt.title(f"Confusion matrix: {best_name}")
    plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")

    # The final model is trained on all the data
    final = build_models()[best_name].fit(df["message"], df["is_spam"])
    joblib.dump({"model": final, "name": best_name}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
