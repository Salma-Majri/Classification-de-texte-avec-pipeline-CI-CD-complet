import argparse
import os
import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text

# Hyperparametres (visibles en haut pour le suivi MLflow de J4)
MAX_FEATURES = 5000
C = 1.0
NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
SEED = 42
MODEL_PATH = os.path.join("models", "model.pkl")


def main():
    parser = argparse.ArgumentParser(description="Entrainement du modele de classification")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sous-echantillon rapide pour la CI (ex: 2000)")
    args = parser.parse_args()

    print("Chargement du dataset IMDB...")
    dataset = load_dataset("imdb")
    df = pd.concat([
        pd.DataFrame(dataset["train"]),
        pd.DataFrame(dataset["test"]),
    ], ignore_index=True)

    if args.sample:
        print(f"Mode rapide : {args.sample} exemples...")
        df = df.sample(n=min(args.sample, len(df)), random_state=SEED)

    print("Pretraitement (clean_text)...")
    df["clean"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=TEST_SIZE,
        stratify=df["label"], random_state=SEED,
    )

    print("Entrainement (TF-IDF + LogisticRegression)...")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)),
        ("clf", LogisticRegression(C=C, random_state=SEED, max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
    seuil = 0.85 if args.sample is None else 0.80
    print(f"OK : seuil {seuil} atteint" if acc >= seuil else f"ALERTE : accuracy < {seuil}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modele sauvegarde dans {MODEL_PATH}")


if __name__ == "__main__":
    main()