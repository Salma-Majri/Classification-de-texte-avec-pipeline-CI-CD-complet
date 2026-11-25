import argparse
import os
import joblib
import numpy as np
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text

# Hyperparametres (visibles en haut pour le suivi MLflow)
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
    if acc >= seuil:
        print(f"OK : seuil {seuil} atteint")
    else:
        print(f"ALERTE : accuracy < {seuil}")

        # === DEBUT AJOUT MLFLOW AVANCÉ ===
    import os
    os.environ["MLFLOW_TRACKING_USERNAME"] = "fatimazahramassane"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "9a31091bc76bd61738366ea2145978a3926cc93c"
    
    mlflow.set_tracking_uri(
        "https://dagshub.com/Salma-Majri/"
        "Classification-de-texte-avec-pipeline-CI-CD-complet.mlflow"
    )

    with mlflow.start_run(run_name="tfidf_logreg_baseline"):
        mlflow.log_param("max_features", MAX_FEATURES)
        mlflow.log_param("C", C)
        mlflow.log_param("ngram_range", f"{NGRAM_RANGE[0]}-{NGRAM_RANGE[1]}")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        
        # NOUVEAU : Extraire les 10 mots les plus importants (Interprétabilité)
        feature_names = model.named_steps['tfidf'].get_feature_names_out()
        coefs = model.named_steps['clf'].coef_[0]
        top_positive_words = [feature_names[i] for i in np.argsort(coefs)[-10:]]
        mlflow.log_param("top_10_positive_words", ", ".join(top_positive_words))
        
        # NOUVEAU : Déduire la signature du modèle (Input/Output) pour l'API
        sample_input = X_train[:5]
        sample_prediction = model.predict(sample_input)
        signature = infer_signature(sample_input, sample_prediction)
        
        # Enregistrement avec la signature
        mlflow.sklearn.log_model(
            model, 
            "model_sklearn",
            signature=signature, # <-- C'est ça qui aide Salma !
            skops_trusted_types=['scipy.sparse._csr.csr_matrix']
        )
        
        print("Metriques, signature et modele enregistres dans MLflow (DagsHub).")
    # === FIN AJOUT MLFLOW AVANCÉ ===
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modele sauvegarde dans {MODEL_PATH}")


if __name__ == "__main__":
    main()