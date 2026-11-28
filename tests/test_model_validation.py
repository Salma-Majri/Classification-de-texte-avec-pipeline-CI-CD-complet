import joblib
import os
import pytest
import sys
import subprocess
from src.preprocessing import clean_text
# Chemin relatif vers le modèle (comme dans train.py)
MODEL_PATH = os.path.join("models", "model.pkl")

@pytest.fixture(scope="module")
def model():
    """Charge le modele ; le regenere via train.py s'il est absent (cas CI)."""
    if not os.path.exists(MODEL_PATH):
        print("\n[INFO] Modele absent du workspace, regeneration via src.train ...\n")
        subprocess.run(
            [sys.executable, "-m", "src.train", "--sample", "10000"],
            check=True,
        )
    
    assert os.path.exists(MODEL_PATH), f"Le modele n'existe pas a {MODEL_PATH}"
    return joblib.load(MODEL_PATH)
def test_seuil_de_performance(model):
    """Vérifie que le modèle atteint le seuil minimal d'accuracy sur un échantillon de test."""
    # On utilise un petit échantillon connu pour valider rapidement
    from datasets import load_dataset
    import pandas as pd
    from sklearn.metrics import accuracy_score
    
    dataset = load_dataset("imdb", split="test[:1000]") # 1000 exemples pour aller vite
    df = pd.DataFrame(dataset)
    df["clean"] = df["text"].apply(clean_text)
    
    y_pred = model.predict(df["clean"])
    acc = accuracy_score(df["label"], y_pred)
    
    # Le seuil exigé par le contrat
    assert acc >= 0.84, f"L'accuracy est de {acc:.4f}, en dessous du seuil de 0.84 requis."


def test_format_des_sorties(model):
    """Vérifie que predict_proba renvoie bien des probabilités entre 0 et 1."""
    texte_test = "This is a wonderful and amazing movie"
    texte_clean = clean_text(texte_test)
    probas = model.predict_proba([texte_clean])[0]
    
    assert len(probas) == 2, "Le modèle doit renvoyer 2 probabilités (classe 0 et classe 1)"
    assert all(0.0 <= p <= 1.0 for p in probas), "Les probabilités doivent être comprises entre 0 et 1"
    assert abs(sum(probas) - 1.0) < 1e-5, "La somme des probabilités doit être égale à 1"


def test_cas_limites(model):
    """Vérifie que le modèle ne plante pas sur des entrées étranges (cas limites)."""
    cas_limites = [
        "",                                  # Texte vide
        "   ",                               # Que des espaces
        "a",                                 # Un seul caractère
        "wow " * 500,                        # Texte très long
        "12345 !@#$%",                       # Que des chiffres et symboles
    ]
    
    for texte in cas_limites:
        texte_clean = clean_text(texte)
        # Le modèle ne doit pas lever d'exception
        probas = model.predict_proba([texte_clean])[0]
        assert len(probas) == 2