# Classification de texte avec chaîne CI/CD complète

Projet de fin de module **DevOps & MLOps** (Pr. Soufiane HAMIDA).
Catégorisation automatique d'avis (IMDB Reviews) avec déploiement continu sécurisé par des tests et un retour arrière.

## État d'avancement

- [x] Structure du projet et dépendances
- [x] API FastAPI (`/`, `/health`, `/predict`), `/predict` renvoie pour l'instant une réponse factice
- [x] Tests unitaires (pytest)
- [x] CI GitHub Actions : tests + build Docker + smoke test
- [x] Image Docker de l'API
- [ ] Modèle NLP (TF-IDF + LogReg) intégré à l'API
- [ ] Suivi des expériences (MLflow), données versionnées (DVC), rapports CML
- [ ] Déploiement staging puis production, retour arrière automatique
- [ ] Supervision

## Structure du projet

```
.
├── .github/workflows/   # pipeline CI/CD (GitHub Actions)
├── api/                 # API FastAPI (app.py)
├── docker/              # Dockerfile.api
├── src/                 # code NLP (prétraitement, entraînement)
├── tests/               # tests unitaires (pytest)
└── requirements.txt
```

## Installation locale

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Lancer l'API

```bash
uvicorn api.app:app --reload
```
Documentation interactive : http://localhost:8000/docs

| Endpoint   | Méthode | Rôle                          |
|------------|---------|-------------------------------|
| `/`        | GET     | Message de bienvenue          |
| `/health`  | GET     | État de l'API                 |
| `/predict` | POST    | Prédiction à partir d'un texte|

Exemple :
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"great movie"}'
```

## Tests

```bash
python -m pytest tests/ -v
```

## Docker

```bash
docker build -t text-classification-api:latest -f docker/Dockerfile.api .
docker run -d -p 8000:8000 --name api text-classification-api:latest
curl http://localhost:8000/health
```

## CI/CD

Le workflow `.github/workflows/ci-cd.yml` se déclenche sur `push` (`main`, `devops/*`) et sur `pull_request` vers `main` :
1. installation des dépendances
2. tests unitaires
3. build de l'image Docker
4. smoke test du conteneur (`/health`)

## Organisation Git

- `main` : branche stable
- `devops/*` : CI/CD, Docker, déploiement
- `nlp/*` : modèle et prétraitement
- Commits au format `type(scope): message` (ex. `feat(devops): ...`)

## Équipe

- Salma Majri : DevOps / MLOps
- Fatimazahra Massane : NLP
