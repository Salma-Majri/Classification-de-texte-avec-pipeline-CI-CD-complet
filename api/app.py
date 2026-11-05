from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="API de Classification de Texte - IMDB",
    version="1.0.0",
    description="API FastAPI pour la classification des avis IMDB (Projet MLOps)"
)

class ReviewRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de classification de texte (IMDB) !"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(review: ReviewRequest):
    if not review.text or not review.text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")
    
    return {
        "text": review.text,
        "predicted_category": "Positif",
        "confidence": 0.85
    }