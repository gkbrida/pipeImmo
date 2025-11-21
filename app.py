from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from joblib import load

# Charger le modèle
saved = load("pipeline_prix_vente_terrains.joblib")

app = FastAPI(title="Land Price Predictor")

# Définition des entrées
class InputData(BaseModel):
    superficie_m2: float
    titreFoncier: int
    acces: int
    prix_moyen: float
    prix_min: float
    prix_max: float
    prix_median: float
    prix_q1: float
    prix_q3: float
    variance_prix: float

@app.post("/")
def predict_price(data: InputData):
    # Convertir en DataFrame avec une seule ligne
    df = pd.DataFrame([data.dict()])

    # Prédiction
    pipeline = saved['best_model_']
    pred = pipeline.predict(df)[0]

    return {"prediction": float(pred)}
