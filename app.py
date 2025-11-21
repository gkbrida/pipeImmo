from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from joblib import load
from typing import List


# Charger le modèle sauvegardé
pipeline = load("pipeline_prix_vente_terrains.joblib")  # ton pipeline RealEstatePipeline

app = FastAPI(title="land Price Predictor")

# Définition des entrées
class InputData(BaseModel):
    # toutes les colonnes features requises
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

@app.post("/predict")
def predict_price(data: List[InputData]):
    # Convertir en DataFrame
    df = pd.DataFrame([d.dict() for d in data])
    # Prédiction
    preds = pipeline.predict(df)
    return {"predictions": preds.tolist()}
