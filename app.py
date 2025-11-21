from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from joblib import load
from typing import List
from sklearn.ensemble import IsolationForest
from typing import Optional, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


class OutlierTagger(BaseEstimator, TransformerMixin):
    """Tagge les outliers avec IsolationForest et ajoute une colonne 'is_outlier'.
    Ne supprime pas les lignes (compatible avec CV).
    Paramètres:
        contamination: proportion estimée d'outliers
        random_state
    """
    def __init__(self, contamination:float=0.01, random_state:Optional[int]=0):
        self.contamination = contamination
        self.random_state = random_state
        self.iforest = None

    def fit(self, X, y=None):
        # IsolationForest attend un array; on clone pour être propre
        self.iforest = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        self.iforest.fit(X)
        return self

    def transform(self, X, y=None):
        X = np.asarray(X)
        labels = self.iforest.predict(X)  # 1 pour inlier, -1 pour outlier
        is_outlier = (labels == -1).astype(int).reshape(-1, 1)
        return np.hstack([X, is_outlier])

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
