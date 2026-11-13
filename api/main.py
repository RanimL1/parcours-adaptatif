"""API : uvicorn api.main:app --reload   (depuis la racine du projet)

Variables d'environnement : DATA_PATH (CSV), ALPHA (défaut 0.5)
"""
import os
import sys
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_loader import load
from src.pipeline import build

DATA_PATH = os.getenv("DATA_PATH", "data/raw/interactions_real_rich_processed.csv")
ALPHA = float(os.getenv("ALPHA", "0.5"))

app = FastAPI(title="Parcours d'apprentissage adaptatif")
S = None


@app.on_event("startup")
def startup():
    global S
    S = build(load(DATA_PATH), alpha=ALPHA)   # modèle entraîné sur toutes les données


class NewStudent(BaseModel):
    resultats: Dict[int, float]   # {competence_id: taux de réussite 0..1} (test de positionnement)
    k: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "etudiants": len(S.sm.history), "competences": len(S.sm.skills),
            "aretes_prerequis": S.G.number_of_edges()}


@app.get("/recommend/{student_id}")
def recommend(student_id: int, k: int = 5, max_difficulty: Optional[float] = None):
    known = student_id in S.sm.history
    h = S.sm.get_history(student_id)
    recs = S.hyb.recommend(h, k=k, max_difficulty=max_difficulty)
    return {"student_id": student_id, "cold_start": (not known) or len(h) < S.hyb.cold_n,
            "niveau_estime": round(S.sm.level(h), 3), "recommandations": recs.to_dict("records")}


@app.get("/positioning-test")
def positioning_test(n: int = 5):
    """Cold start : compétences réparties du plus facile au plus difficile à faire passer à un nouvel étudiant."""
    ordered = sorted(S.sm.skills, key=S.sm.difficulty)
    step = max(1, len(ordered) // n)
    return {"competences": [int(s) for s in ordered[::step][:n]]}


@app.post("/recommend")
def recommend_new(body: NewStudent):
    if not body.resultats:
        raise HTTPException(400, "resultats vide : utiliser /positioning-test d'abord")
    h = pd.Series(body.resultats, dtype=float)
    recs = S.hyb.recommend(h, k=body.k)
    return {"cold_start": True, "niveau_estime": round(S.sm.level(h), 3),
            "recommandations": recs.to_dict("records")}
