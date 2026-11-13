"""Métriques : RMSE/MAE (prédiction) et Precision@K / NDCG@K (classement)."""
import numpy as np
import pandas as pd
from .data_loader import TARGET


def rmse_mae(score_fn, test, sm):
    errs = []
    for u, g in test.groupby("user_id_new"):
        pred = score_fn(sm.get_history(u), list(g["skill_id_new"]))
        errs.extend(pred.values - g[TARGET].values)
    errs = np.array(errs)
    return float(np.sqrt((errs ** 2).mean())), float(np.abs(errs).mean())


def ranking_metrics(rank_fn, test, sm, K=5, thr=0.6):
    """Pertinent = compétence du test réussie (taux >= thr). Candidats = tout ce que l'étudiant n'a pas vu en train."""
    prec, ndcg = [], []
    for u, g in test.groupby("user_id_new"):
        rel = set(g.loc[g[TARGET] >= thr, "skill_id_new"])
        if not rel:
            continue
        h = sm.get_history(u)
        cands = [s for s in sm.skills if s not in h.index]
        top = rank_fn(h, cands)[:K]
        hits = [1 if s in rel else 0 for s in top]
        prec.append(sum(hits) / K)
        dcg = sum(x / np.log2(i + 2) for i, x in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), K)))
        ndcg.append(dcg / idcg)
    return float(np.mean(prec)), float(np.mean(ndcg))


def rank_by(score_fn):
    return lambda h, cands: list(score_fn(h, cands).sort_values(ascending=False).index)


def compare(models, test, sm, K=5):
    """models : {nom: (score_fn, rank_fn)}"""
    rows = []
    for name, (score_fn, rank_fn) in models.items():
        rmse, mae = rmse_mae(score_fn, test, sm)
        p, n = ranking_metrics(rank_fn, test, sm, K)
        rows.append({"modèle": name, "RMSE": rmse, "MAE": mae, f"Precision@{K}": p, f"NDCG@{K}": n})
    return pd.DataFrame(rows).round(4)
