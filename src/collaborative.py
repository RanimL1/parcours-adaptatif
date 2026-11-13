"""Filtrage collaboratif : SVD sur la matrice étudiant x compétence (centrée par étudiant).
Un étudiant (ou un historique tronqué) est projeté par « fold-in » (moindres carrés ridge)."""
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from .data_loader import TARGET


class SVDRecommender:
    def __init__(self, sm, k=20, reg=1.0):
        self.sm, self.k, self.reg = sm, k, reg

    def fit(self, train):
        users = sorted(train["user_id_new"].unique())
        self.sidx = {s: i for i, s in enumerate(self.sm.skills)}
        uidx = {u: i for i, u in enumerate(users)}
        R = np.zeros((len(users), len(self.sm.skills)))
        M = np.zeros_like(R)
        ui = train["user_id_new"].map(uidx).values
        si = train["skill_id_new"].map(self.sidx).values
        R[ui, si] = train[TARGET].values
        M[ui, si] = 1
        mu = R.sum(1) / np.maximum(M.sum(1), 1)
        Rc = (R - mu[:, None]) * M
        k = max(1, min(self.k, min(R.shape) - 1))
        _, _, Vt = svds(Rc, k=k)
        self.Q = Vt                      # k x n_skills : facteurs de compétences
        return self

    def score(self, history, cands):
        cands = list(cands)
        h = [s for s in history.index if s in self.sidx]
        if not h:
            return pd.Series([self.sm.skill_mean.get(s, self.sm.global_mean) for s in cands], index=cands)
        mu = self.sm.level(history)
        r = history.loc[h].values - mu
        Qo = self.Q[:, [self.sidx[s] for s in h]]
        p = np.linalg.solve(Qo @ Qo.T + self.reg * np.eye(self.Q.shape[0]), Qo @ r)
        pred = np.full(len(cands), mu)
        for k, s in enumerate(cands):
            if s in self.sidx:
                pred[k] = mu + p @ self.Q[:, self.sidx[s]]
        return pd.Series(np.clip(pred, 0, 1), index=cands)
