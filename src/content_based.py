"""Contenu : similarité entre compétences calculée sur leurs caractéristiques
(difficulté, popularité, niveau moyen des étudiants qui la travaillent)."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


class ContentBased:
    def __init__(self, sm, train, bandwidth=1.0):
        lvl = train["user_id_new"].map(sm.user_level)
        audience = train.assign(lvl=lvl).groupby("skill_id_new")["lvl"].mean()
        F = pd.DataFrame({"diff": 1 - sm.skill_mean,
                          "pop": np.log1p(sm.skill_pop),
                          "audience": audience})
        Z = ((F - F.mean()) / F.std().replace(0, 1)).fillna(0)
        d2 = squareform(pdist(Z.values, "sqeuclidean"))
        self.S = np.exp(-d2 / (2 * bandwidth ** 2))   # similarité gaussienne
        self.idx = {s: i for i, s in enumerate(Z.index)}
        self.sm = sm

    def score(self, history, cands):
        """Taux de réussite prédit = moyenne pondérée par la similarité des compétences déjà travaillées."""
        cands = list(cands)
        prior = np.array([self.sm.skill_mean.get(s, self.sm.global_mean) for s in cands])
        h = [s for s in history.index if s in self.idx]
        if not h:
            return pd.Series(prior, index=cands)
        hi = [self.idx[s] for s in h]
        r = history.loc[h].values
        pred = prior.copy()
        for k, s in enumerate(cands):
            if s in self.idx:
                w = self.S[self.idx[s], hi]
                pred[k] = (w @ r) / (w.sum() + 1e-9)
        return pd.Series(pred, index=cands)
