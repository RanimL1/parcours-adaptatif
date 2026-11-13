"""Hybride : CASCADE (filtre de niveau) puis PONDÉRATION (alpha * contenu + (1-alpha) * collaboratif)."""
import pandas as pd


class Hybrid:
    def __init__(self, sm, kf, cb, cf, alpha=0.5, cold_n=3, cold_alpha=0.8):
        self.sm, self.kf, self.cb, self.cf = sm, kf, cb, cf
        self.alpha, self.cold_n, self.cold_alpha = alpha, cold_n, cold_alpha

    def _alpha(self, history):
        # cold start : le CF est peu fiable -> on s'appuie davantage sur le contenu
        return self.cold_alpha if len(history) < self.cold_n else self.alpha

    def score(self, history, cands):
        a = self._alpha(history)
        return a * self.cb.score(history, cands) + (1 - a) * self.cf.score(history, cands)

    def recommend(self, history, k=5, use_filter=True, max_difficulty=None, exclude=()):
        cands = [s for s in self.sm.skills if s not in history.index and s not in set(exclude)]
        if max_difficulty is not None:
            cands = [s for s in cands if self.sm.difficulty(s) <= max_difficulty]
        if use_filter:
            cands = self.kf.eligible(history, cands)
        sc = self.score(history, cands)
        sc = sc + pd.Series({s: self.kf.bonus(history, s) for s in sc.index})   # remédiation (prérequis non acquis)
        sc = sc.sort_values(ascending=False).head(k)
        a = self._alpha(history)
        out = []
        for s, v in sc.items():
            why = (f"{self.kf.explain(history, s)} ; score hybride {v:.2f} "
                   f"(α={a:.1f} contenu / {1 - a:.1f} collaboratif)")
            out.append({"skill_id_new": s, "score": round(float(v), 3), "explication": why})
        return pd.DataFrame(out)
