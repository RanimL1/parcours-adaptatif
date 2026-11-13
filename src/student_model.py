"""Modèle de niveau : maîtrise par compétence + niveau global (calculés sur le TRAIN uniquement).

On ne réutilise pas user_mean_correct / skill_mean_correct du CSV : ils sont calculés
sur toutes les données et contiennent donc la cible du test (fuite).
"""
import numpy as np
import pandas as pd
from .data_loader import TARGET


class StudentModel:
    def __init__(self, shrink=3):
        self.shrink = shrink  # lissage bayésien pour les étudiants avec peu d'historique

    def fit(self, train):
        self.global_mean = float(train[TARGET].mean())
        g = train.groupby("user_id_new")[TARGET]
        self.user_level, self.user_count = g.mean(), g.size()
        gs = train.groupby("skill_id_new")[TARGET]
        self.skill_mean, self.skill_pop = gs.mean(), gs.size()
        self.history = {u: d.set_index("skill_id_new")[TARGET]
                        for u, d in train.groupby("user_id_new")}
        self.skills = np.array(sorted(self.skill_mean.index))
        return self

    def get_history(self, user):
        return self.history.get(user, pd.Series(dtype=float))

    def level(self, history):
        """Niveau estimé dans [0, 1] ; retombe sur la moyenne globale si historique vide."""
        n = len(history)
        return (history.sum() + self.shrink * self.global_mean) / (n + self.shrink)

    def difficulty(self, skill):
        return 1.0 - float(self.skill_mean.get(skill, self.global_mean))
