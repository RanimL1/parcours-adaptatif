"""Chargement du CSV et découpage train / test."""
import numpy as np
import pandas as pd

KEYS = ["user_id_new", "skill_id_new"]
TARGET = "target_correct_rate"


def load(path):
    """Une ligne par couple (étudiant, compétence)."""
    df = pd.read_csv(path)
    df = df.dropna(subset=KEYS + [TARGET]).drop_duplicates(KEYS)
    return df.reset_index(drop=True)


def split_per_user(df, test_frac=0.2, min_items=5, seed=42):
    """Retire ~20 % des compétences de chaque étudiant (aléatoire, car pas de timestamp).

    Les étudiants avec moins de `min_items` compétences restent entièrement en train.
    """
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, g in df.groupby("user_id_new"):
        if len(g) < min_items:
            continue
        n = max(1, int(round(len(g) * test_frac)))
        test_idx.extend(rng.choice(g.index.values, n, replace=False))
    test = df.loc[test_idx]
    train = df.drop(index=test_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def save_processed(df, path="data/processed/interactions.parquet"):
    """Interactions nettoyées (une ligne par couple étudiant-compétence) au format Parquet."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[KEYS + [TARGET]].to_parquet(path, index=False)
    return path
