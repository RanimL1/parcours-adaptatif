"""Convertit le CSV brut ASSISTments en data/raw/interactions_real_rich_processed.csv
et sauvegarde les noms de cours dans data/raw/skill_names.csv

Utilisation (depuis la racine du projet) : python prepare_data.py
"""
import pandas as pd

RAW = "data/raw/assistments_2009_2010.csv"
OUT = "data/raw/interactions_real_rich_processed.csv"
NAMES = "data/raw/skill_names.csv"

df = pd.read_csv(RAW, usecols=["user_id", "skill_id", "skill_name", "correct"], encoding="latin-1", low_memory=False)
df = df.dropna(subset=["user_id", "skill_id", "correct"])
df["skill_id"] = df["skill_id"].astype(int)

agg = (df.groupby(["user_id", "skill_id"])["correct"]
         .mean().rename("target_correct_rate").reset_index())
agg["user_id_new"] = agg["user_id"].astype("category").cat.codes
agg["skill_id_new"] = agg["skill_id"].astype("category").cat.codes
agg[["user_id_new", "skill_id_new", "target_correct_rate"]].to_csv(OUT, index=False)

# correspondance skill_id_new -> nom du cours (un skill_id peut avoir plusieurs libellés : on garde le plus fréquent)
names = (df.dropna(subset=["skill_name"]).groupby("skill_id")["skill_name"]
           .agg(lambda s: s.value_counts().index[0]).rename("skill_name").reset_index())
ids = agg[["skill_id", "skill_id_new"]].drop_duplicates()
ids.merge(names, on="skill_id", how="left")[["skill_id_new", "skill_name"]] \
   .sort_values("skill_id_new").to_csv(NAMES, index=False)

print(agg.shape, agg.user_id_new.nunique(), "étudiants,", agg.skill_id_new.nunique(), "cours")