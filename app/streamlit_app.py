
"""Application de recommandation de cours."""

import os
import sys
from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import streamlit as st

from src.data_loader import load
from src.pipeline import build

# Chemins absolus des données
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "interactions_real_rich_processed.csv"

NAMES_PATH = BASE_DIR / "data" / "raw" / "skill_names.csv"

st.set_page_config(
    page_title="Recommandation de cours",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource(show_spinner="Entraînement du système de recommandation…")
def get_system():

    # Vérification du dataset
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}"
        )

    # Construction du système
    df = load(DATA_PATH)
    S = build(df)

    # Chargement des noms des compétences
    names = {}

    if NAMES_PATH.is_file():
        nm = pd.read_csv(NAMES_PATH).dropna()
        names = dict(zip(
            nm.skill_id_new.astype(int),
            nm.skill_name
        ))
    else:
        st.warning(f"Fichier des noms introuvable : {NAMES_PATH}")

    # Calcul des niveaux de difficulté
    diff = pd.Series({
        s: S.sm.difficulty(s)
        for s in S.sm.skills
    })

    q1, q2 = diff.quantile([1 / 3, 2 / 3])

    return S, names, q1, q2


S, NAMES, Q1, Q2 = get_system()
