
"""Application de recommandation de cours - Diagnostic."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import streamlit as st

from src.data_loader import load
from src.pipeline import build

DATA_PATH = BASE_DIR / "data" / "raw" / "interactions_real_rich_processed.csv"
NAMES_PATH = BASE_DIR / "data" / "raw" / "skill_names.csv"

st.set_page_config(
    page_title="Recommandation de cours",
    page_icon="🎓",
    layout="wide"
)

# Interface de diagnostic
st.title("🎓 Application de recommandation de cours")
st.success("L'interface Streamlit fonctionne !")

st.write("Racine du projet :", str(BASE_DIR))
st.write("Dataset trouvé :", DATA_PATH.is_file())
st.write("Fichier des compétences trouvé :", NAMES_PATH.is_file())


@st.cache_resource(show_spinner="Entraînement du système de recommandation…")
def get_system():

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

    df = load(DATA_PATH)
    S = build(df)

    names = {}

    if NAMES_PATH.is_file():
        nm = pd.read_csv(NAMES_PATH).dropna()
        names = dict(zip(
            nm.skill_id_new.astype(int),
            nm.skill_name
        ))

    diff = pd.Series({
        s: S.sm.difficulty(s)
        for s in S.sm.skills
    })

    q1, q2 = diff.quantile([1 / 3, 2 / 3])

    return S, names, q1, q2


if st.button("Charger le système de recommandation"):

    try:
        with st.spinner("Chargement en cours..."):
            S, NAMES, Q1, Q2 = get_system()

        st.success("Système chargé avec succès !")
        st.write("Nombre de compétences :", len(S.sm.skills))

    except Exception as e:
        st.error("Une erreur est survenue pendant le chargement.")
        st.exception(e)
