"""Application de recommandation de cours.

Lancer (depuis la racine du projet) :  streamlit run app/app_cours.py
Prérequis : python prepare_data.py  (génère le CSV traité et data/raw/skill_names.csv)
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from src.data_loader import load
from src.pipeline import build

DATA_PATH = os.getenv("DATA_PATH", "data/raw/interactions_real_rich_processed.csv")
NAMES_PATH = "data/raw/skill_names.csv"

st.set_page_config(page_title="Recommandation de cours", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Entraînement du système de recommandation…")
def get_system():
    S = build(load(DATA_PATH))
    names = {}
    if os.path.exists(NAMES_PATH):
        nm = pd.read_csv(NAMES_PATH).dropna()
        names = dict(zip(nm.skill_id_new.astype(int), nm.skill_name))
    diff = pd.Series({s: S.sm.difficulty(s) for s in S.sm.skills})
    q1, q2 = diff.quantile([1 / 3, 2 / 3])
    return S, names, q1, q2


S, NAMES, Q1, Q2 = get_system()


def course(s):
    return NAMES.get(int(s), f"Cours {int(s)}")


def level_label(s):
    d = S.sm.difficulty(s)
    return "🟢 Débutant" if d <= Q1 else ("🟡 Intermédiaire" if d <= Q2 else "🔴 Avancé")


def reasons(h, s, cold):
    """Explication en langage naturel d'une recommandation."""
    out = []
    blocked = S.kf._blocked_skills(h, s)
    if blocked:
        b = blocked[0]
        out.append(f"Ce cours est un prérequis de **{course(b)}**, où ta réussite n'est que de {h[b]:.0%} : "
                   "le revoir devrait t'aider à progresser.")
    pre = S.kf.prereqs(s)
    if pre:
        out.append("Tu maîtrises ses prérequis : " + ", ".join(course(p) for p in pre) + ".")
    else:
        out.append("Aucun prérequis n'est nécessaire.")
    out.append(f"Sa difficulté ({S.sm.difficulty(s):.0%}) est adaptée à ton niveau estimé ({S.sm.level(h):.0%}).")
    if cold:
        out.append("Ton historique est court : la recommandation s'appuie surtout sur les caractéristiques des cours.")
    else:
        out.append("Des étudiants au profil proche du tien ont bien réussi ce cours.")
    return out


# ------------------------------------------------------------------ barre latérale
st.sidebar.header("Paramètres")
k = st.sidebar.slider("Nombre de cours recommandés", 1, 10, 5)
max_diff = st.sidebar.slider("Difficulté maximale", 0.0, 1.0, 1.0, 0.05,
                             help="0 = très facile, 1 = très difficile")
S.hyb.alpha = st.sidebar.slider("α (poids du contenu vs collaboratif)", 0.0, 1.0, 0.5, 0.1)

st.title("🎓 Recommandation de cours")
st.caption("Système hybride : contenu + filtrage collaboratif (SVD) + filtre de niveau et de prérequis.")

mode = st.radio("Qui es-tu ?", ["Étudiant existant", "Nouvel étudiant (test de positionnement)"], horizontal=True)

if mode == "Étudiant existant":
    sid = st.selectbox("Étudiant", sorted(S.sm.history.keys()))
    h = S.sm.get_history(sid)
else:
    ordered = sorted(S.sm.skills, key=S.sm.difficulty)
    test = [int(s) for s in ordered[:: max(1, len(ordered) // 5)][:5]]
    st.write("Indique ta réussite (0 % à 100 %) sur ces cours, du plus facile au plus difficile :")
    cols = st.columns(len(test))
    h = pd.Series({s: c.slider(course(s), 0, 100, 50, 5, key=f"t{s}") / 100 for s, c in zip(test, cols)})

cold = len(h) < S.hyb.cold_n

# ------------------------------------------------------------------ profil
c1, c2, c3 = st.columns(3)
c1.metric("Niveau estimé", f"{S.sm.level(h):.0%}")
c2.metric("Cours déjà suivis", len(h))
c3.metric("Profil", "Nouveau (cold-start)" if cold else "Historique suffisant")

with st.expander("Mes cours déjà suivis"):
    if len(h):
        st.dataframe(pd.DataFrame({"Cours": [course(s) for s in h.index],
                                   "Réussite": (h.values * 100).round(0).astype(int),
                                   "Niveau": [level_label(s) for s in h.index]})
                     .sort_values("Réussite"), hide_index=True, width="stretch")
    else:
        st.write("Aucun cours suivi.")

# ------------------------------------------------------------------ recommandations
st.subheader("Cours recommandés pour toi")
recs = S.hyb.recommend(h, k=k, max_difficulty=max_diff)
for i, r in enumerate(recs.itertuples(), 1):
    s = int(r.skill_id_new)
    with st.container(border=True):
        a, b = st.columns([4, 1])
        a.markdown(f"### {i}. {course(s)}")
        b.markdown(f"**{level_label(s)}**")
        st.progress(min(max(float(r.score), 0.0), 1.0), text=f"Réussite estimée : {min(max(r.score, 0), 1):.0%}")
        st.markdown("**Pourquoi ce cours ?**")
        for line in reasons(h, s, cold):
            st.markdown(f"- {line}")
        with st.expander("Détail technique"):
            st.caption(r.explication)