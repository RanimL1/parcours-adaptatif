"""Graphe de prérequis APPRIS à partir des données (compétences anonymisées -> pas de graphe manuel possible).

Règle : A est prérequis de B si
  - assez d'étudiants ont fait A et B (min_common),
  - parmi ceux qui maîtrisent B, une grande part maîtrise aussi A (confiance >= min_conf),
  - cette confiance dépasse le taux de maîtrise de base de A (lift >= min_lift, sinon A est juste « facile »),
  - A est nettement plus facile que B (ordre strict -> le graphe est un DAG).
Ce sont des corrélations, pas des prérequis pédagogiques validés : à signaler dans les limites.
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
from .data_loader import TARGET


def build_graph(train, sm, thr=0.6, min_common=30, min_conf=0.8, min_lift=1.05,
                min_gap=0.05, max_prereq=3):
    users = sorted(train["user_id_new"].unique())
    skills = list(sm.skills)
    uidx = {u: i for i, u in enumerate(users)}
    sidx = {s: i for i, s in enumerate(skills)}
    O = np.zeros((len(users), len(skills)))
    Ms = np.zeros_like(O)
    ui = train["user_id_new"].map(uidx).values
    si = train["skill_id_new"].map(sidx).values
    O[ui, si] = 1
    Ms[ui, si] = (train[TARGET].values >= thr).astype(float)

    n_ab = O.T @ O                       # étudiants ayant fait A et B
    b_mast = O.T @ Ms                    # [a,b] : A fait ET B maîtrisé
    both = Ms.T @ Ms                     # [a,b] : A et B maîtrisés
    a_mast = Ms.T @ O                    # [a,b] : A maîtrisé ET B fait
    conf = both / np.maximum(b_mast, 1)  # P(A maîtrisé | B maîtrisé)
    base = a_mast / np.maximum(n_ab, 1)  # P(A maîtrisé | A et B faits)
    lift = conf / np.maximum(base, 1e-9)
    diff = np.array([sm.difficulty(s) for s in skills])

    G = nx.DiGraph()
    for s, d in zip(skills, diff):
        G.add_node(s, difficulty=float(d))
    for b in range(len(skills)):
        ok = [a for a in range(len(skills))
              if a != b and n_ab[a, b] >= min_common and b_mast[a, b] >= min_common / 2
              and conf[a, b] >= min_conf and lift[a, b] >= min_lift and diff[a] + min_gap <= diff[b]]
        for a in sorted(ok, key=lambda a: -conf[a, b])[:max_prereq]:
            G.add_edge(skills[a], skills[b], confiance=float(conf[a, b]), effectif=int(n_ab[a, b]))
    return G


def save_knowledge(G, sm, folder="data/knowledge"):
    """Écrit competences.csv (id, nom, niveau) et prerequis.csv (competence, prerequis)."""
    os.makedirs(folder, exist_ok=True)
    comp = pd.DataFrame({"id": sm.skills})
    comp["nom"] = [f"Compétence {s}" for s in sm.skills]        # à renommer si tu retrouves les vrais noms
    diff = pd.Series([sm.difficulty(s) for s in sm.skills])
    comp["niveau"] = pd.qcut(diff.rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
    comp.to_csv(os.path.join(folder, "competences.csv"), index=False)
    pre = pd.DataFrame([{"competence": b, "prerequis": a, "confiance": round(d["confiance"], 3),
                         "effectif": d["effectif"]} for a, b, d in G.edges(data=True)],
                       columns=["competence", "prerequis", "confiance", "effectif"])
    pre.to_csv(os.path.join(folder, "prerequis.csv"), index=False)
    return comp, pre


def load_graph(folder="data/knowledge"):
    comp = pd.read_csv(os.path.join(folder, "competences.csv"))
    pre = pd.read_csv(os.path.join(folder, "prerequis.csv"))
    G = nx.DiGraph()
    G.add_nodes_from(comp["id"])
    for _, r in pre.iterrows():
        G.add_edge(r["prerequis"], r["competence"], confiance=r["confiance"], effectif=r["effectif"])
    return G
