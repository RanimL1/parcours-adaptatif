"""Pipeline complet : python run.py [chemin_du_csv]"""
import sys
import pandas as pd
from src.data_loader import load, split_per_user, save_processed
from src.knowledge_graph import save_knowledge
from src.pipeline import build, tune_alpha
from src.evaluation import compare, rank_by

K = 5


def main(path):
    df = load(path)
    save_processed(df)
    nu, ns = df.user_id_new.nunique(), df.skill_id_new.nunique()
    print(f"{len(df)} lignes, {nu} étudiants, {ns} compétences, densité {len(df) / (nu * ns):.2%}")
    train, test = split_per_user(df)

    print("Réglage de alpha (validation) :")
    alpha = tune_alpha(train, K)
    print("alpha retenu =", alpha)

    S = build(train, alpha)
    comp, pre = save_knowledge(S.G, S.sm)
    print(f"Graphe de prérequis : {S.G.number_of_edges()} arêtes -> data/knowledge/")

    sm, hyb = S.sm, S.hyb
    base = lambda h, c: pd.Series([sm.skill_mean.get(s, sm.global_mean) for s in c], index=c)
    scorers = {"Popularité (baseline)": base, "Contenu seul": S.cb.score,
               "Collaboratif (SVD)": S.cf.score, "Hybride sans filtre": hyb.score}
    models = {n: (f, rank_by(f)) for n, f in scorers.items()}
    models["Hybride + filtre (niveau + prérequis)"] = (
        hyb.score, lambda h, c: list(hyb.recommend(h, k=len(c))["skill_id_new"]))
    res = compare(models, test, sm, K)
    print("\n=== Résultats comparatifs ===")
    print(res.to_string(index=False))
    res.to_csv("resultats.csv", index=False)

    users = sm.user_count.sort_values()
    rich, mid = users.index[-1], users.index[len(users) // 2]
    profils = {
        "A (cold start : 1 seule interaction)": (sm.get_history(mid).iloc[:1], {}),
        "B (historique riche)": (sm.get_history(rich), {}),
        "C (contraintes : difficulté <= 0.5)": (sm.get_history(rich), {"max_difficulty": 0.5}),
    }
    for nom, (h, kw) in profils.items():
        print(f"\n--- Utilisateur {nom} | {len(h)} compétences connues ---")
        print(hyb.recommend(h, k=K, **kw).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/interactions_real_rich_processed.csv")
