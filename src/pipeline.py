"""Assemblage commun (run.py, API, Streamlit, notebooks)."""
from dataclasses import dataclass
from .data_loader import load, split_per_user
from .student_model import StudentModel
from .knowledge_graph import build_graph
from .knowledge_filter import KnowledgeFilter
from .content_based import ContentBased
from .collaborative import SVDRecommender
from .hybrid import Hybrid
from .evaluation import ranking_metrics


@dataclass
class System:
    sm: object
    G: object
    kf: object
    cb: object
    cf: object
    hyb: object


def build(train, alpha=0.5, use_graph=True, k=20):
    sm = StudentModel().fit(train)
    G = build_graph(train, sm) if use_graph else None
    kf = KnowledgeFilter(sm, graph=G)
    cb = ContentBased(sm, train)
    cf = SVDRecommender(sm, k=k).fit(train)
    return System(sm, G, kf, cb, cf, Hybrid(sm, kf, cb, cf, alpha=alpha))


def tune_alpha(train, K=5, grid=(0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0), verbose=True):
    """Règle alpha sur une validation prise DANS le train (jamais sur le test)."""
    tr, val = split_per_user(train, seed=7)
    sysm = build(tr)
    best, best_ndcg = 0.5, -1
    for a in grid:
        sysm.hyb.alpha = a
        rank = lambda h, c: list(sysm.hyb.recommend(h, k=K)["skill_id_new"])
        _, ndcg = ranking_metrics(rank, val, sysm.sm, K)
        if verbose:
            print(f"  alpha={a:.1f}  NDCG@{K}(val)={ndcg:.4f}")
        if ndcg > best_ndcg:
            best, best_ndcg = a, ndcg
    return best
