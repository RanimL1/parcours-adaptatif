"""Bloc « connaissances » : règle de niveau + (optionnel) graphe de prérequis appris."""


class KnowledgeFilter:
    def __init__(self, sm, margin=0.15, min_keep=10, graph=None, mastery=0.6, struggle=0.5, bonus=0.1):
        self.sm, self.margin, self.min_keep = sm, margin, min_keep
        self.graph, self.mastery, self.struggle, self.bonus_value = graph, mastery, struggle, bonus

    # --- graphe ---
    def prereqs(self, skill):
        if self.graph is None or skill not in self.graph:
            return []
        return [int(p) if hasattr(p, "item") else p for p in self.graph.predecessors(skill)]

    def _mastered(self, history, p):
        return p in history.index and history[p] >= self.mastery

    def _blocked_skills(self, history, skill):
        """Compétences en difficulté (réussite < struggle) dont `skill` est un prérequis."""
        if self.graph is None or skill not in self.graph:
            return []
        return [b for b in self.graph.successors(skill) if b in history.index and history[b] < self.struggle]

    def bonus(self, history, skill):
        """Petit bonus de score pour une compétence qui débloque une compétence où l'étudiant échoue."""
        return self.bonus_value if self._blocked_skills(history, skill) else 0.0

    # --- filtre ---
    def eligible(self, history, candidates):
        lvl = self.sm.level(history)
        ok = [s for s in candidates if self.sm.difficulty(s) <= lvl + self.margin]
        if self.graph is not None:
            g = [s for s in ok if all(self._mastered(history, p) for p in self.prereqs(s))]
            if len(g) >= self.min_keep:
                ok = g
        if len(ok) < self.min_keep:  # sécurité : ne jamais renvoyer une liste vide
            ok = sorted(candidates, key=self.sm.difficulty)[: self.min_keep]
        return ok

    def explain(self, history, skill):
        lvl, d = self.sm.level(history), self.sm.difficulty(skill)
        txt = f"difficulté {d:.2f} ≤ niveau estimé {lvl:.2f} + marge {self.margin:.2f}"
        blocked = self._blocked_skills(history, skill)
        if blocked:
            b = blocked[0]
            return (f"recommandé car le prérequis (compétence {skill}) de la compétence {b} "
                    f"(réussite {history[b]:.0%}) n'est pas acquis ; {txt}")
        pre = self.prereqs(skill)
        if pre:
            return f"prérequis {pre} acquis ; {txt}"
        return f"aucun prérequis ; {txt}"
