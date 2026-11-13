# Parcours d'apprentissage adaptatif — recommandation hybride

Système de recommandation de compétences à travailler, combinant :

- **Contenu** : profil de l'étudiant comparé aux caractéristiques des compétences ;
- **Filtrage collaboratif** : factorisation matricielle (SVD) ;
- **Graphe de prérequis** appris par corrélation entre compétences ;
- **Filtre de connaissances** : niveau estimé de l'étudiant et prérequis acquis.

Chaque recommandation est accompagnée d'une explication (prérequis, difficulté, niveau estimé, poids contenu/collaboratif).

Données : [ASSISTments 2009-2010](https://sites.google.com/site/assistmentsdata/) (interactions étudiant-problème).

## Stack

Python · pandas · NumPy · SciPy · scikit-learn · NetworkX · FastAPI · Streamlit · Jupyter

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Données

Le CSV brut n'est pas versionné (83 Mo). Placer `assistments_2009_2010.csv` dans `data/raw/`, puis générer le fichier d'interactions :

```bash
python prepare_data.py
```

Cela produit `data/raw/interactions_real_rich_processed.csv` (une ligne par couple étudiant-compétence, colonnes `user_id_new`, `skill_id_new`, `target_correct_rate`).

## Utilisation (depuis la racine du projet)

```bash
python run.py                          # pipeline complet : Parquet, graphe, résultats comparatifs, profils A/B/C
uvicorn api.main:app --reload          # API : http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py     # démo : http://localhost:8501
jupyter notebook notebooks/            # notebooks 01 -> 06
```

### API

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | État du service (étudiants, compétences, arêtes de prérequis) |
| GET | `/recommend/{student_id}?k=5&max_difficulty=0.5` | Recommandations pour un étudiant existant |
| GET | `/positioning-test?n=5` | Compétences à proposer à un nouvel étudiant (cold start) |
| POST | `/recommend` | Recommandations à partir de résultats de positionnement `{"resultats": {id: taux}, "k": 5}` |

Variables d'environnement : `DATA_PATH` (CSV), `ALPHA` (défaut 0.5).

## Résultats

Évaluation sur 20 % des compétences retirées de chaque étudiant (K = 5) :

| Modèle | RMSE | MAE | Precision@5 | NDCG@5 |
|---|---|---|---|---|
| Popularité (baseline) | 0.2937 | 0.2331 | 0.0369 | 0.0454 |
| Contenu seul | 0.2698 | 0.2063 | 0.0574 | 0.0765 |
| Collaboratif (SVD) | 0.2684 | 0.2089 | 0.1681 | 0.2716 |
| Hybride sans filtre | 0.2684 | 0.2089 | 0.1681 | 0.2716 |
| Hybride + filtre (niveau + prérequis) | 0.2684 | 0.2089 | 0.1144 | 0.1729 |

Le réglage de α sur validation retient 0 (collaboratif seul). Le filtre niveau + prérequis réduit le NDCG : il impose des contraintes pédagogiques au détriment du classement pur.

## Arborescence

```
data/raw/         CSV d'origine (non versionné)
data/processed/   interactions.parquet (généré)
data/knowledge/   competences.csv, prerequis.csv (générés par le graphe appris)
notebooks/        01_exploration ... 06_evaluation
src/              data_loader, knowledge_graph, knowledge_filter, student_model,
                  content_based, collaborative, hybrid, evaluation, pipeline
api/main.py       API FastAPI
app/              démo Streamlit
run.py            pipeline complet
prepare_data.py   génération du CSV d'interactions
```

## Limites

- Pas de timestamp : découpage aléatoire par étudiant (pas temporel).
- Compétences anonymisées : prérequis **appris** par corrélation (non validés pédagogiquement) ; le nom est « Compétence id ».
- Niveau et difficulté recalculés sur le train pour éviter la fuite des colonnes `*_mean_correct` du CSV.
