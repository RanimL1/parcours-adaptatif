# Parcours d'apprentissage adaptatif — recommandation hybride

Données : `interactions_real_rich_processed.csv` (une ligne par couple étudiant-compétence) à placer dans `data/raw/`.

## Installation
    pip install -r requirements.txt

## Utilisation (depuis la racine du projet)
    python run.py                         # pipeline complet : Parquet, graphe, résultats comparatifs, profils A/B/C
    jupyter notebook notebooks/           # 01 -> 06
    uvicorn api.main:app --reload         # GET /recommend/{student_id}, /positioning-test, POST /recommend
    streamlit run app/streamlit_app.py    # démo

## Arborescence
    data/raw/        CSV d'origine
    data/processed/  interactions.parquet (généré)
    data/knowledge/  competences.csv (id, nom, niveau), prerequis.csv (generés par le graphe appris)
    notebooks/       01_exploration ... 06_evaluation
    src/             data_loader, knowledge_graph, knowledge_filter, student_model,
                     content_based, collaborative, hybrid, evaluation, pipeline
    api/main.py      FastAPI
    app/             Streamlit

## Limites à mentionner dans le rapport
- Pas de timestamp : découpage aléatoire par étudiant (pas temporel).
- Compétences anonymisées : prérequis APPRIS par corrélation (pas validés pédagogiquement) ; `nom` = « Compétence id ».
- Niveau et difficulté recalculés sur le train pour éviter la fuite des colonnes *_mean_correct du CSV.
