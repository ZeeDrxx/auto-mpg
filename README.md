# Auto MPG — analyse descriptive et prédiction

**Auteur :** Zayd Bentalha — EMSI S9, SmartDev, Devoir 02

Application Streamlit sur le dataset Auto MPG (398 véhicules, 1970-1982, copie CSV du dépôt Seaborn).

- **Onglet « Analyse descriptive » (sans ML)** : filtres par origine, nombre de cylindres, année et mpg minimal ; efficacité, consommation, puissance et poids moyens ; relation poids / consommation avec droite de tendance ; consommation moyenne par année ; comparaison par origine (tableau, effectifs, boîtes à moustaches).
- **Onglet « Prédiction »** : un `RandomForestRegressor` (300 arbres) estime la consommation en L/100 km à partir des cylindres, de la cylindrée, de la puissance, du poids, de l'accélération, de l'année et de l'origine.

## Choix et préparation des données

- Conversions : poids en kg (× 0,45359237), consommation en L/100 km (235,215 / mpg), année = 1900 + `model_year`.
- Les 6 véhicules sans puissance (`horsepower`) sont ignorés dans les moyennes et exclus de l'apprentissage.
- Une couleur fixe par origine dans tous les graphiques.
- Évaluation du modèle : découpage 80 / 20 (`random_state=42`), puis réentraînement sur toutes les données pour les prédictions.

| Métrique (jeu de test, 79 véhicules) | Valeur |
|---|---|
| MAE | 0,83 L/100 km |
| RMSE | 1,15 L/100 km |
| R² | 0,907 |
| MAE en validation croisée (5 plis) | 1,13 L/100 km |

Un intervalle indicatif (percentiles 10-90 des prédictions des arbres) et les 5 véhicules réels les plus proches accompagnent chaque estimation.

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement

- Application : Streamlit Community Cloud (fichier principal `app.py`).
- Page de présentation : GitHub Pages (`index.html`), qui intègre l'application.
