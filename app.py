# ==========================================================
# DEVOIR 02 - Exploration et prédiction : données Auto MPG
# Auteur : Zayd Bentalha
#
# Onglet 1 : analyse descriptive (sans modèle ML)
# Onglet 2 : prédiction de la consommation (Random Forest)
#
# Lancement : streamlit run app.py
# ==========================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(page_title="Auto MPG", page_icon="🚗", layout="wide")

CHEMIN_CSV = Path(__file__).parent / "mpg.csv"

# Conversions : 1 lb = 0.45359237 kg ; L/100 km = 235.215 / mpg (gallon US).
LB_EN_KG = 0.45359237
MPG_EN_L100 = 235.215

# Une couleur fixe par origine, identique dans tous les graphiques.
COULEURS = {"usa": "#2a6fdb", "japan": "#8cc5ff", "europe": "#e8463a"}
ORDRE_ORIGINES = ["usa", "japan", "europe"]

LIBELLES = {
    "mpg": "Efficacité (mpg)",
    "consumption_l_100km": "Consommation (L/100 km)",
    "weight_kg": "Poids (kg)",
    "horsepower": "Puissance (hp)",
    "displacement": "Cylindrée (pouces³)",
    "acceleration": "Accélération 0-60 mph (s)",
    "cylinders": "Cylindres",
    "year": "Année du modèle",
    "origin": "Origine",
}

COLONNES_AFFICHEES = [
    "name", "origin", "year", "cylinders", "mpg", "consumption_l_100km",
    "horsepower", "weight_kg", "acceleration",
]


# ----------------------------------------------------------
# Chargement et préparation des données
# ----------------------------------------------------------

@st.cache_data
def charger_donnees():
    df = pd.read_csv(CHEMIN_CSV)
    df["year"] = 1900 + df["model_year"]
    df["weight_kg"] = (df["weight"] * LB_EN_KG).round(2)
    df["consumption_l_100km"] = (MPG_EN_L100 / df["mpg"]).round(2)
    return df


df = charger_donnees()


# ----------------------------------------------------------
# Filtres (barre latérale)
# ----------------------------------------------------------

st.sidebar.header("Filtres")

origine = st.sidebar.selectbox("Origine", ["Toutes"] + ORDRE_ORIGINES)

cylindres_dispo = sorted(df["cylinders"].unique())
cylindres = st.sidebar.multiselect(
    "Nombre de cylindres", cylindres_dispo, default=cylindres_dispo
)

annee_min, annee_max = int(df["year"].min()), int(df["year"].max())
annees = st.sidebar.slider(
    "Année du modèle", annee_min, annee_max, (annee_min, annee_max)
)

mpg_min = st.sidebar.slider(
    "Efficacité minimale (mpg)",
    float(df["mpg"].min()), float(df["mpg"].max()), float(df["mpg"].min()),
    step=0.5,
)

st.sidebar.caption("Les filtres s'appliquent à l'onglet « Analyse descriptive ».")

masque = (
    df["cylinders"].isin(cylindres)
    & df["year"].between(*annees)
    & (df["mpg"] >= mpg_min)
)
if origine != "Toutes":
    masque &= df["origin"] == origine
filtre = df[masque]


# ----------------------------------------------------------
# En-tête
# ----------------------------------------------------------

st.title("🚗 Exploration des données automobiles")
st.caption(
    "Dataset historique Auto MPG (398 véhicules, 1970-1982) · "
    "copie CSV publique du dépôt Seaborn"
)

onglet_analyse, onglet_prediction = st.tabs(
    ["📊 Analyse descriptive", "🤖 Prédiction (Random Forest)"]
)


# ==========================================================
# ONGLET 1 : ANALYSE DESCRIPTIVE (sans modèle ML)
# ==========================================================

with onglet_analyse:
    st.info(
        "mpg élevé = meilleure efficacité. L/100 km faible = consommation "
        "moindre. Cet onglet est une analyse descriptive, sans modèle ML."
    )

    if filtre.empty:
        st.warning("Aucun véhicule ne correspond aux filtres choisis.")
    else:
        # --- 1. Indicateurs et données filtrées ---
        st.header("1 — Indicateurs et données filtrées")
        st.write(f"**{len(filtre)}** observations sur {len(df)}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Efficacité moyenne", f"{filtre['mpg'].mean():.1f} mpg")
        c2.metric(
            "Consommation moyenne",
            f"{filtre['consumption_l_100km'].mean():.1f} L/100 km",
        )
        c3.metric("Puissance moyenne", f"{filtre['horsepower'].mean():.1f} hp")
        c4.metric("Poids moyen", f"{filtre['weight_kg'].mean():.0f} kg")

        nb_hp_manquants = int(filtre["horsepower"].isna().sum())
        st.caption(
            "Les moyennes ignorent les valeurs manquantes "
            f"({nb_hp_manquants} puissance(s) manquante(s) dans la sélection). "
            "La consommation moyenne est la moyenne des conversions "
            "individuelles, pas la conversion du mpg moyen."
        )

        st.dataframe(
            filtre[COLONNES_AFFICHEES].reset_index(drop=True),
            height=320,
        )

        # --- 2. Choisir un véhicule ---
        st.header("2 — Choisir un véhicule")
        options = {
            # ligne.name serait l'index de la ligne : on lit la colonne.
            f"{ligne['name']} — {ligne['year']} (ligne {index + 1})": index
            for index, ligne in filtre.iterrows()
        }
        choix = st.selectbox("Véhicule", list(options))
        vehicule = df.loc[[options[choix]], COLONNES_AFFICHEES]
        st.dataframe(vehicule.reset_index(drop=True))

        ecart = (
            vehicule["consumption_l_100km"].iloc[0]
            - filtre["consumption_l_100km"].mean()
        )
        st.write(
            f"Consommation {'supérieure' if ecart > 0 else 'inférieure'} de "
            f"**{abs(ecart):.1f} L/100 km** à la moyenne de la sélection."
        )

        # --- 3. Graphiques ---
        st.header("3 — Comparaisons graphiques")

        st.subheader("Relation entre poids et consommation")
        fig = px.scatter(
            filtre, x="weight_kg", y="consumption_l_100km", color="origin",
            color_discrete_map=COULEURS,
            category_orders={"origin": ORDRE_ORIGINES},
            hover_name="name",
            hover_data=["year", "cylinders", "horsepower", "mpg"],
            labels=LIBELLES,
        )
        if len(filtre) >= 2:
            # Droite de tendance (moindres carrés) sur toute la sélection.
            pente, ordonnee = np.polyfit(
                filtre["weight_kg"], filtre["consumption_l_100km"], 1
            )
            x_ligne = np.array([filtre["weight_kg"].min(), filtre["weight_kg"].max()])
            fig.add_scatter(
                x=x_ligne, y=pente * x_ligne + ordonnee, mode="lines",
                name="Tendance", line=dict(color="#555", dash="dash"),
            )
        st.plotly_chart(fig, width="stretch")

        if len(filtre) >= 2:
            correlation = filtre["weight_kg"].corr(filtre["consumption_l_100km"])
            st.caption(
                f"Corrélation de Pearson poids / consommation : **{correlation:.2f}**. "
                f"En moyenne, +100 kg ≈ +{pente * 100:.2f} L/100 km."
            )

        st.subheader("Consommation moyenne par année de modèle")
        par_annee = (
            filtre.groupby(["year", "origin"], as_index=False)["consumption_l_100km"]
            .mean()
        )
        fig = px.line(
            par_annee, x="year", y="consumption_l_100km", color="origin",
            markers=True, color_discrete_map=COULEURS,
            category_orders={"origin": ORDRE_ORIGINES}, labels=LIBELLES,
        )
        st.plotly_chart(fig, width="stretch")

        # --- 4. Comparaison par origine ---
        st.header("4 — Comparer les véhicules par origine")

        resume = (
            filtre.groupby("origin")
            .agg(
                nombre=("name", "size"),
                mpg_moyen=("mpg", "mean"),
                consommation_moyenne=("consumption_l_100km", "mean"),
                puissance_moyenne=("horsepower", "mean"),
                poids_moyen_kg=("weight_kg", "mean"),
                cylindres_moyens=("cylinders", "mean"),
            )
            .reindex([o for o in ORDRE_ORIGINES if o in filtre["origin"].unique()])
            .round(1)
        )
        st.dataframe(resume)

        g1, g2 = st.columns(2)
        with g1:
            fig = px.bar(
                resume.reset_index(), x="origin", y="nombre", color="origin",
                color_discrete_map=COULEURS, labels={**LIBELLES, "nombre": "Nombre"},
                title="Nombre d'observations par origine",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
        with g2:
            fig = px.box(
                filtre, x="origin", y="consumption_l_100km", color="origin",
                color_discrete_map=COULEURS,
                category_orders={"origin": ORDRE_ORIGINES},
                labels=LIBELLES, points="outliers",
                title="Distribution de la consommation par origine",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

        st.caption(
            "Attention : la comparaison par origine mélange des véhicules de "
            "tailles différentes (les véhicules américains sont en moyenne plus "
            "lourds et ont plus de cylindres) : l'origine n'est pas la seule "
            "cause des écarts observés."
        )


# ==========================================================
# ONGLET 2 : PRÉDICTION (Random Forest Regressor)
# ==========================================================

VARIABLES_NUM = [
    "cylinders", "displacement", "horsepower", "weight_kg", "acceleration", "year",
]
VARIABLES = VARIABLES_NUM + ["origin"]
CIBLE = "consumption_l_100km"


def creer_modele():
    preparation = ColumnTransformer(
        [("origine", OneHotEncoder(handle_unknown="ignore"), ["origin"])],
        remainder="passthrough",
    )
    return Pipeline([
        ("preparation", preparation),
        ("foret", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1,
        )),
    ])


@st.cache_resource
def entrainer_modele():
    # Les 6 véhicules sans puissance sont exclus de l'apprentissage.
    donnees = df.dropna(subset=VARIABLES + [CIBLE])
    X, y = donnees[VARIABLES], donnees[CIBLE]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    modele = creer_modele().fit(X_train, y_train)
    prediction = modele.predict(X_test)

    metriques = {
        "MAE": mean_absolute_error(y_test, prediction),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
        "R2": r2_score(y_test, prediction),
        # Référence naïve : prédire toujours la moyenne d'entraînement.
        "MAE_reference": mean_absolute_error(
            y_test, np.full(len(y_test), y_train.mean())
        ),
        "MAE_validation_croisee": -cross_val_score(
            creer_modele(), X, y, cv=5, scoring="neg_mean_absolute_error"
        ).mean(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_exclus": len(df) - len(donnees),
    }

    noms = modele.named_steps["preparation"].get_feature_names_out()
    importances = (
        pd.Series(modele.named_steps["foret"].feature_importances_, index=noms)
        .rename(lambda nom: "origin" if nom.startswith("origine__") else nom.split("__")[-1])
        .groupby(level=0).sum()
        .sort_values()
    )

    test = X_test.assign(reel=y_test.values, predit=prediction)

    # Modèle final réentraîné sur toutes les données pour les prédictions.
    modele_final = creer_modele().fit(X, y)
    return modele_final, metriques, importances, test, donnees


with onglet_prediction:
    st.info(
        "Saisissez les caractéristiques d'un véhicule : un Random Forest "
        "Regressor entraîné sur le dataset estime sa consommation (L/100 km). "
        "Le modèle utilise toutes les données, indépendamment des filtres."
    )

    with st.spinner("Entraînement du modèle..."):
        modele, metriques, importances, test, donnees_modele = entrainer_modele()

    st.header("1 — Caractéristiques du véhicule")

    def bornes(colonne):
        serie = donnees_modele[colonne]
        return float(serie.min()), float(serie.max()), float(serie.median())

    with st.form("formulaire_prediction"):
        f1, f2, f3 = st.columns(3)
        with f1:
            origine_saisie = st.selectbox("Origine", ORDRE_ORIGINES)
            cylindres_saisis = st.selectbox(
                "Nombre de cylindres", cylindres_dispo,
                index=cylindres_dispo.index(4),
            )
            annee_saisie = st.slider(
                "Année du modèle", annee_min, annee_max, 1976
            )
        with f2:
            mini, maxi, med = bornes("weight_kg")
            poids_saisi = st.slider("Poids (kg)", mini, maxi, med, step=10.0)
            mini, maxi, med = bornes("horsepower")
            puissance_saisie = st.slider("Puissance (hp)", mini, maxi, med, step=1.0)
        with f3:
            mini, maxi, med = bornes("displacement")
            cylindree_saisie = st.slider(
                "Cylindrée (pouces³)", mini, maxi, med, step=1.0
            )
            mini, maxi, med = bornes("acceleration")
            acceleration_saisie = st.slider(
                "Accélération 0-60 mph (s)", mini, maxi, med, step=0.1
            )
        envoye = st.form_submit_button("Estimer la consommation", type="primary")

    saisie = pd.DataFrame([{
        "cylinders": cylindres_saisis,
        "displacement": cylindree_saisie,
        "horsepower": puissance_saisie,
        "weight_kg": poids_saisi,
        "acceleration": acceleration_saisie,
        "year": annee_saisie,
        "origin": origine_saisie,
    }])

    st.header("2 — Consommation estimée")

    consommation = float(modele.predict(saisie)[0])

    # Dispersion des prédictions des arbres : indication d'incertitude.
    X_saisie = modele.named_steps["preparation"].transform(saisie)
    par_arbre = np.array([
        arbre.predict(X_saisie)[0]
        for arbre in modele.named_steps["foret"].estimators_
    ])
    bas, haut = np.percentile(par_arbre, [10, 90])

    r1, r2, r3 = st.columns(3)
    r1.metric("Consommation estimée", f"{consommation:.1f} L/100 km")
    r2.metric("Équivalent", f"{MPG_EN_L100 / consommation:.1f} mpg")
    r3.metric("Fourchette des arbres (10-90 %)", f"{bas:.1f} – {haut:.1f} L/100 km")

    moyenne_globale = donnees_modele[CIBLE].mean()
    st.write(
        f"Soit **{abs(consommation - moyenne_globale):.1f} L/100 km "
        f"{'de plus' if consommation > moyenne_globale else 'de moins'}** "
        f"que la moyenne du dataset ({moyenne_globale:.1f} L/100 km)."
    )

    # Véhicules réels les plus proches (variables numériques standardisées).
    numeriques = donnees_modele[VARIABLES_NUM]
    ecarts = ((numeriques - saisie[VARIABLES_NUM].iloc[0]) / numeriques.std()) ** 2
    proches = donnees_modele.assign(distance=np.sqrt(ecarts.sum(axis=1)))
    proches = proches[proches["origin"] == origine_saisie].nsmallest(5, "distance")
    st.write("Véhicules réels les plus proches (même origine) :")
    st.dataframe(proches[COLONNES_AFFICHEES].reset_index(drop=True))

    if not envoye:
        st.caption("Modifiez les valeurs puis cliquez sur « Estimer la consommation ».")

    # --- Qualité du modèle ---
    st.header("3 — Qualité du modèle")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE (test)", f"{metriques['MAE']:.2f} L/100 km")
    m2.metric("RMSE (test)", f"{metriques['RMSE']:.2f} L/100 km")
    m3.metric("R² (test)", f"{metriques['R2']:.3f}")
    m4.metric("MAE validation croisée (5 plis)", f"{metriques['MAE_validation_croisee']:.2f}")

    st.caption(
        f"Apprentissage : {metriques['n_train']} véhicules, test : "
        f"{metriques['n_test']} véhicules (20 %, random_state = 42). "
        f"{metriques['n_exclus']} véhicules sans puissance exclus. "
        f"Référence naïve (toujours prédire la moyenne) : MAE = "
        f"{metriques['MAE_reference']:.2f} L/100 km."
    )

    q1, q2 = st.columns(2)
    with q1:
        fig = px.scatter(
            test, x="reel", y="predit", color="origin",
            color_discrete_map=COULEURS,
            category_orders={"origin": ORDRE_ORIGINES},
            labels={"reel": "Consommation réelle (L/100 km)",
                    "predit": "Consommation prédite (L/100 km)",
                    "origin": "Origine"},
            title="Réel vs prédit (jeu de test)",
        )
        limites = [test[["reel", "predit"]].min().min(), test[["reel", "predit"]].max().max()]
        fig.add_scatter(
            x=limites, y=limites, mode="lines", name="Prédiction parfaite",
            line=dict(color="#555", dash="dash"),
        )
        st.plotly_chart(fig, width="stretch")
    with q2:
        fig = px.bar(
            x=importances.values,
            y=[LIBELLES.get(nom, nom) for nom in importances.index],
            orientation="h",
            labels={"x": "Importance", "y": ""},
            title="Importance des variables",
        )
        st.plotly_chart(fig, width="stretch")
