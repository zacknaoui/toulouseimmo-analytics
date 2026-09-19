"""
04_optimize_hyperparameters.py
=================================
Bloc 5 — C5.2.4 : Optimisation de la performance des modèles

Leviers d'optimisation utilisés :
  - Hyperparamètres (RandomizedSearchCV, validation croisée à 5 plis)
  - Infrastructure de calcul (n_jobs=-1, parallélisation de la recherche)
Moyens utilisés : scikit-learn (RandomizedSearchCV), xgboost.

Le script 03 a montré un fort sur-apprentissage des modèles à base d'arbres
(R² train ≈0.58-0.60 contre R² test ≈0.25-0.26) : ce script recherche donc
en priorité des hyperparamètres qui RÉDUISENT ce sur-apprentissage
(profondeur, régularisation) plutôt que de complexifier encore le modèle.

Sortie : comparaison avant/après optimisation + analyse des résidus du
meilleur modèle final, et sélection du modèle à sauvegarder (script 05).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
REPORTS = BASE / "reports"
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATA / "X_train.csv")
X_test = pd.read_csv(DATA / "X_test.csv")
y_train = pd.read_csv(DATA / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(DATA / "y_test.csv").squeeze("columns")
selected = pd.read_csv(DATA / "selected_features.csv")["variable"].tolist()
X_train, X_test = X_train[selected], X_test[selected]

avant = pd.read_csv(REPORTS / "comparaison_modeles.csv").set_index("modele")

cv = KFold(n_splits=5, shuffle=True, random_state=42)


def metrics_of(model, X_tr, y_tr, X_te, y_te):
    p_tr, p_te = model.predict(X_tr), model.predict(X_te)
    return dict(
        rmse_train=round(float(np.sqrt(mean_squared_error(y_tr, p_tr))), 1),
        rmse_test=round(float(np.sqrt(mean_squared_error(y_te, p_te))), 1),
        mae_train=round(float(mean_absolute_error(y_tr, p_tr)), 1),
        mae_test=round(float(mean_absolute_error(y_te, p_te)), 1),
        r2_train=round(float(r2_score(y_tr, p_tr)), 4),
        r2_test=round(float(r2_score(y_te, p_te)), 4),
    )


resultats_opt = {}

# ---------------------------------------------------------------------------
# Random Forest — recherche orientée régularisation (limiter la profondeur,
# augmenter min_samples_leaf) pour combattre le sur-apprentissage constaté.
# ---------------------------------------------------------------------------
print("=" * 70)
print("OPTIMISATION — Random Forest (RandomizedSearchCV, 5-fold CV)")
print("=" * 70)
grille_rf = dict(
    n_estimators=randint(200, 600),
    max_depth=randint(3, 10),
    min_samples_leaf=randint(5, 40),
    max_features=uniform(0.4, 0.6),
)
t0 = time.time()
search_rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=grille_rf, n_iter=25, cv=cv,
    scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1,
)
search_rf.fit(X_train, y_train)
print(f"  Meilleurs hyperparamètres : {search_rf.best_params_}")
print(f"  Temps de recherche : {time.time() - t0:.1f}s")
rf_opt = search_rf.best_estimator_
m = metrics_of(rf_opt, X_train, y_train, X_test, y_test)
resultats_opt["Random Forest (optimisé)"] = m
print(f"  RMSE test avant={avant.loc['Random Forest','rmse_test']:.0f}€ -> après={m['rmse_test']:.0f}€")
print(f"  R² test   avant={avant.loc['Random Forest','r2_test']:.4f} -> après={m['r2_test']:.4f}\n")

# ---------------------------------------------------------------------------
# XGBoost — recherche sur profondeur, learning rate, régularisation L1/L2
# ---------------------------------------------------------------------------
print("=" * 70)
print("OPTIMISATION — XGBoost (RandomizedSearchCV, 5-fold CV)")
print("=" * 70)
grille_xgb = dict(
    n_estimators=randint(150, 500),
    max_depth=randint(2, 6),
    learning_rate=uniform(0.01, 0.15),
    subsample=uniform(0.6, 0.4),
    colsample_bytree=uniform(0.5, 0.5),
    reg_alpha=uniform(0, 2),
    reg_lambda=uniform(0.5, 3),
)
t0 = time.time()
search_xgb = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1),
    param_distributions=grille_xgb, n_iter=30, cv=cv,
    scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1,
)
search_xgb.fit(X_train, y_train)
print(f"  Meilleurs hyperparamètres : {search_xgb.best_params_}")
print(f"  Temps de recherche : {time.time() - t0:.1f}s")
xgb_opt = search_xgb.best_estimator_
m = metrics_of(xgb_opt, X_train, y_train, X_test, y_test)
resultats_opt["XGBoost (optimisé)"] = m
print(f"  RMSE test avant={avant.loc['XGBoost','rmse_test']:.0f}€ -> après={m['rmse_test']:.0f}€")
print(f"  R² test   avant={avant.loc['XGBoost','r2_test']:.4f} -> après={m['r2_test']:.4f}\n")

# Régression linéaire (référence, rappel du script 03 — pas d'hyperparamètre à optimiser)
scaler = StandardScaler().fit(X_train)
lr = LinearRegression().fit(scaler.transform(X_train), y_train)
m_lr = metrics_of(lr, scaler.transform(X_train), y_train, scaler.transform(X_test), y_test)
resultats_opt["Régression linéaire (référence)"] = m_lr

# ---------------------------------------------------------------------------
# Comparaison finale + sélection du modèle
# ---------------------------------------------------------------------------
comp = pd.DataFrame(resultats_opt).T.sort_values("rmse_test")
print("=" * 70)
print("COMPARAISON FINALE APRÈS OPTIMISATION")
print("=" * 70)
print(comp.to_string())

meilleur_nom = comp.index[0]
print(f"\n[OK] Modèle final retenu : {meilleur_nom}")

modeles = {"Random Forest (optimisé)": rf_opt, "XGBoost (optimisé)": xgb_opt, "Régression linéaire (référence)": lr}
meilleur_modele = modeles[meilleur_nom]

# --- Analyse des prédictions : résidus --------------------------------------
X_test_pred = scaler.transform(X_test) if meilleur_nom.startswith("Régression") else X_test
pred_test = meilleur_modele.predict(X_test_pred)
residus = y_test.values - pred_test

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].scatter(pred_test, y_test, alpha=0.25, s=12, color="#2f5f8a")
lims = [min(pred_test.min(), y_test.min()), max(pred_test.max(), y_test.max())]
axes[0].plot(lims, lims, "r--", linewidth=1)
axes[0].set_xlabel("Prix au m² prédit (€)")
axes[0].set_ylabel("Prix au m² réel (€)")
axes[0].set_title(f"Prédit vs réel — {meilleur_nom}")

axes[1].hist(residus, bins=40, color="#8a5a2f")
axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
axes[1].set_xlabel("Résidu (réel - prédit, €)")
axes[1].set_title("Distribution des résidus (jeu de test)")
plt.tight_layout()
plt.savefig(FIG / "residus_modele_final.png", dpi=130)
print(f"[OK] Figure résidus écrite : {FIG / 'residus_modele_final.png'}")

resume = {
    "comparaison_avant": avant.reset_index().to_dict(orient="records"),
    "comparaison_apres": comp.reset_index().rename(columns={"index": "modele"}).to_dict(orient="records"),
    "meilleurs_hyperparametres_rf": search_rf.best_params_,
    "meilleurs_hyperparametres_xgb": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in search_xgb.best_params_.items()},
    "modele_final_retenu": meilleur_nom,
    "residus_stats": dict(
        moyenne=round(float(residus.mean()), 1),
        ecart_type=round(float(residus.std()), 1),
        mediane_abs=round(float(np.median(np.abs(residus))), 1),
    ),
}
with open(REPORTS / "optimisation.json", "w") as f:
    json.dump(resume, f, indent=2, ensure_ascii=False)
print(f"[OK] Rapport écrit : {REPORTS / 'optimisation.json'}")

# on ré-exporte les objets nécessaires pour le script de sauvegarde (05)
import pickle
with open(DATA / "_meilleur_modele_tmp.pkl", "wb") as f:
    pickle.dump(dict(nom=meilleur_nom, modele=meilleur_modele,
                      scaler=scaler if meilleur_nom.startswith("Régression") else None,
                      features=selected), f)
