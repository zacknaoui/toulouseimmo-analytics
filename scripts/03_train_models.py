"""
03_train_models.py
=====================
Bloc 5 — C5.2.3 : Entraînement d'un modèle d'apprentissage automatique

Entraîne et compare trois modèles de régression pour prédire prix_m2 :
  - Régression linéaire (baseline interprétable)
  - Random Forest Regressor (scikit-learn)
  - XGBoost Regressor (xgboost)

Librairies utilisées : scikit-learn, xgboost.
Le bon déroulement de l'entraînement et la performance de chaque modèle
sont tracés (temps d'entraînement, RMSE, MAE, R² sur train et test).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)

X_train = pd.read_csv(DATA / "X_train.csv")
X_test = pd.read_csv(DATA / "X_test.csv")
y_train = pd.read_csv(DATA / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(DATA / "y_test.csv").squeeze("columns")

selected = pd.read_csv(DATA / "selected_features.csv")["variable"].tolist()
X_train, X_test = X_train[selected], X_test[selected]

print(f"Entraînement sur {len(selected)} variables, {len(X_train)} lignes train / {len(X_test)} lignes test.\n")


def evaluate(name, model, X_tr, y_tr, X_te, y_te, train_time):
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    metrics = dict(
        modele=name,
        temps_entrainement_s=round(train_time, 3),
        rmse_train=round(float(np.sqrt(mean_squared_error(y_tr, pred_tr))), 1),
        rmse_test=round(float(np.sqrt(mean_squared_error(y_te, pred_te))), 1),
        mae_train=round(float(mean_absolute_error(y_tr, pred_tr)), 1),
        mae_test=round(float(mean_absolute_error(y_te, pred_te)), 1),
        r2_train=round(float(r2_score(y_tr, pred_tr)), 4),
        r2_test=round(float(r2_score(y_te, pred_te)), 4),
    )
    print(f"[{name}]")
    print(f"  Temps d'entraînement : {metrics['temps_entrainement_s']}s")
    print(f"  RMSE  train={metrics['rmse_train']:>7.1f}€  test={metrics['rmse_test']:>7.1f}€")
    print(f"  MAE   train={metrics['mae_train']:>7.1f}€  test={metrics['mae_test']:>7.1f}€")
    print(f"  R²    train={metrics['r2_train']:>7.4f}  test={metrics['r2_test']:>7.4f}\n")
    return metrics


results = []

# 1) Régression linéaire (baseline) — standardisée pour des coefficients comparables
scaler = StandardScaler().fit(X_train)
X_train_sc = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

t0 = time.time()
lr = LinearRegression().fit(X_train_sc, y_train)
results.append(evaluate("Régression linéaire", lr, X_train_sc, y_train, X_test_sc, y_test, time.time() - t0))

# 2) Random Forest
t0 = time.time()
rf = RandomForestRegressor(n_estimators=400, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results.append(evaluate("Random Forest", rf, X_train, y_train, X_test, y_test, time.time() - t0))

# 3) XGBoost
t0 = time.time()
xgb = XGBRegressor(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
)
xgb.fit(X_train, y_train)
results.append(evaluate("XGBoost", xgb, X_train, y_train, X_test, y_test, time.time() - t0))

results_df = pd.DataFrame(results).sort_values("rmse_test")
print("=" * 70)
print("COMPARAISON DES MODÈLES (triés par RMSE test croissant)")
print("=" * 70)
print(results_df.to_string(index=False))

meilleur = results_df.iloc[0]["modele"]
print(f"\n[OK] Meilleur modèle sur cette itération : {meilleur}")

results_df.to_csv(REPORTS / "comparaison_modeles.csv", index=False)
with open(REPORTS / "comparaison_modeles.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"[OK] Rapport écrit : {REPORTS / 'comparaison_modeles.csv'}")
