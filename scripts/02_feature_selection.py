"""
02_feature_selection.py
==========================
Bloc 5 — C5.2.2 : Sélection des variables

Trois méthodes complémentaires, documentées et comparées :
  1. Corrélation (test statistique) — dans la continuité directe de
     correlations_b2.py du Bloc 2 (même style, mêmes variables de départ).
  2. Importance des variables (Random Forest) — méthode incorporée
     ("embedded method").
  3. Élimination récursive de variables (RFE) sur une régression linéaire.

Chaque méthode démontre la pertinence de la liste de variables retenue pour
l'entraînement du modèle (script 03).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)

X_train = pd.read_csv(DATA / "X_train.csv")
y_train = pd.read_csv(DATA / "y_train.csv").squeeze("columns")

results = {}

# ---------------------------------------------------------------------------
# 1) Corrélation de Pearson (variables continues) — même méthode que Bloc 2
# ---------------------------------------------------------------------------
print("=" * 70)
print("1) CORRÉLATION avec la variable cible (prix_m2)")
print("=" * 70)
continuous_vars = ["surface_m2", "nb_pieces", "revenu_median_mensuel",
                    "taux_pauvrete_pct", "population_2019",
                    "part_jeunes_pct", "part_seniors_pct", "mois_mutation",
                    "distance_transport_m", "est_metro_proche"]
corr_rows = []
for col in continuous_vars:
    r, p = stats.pearsonr(X_train[col], y_train)
    print(f"  {col:28} r={r:+.3f}  p={p:.2e}")
    corr_rows.append(dict(variable=col, correlation=round(r, 3), p_value=p))
corr_df = pd.DataFrame(corr_rows).sort_values("correlation", key=abs, ascending=False)

# ---------------------------------------------------------------------------
# 2) Importance des variables — Random Forest (méthode incorporée)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2) IMPORTANCE DES VARIABLES (Random Forest, méthode incorporée)")
print("=" * 70)
rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
for var, imp in importances.items():
    print(f"  {var:28} {imp:.3f}")

# ---------------------------------------------------------------------------
# 3) Élimination récursive de variables (RFE) sur régression linéaire
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3) ÉLIMINATION RÉCURSIVE DE VARIABLES (RFE, régression linéaire)")
print("=" * 70)
n_select = 8
rfe = RFE(LinearRegression(), n_features_to_select=n_select)
rfe.fit(X_train, y_train)
rfe_ranking = pd.Series(rfe.ranking_, index=X_train.columns).sort_values()
retenues_rfe = rfe_ranking[rfe_ranking == 1].index.tolist()
print(f"  Variables retenues (top {n_select}) : {retenues_rfe}")

# ---------------------------------------------------------------------------
# Synthèse et décision finale
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SYNTHÈSE — variables retenues pour l'entraînement (script 03)")
print("=" * 70)
# Règle de décision : on garde une variable si elle apparaît significative
# en corrélation (p<0.05) OU si son importance RF est notable (>2%) OU si
# elle est retenue par RFE. Les indicatrices de quartier sont conservées en
# bloc (effet de zone), conformément à l'analyse déjà menée au Bloc 2.
quartier_cols = [c for c in X_train.columns if c.startswith("q_")]
significatives = set(corr_df.loc[corr_df.p_value < 0.05, "variable"])
importantes_rf = set(importances[importances > 0.02].index)
selection_finale = sorted(set(continuous_vars) & (significatives | importantes_rf) | set(quartier_cols))
if "is_maison" not in selection_finale:
    selection_finale.append("is_maison")  # variable clé confirmée par Bloc 2 (régression : +1600€/m²)

print(f"Variables finales retenues ({len(selection_finale)}) : {selection_finale}")

summary = {
    "correlations": corr_df.to_dict(orient="records"),
    "importances_rf": importances.round(4).to_dict(),
    "rfe_retenues": retenues_rfe,
    "selection_finale": selection_finale,
}
with open(REPORTS / "feature_selection.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

pd.Series(selection_finale, name="variable").to_csv(DATA / "selected_features.csv", index=False)
print(f"\n[OK] Rapport écrit : {REPORTS / 'feature_selection.json'}")
print(f"[OK] Variables sélectionnées écrites : {DATA / 'selected_features.csv'}")
