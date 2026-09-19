"""
06_monitoring.py
===================
Bloc 5 — C5.3.3 : Supervision du système Machine Learning

Deux volets de supervision, tous deux nécessaires pour détecter à la fois une
dérive de performance et une dérive des données en entrée :

  1. SUIVI DE PERFORMANCE : sur un flux de nouvelles transactions dont on
     connaît a posteriori le prix réel, on calcule la MAE glissante (fenêtre
     de N transactions) et on la compare à la MAE de référence obtenue au
     moment de l'entraînement (reports/optimisation.json). Une alerte est
     levée si la MAE glissante dépasse un seuil (ex : +25% de la référence).

  2. DÉRIVE DES DONNÉES (data drift) : on compare la distribution des
     variables d'entrée du flux récent à celle du jeu d'entraînement
     (test de Kolmogorov-Smirnov par variable). Une dérive significative
     indique que le marché a changé et que le modèle doit être réentraîné
     (cf. script 07_data_collection_retrain.py).

Outils : pandas, scipy (test statistique), fichier de log JSON append-only
(journal d'alertes exploitable par un outil de supervision externe type
Grafana/Prometheus en production).
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODELS = BASE / "models"
REPORTS = BASE / "reports"
MONITORING_LOG = REPORTS / "monitoring_log.jsonl"

SEUIL_DEGRADATION_MAE = 1.25      # alerte si MAE glissante > 125% de la MAE de référence
SEUIL_DRIFT_PVALUE = 0.01          # alerte si p-value KS < 1% sur une variable
FENETRE_GLISSANTE = 200            # taille de la fenêtre de calcul de la MAE glissante


def charger_modele():
    with open(MODELS / "latest.json") as f:
        pointeur = json.load(f)
    version_dir = MODELS / pointeur["version"]
    with open(version_dir / "metadata.json") as f:
        meta = json.load(f)
    modele = joblib.load(version_dir / "model.joblib")
    scaler_path = version_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return modele, scaler, meta


def simuler_flux_recent(X_ref: pd.DataFrame, y_ref: pd.Series, n=300, derive=False):
    """
    Simule un petit flux de transactions "récentes" pour la démonstration du
    monitoring. `derive=True` introduit volontairement une dérive (hausse de
    prix non captée par le modèle) pour vérifier que les alertes se déclenchent
    bien — étape indispensable pour valider qu'un système de supervision
    fonctionne réellement, et pas seulement qu'il tourne sans erreur.
    """
    idx = np.random.default_rng(123).choice(len(X_ref), size=n, replace=True)
    X_flux = X_ref.iloc[idx].reset_index(drop=True).copy()
    y_flux = y_ref.iloc[idx].reset_index(drop=True).copy()
    if derive:
        # dérive de marché simulée : +50% de hausse générale des prix, non
        # apprise par le modèle entraîné sur les données passées
        y_flux = y_flux * 1.50
        X_flux["surface_m2"] = X_flux["surface_m2"] * np.random.default_rng(7).uniform(0.85, 1.15, n)
    return X_flux, y_flux


def calculer_mae_glissante(y_true, y_pred, fenetre):
    erreurs_abs = np.abs(np.array(y_true) - np.array(y_pred))
    return pd.Series(erreurs_abs).rolling(fenetre, min_periods=30).mean()


def detecter_drift(X_ref: pd.DataFrame, X_flux: pd.DataFrame) -> dict:
    alertes = {}
    variables_continues = ["surface_m2", "nb_pieces", "revenu_median_mensuel", "mois_mutation"]
    for col in variables_continues:
        stat, p = stats.ks_2samp(X_ref[col], X_flux[col])
        alertes[col] = dict(ks_stat=round(float(stat), 4), p_value=float(p),
                             derive_detectee=bool(p < SEUIL_DRIFT_PVALUE))
    return alertes


def main(simuler_derive: bool = True):
    modele, scaler, meta = charger_modele()
    features = meta["features_attendues"]
    mae_reference = meta["metriques_test"]["mae_test"]

    X_test = pd.read_csv(DATA / "X_test.csv")[features]
    y_test = pd.read_csv(DATA / "y_test.csv").squeeze("columns")

    print(f"Modèle chargé : {meta['modele']} (version {meta['version']})")
    print(f"MAE de référence (évaluation à l'entraînement) : {mae_reference}€\n")

    X_flux, y_flux = simuler_flux_recent(X_test, y_test, n=300, derive=simuler_derive)

    X_pred = pd.DataFrame(scaler.transform(X_flux), columns=features) if scaler is not None else X_flux
    y_pred = modele.predict(X_pred)

    mae_glissante = calculer_mae_glissante(y_flux, y_pred, FENETRE_GLISSANTE)
    mae_actuelle = float(mae_glissante.dropna().iloc[-1]) if mae_glissante.notna().any() else float(np.mean(np.abs(y_flux - y_pred)))
    ratio = mae_actuelle / mae_reference

    alerte_performance = ratio > SEUIL_DEGRADATION_MAE
    print("--- 1) Suivi de performance (MAE glissante) ---")
    print(f"MAE glissante (dernière fenêtre de {FENETRE_GLISSANTE}) : {mae_actuelle:.0f}€")
    print(f"Ratio vs référence : {ratio:.2f}x  "
          f"{'>>> ALERTE : dégradation de performance <<<' if alerte_performance else '(OK)'}")

    print("\n--- 2) Dérive des données (test de Kolmogorov-Smirnov) ---")
    drift = detecter_drift(X_test, X_flux)
    alerte_drift = False
    for var, r in drift.items():
        statut = "DÉRIVE DÉTECTÉE" if r["derive_detectee"] else "stable"
        print(f"  {var:24} KS={r['ks_stat']:.3f}  p={r['p_value']:.2e}  -> {statut}")
        alerte_drift = alerte_drift or r["derive_detectee"]

    entree_log = {
        "horodatage": datetime.now().isoformat(timespec="seconds"),
        "modele_version": meta["version"],
        "mae_reference": mae_reference,
        "mae_glissante": round(mae_actuelle, 1),
        "ratio_degradation": round(ratio, 3),
        "alerte_performance": alerte_performance,
        "drift": drift,
        "alerte_drift": alerte_drift,
    }
    REPORTS.mkdir(exist_ok=True)
    with open(MONITORING_LOG, "a") as f:
        f.write(json.dumps(entree_log, ensure_ascii=False) + "\n")

    print(f"\n[OK] Entrée de supervision journalisée : {MONITORING_LOG}")
    if alerte_performance or alerte_drift:
        print("\n>>> ACTION RECOMMANDÉE : déclencher le réentraînement du modèle")
        print(">>> (voir script 07_data_collection_retrain.py)")
    return entree_log


if __name__ == "__main__":
    main(simuler_derive=True)
