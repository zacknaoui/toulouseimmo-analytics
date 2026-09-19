"""
07_data_collection_retrain.py
================================
Bloc 5 — C5.3.4 : Automatisation du cycle de vie du système ML

Système de collecte de données qui permet :
  - d'historiser les prédictions faites par l'API (table `predictions_log`)
  - de collecter de nouvelles données d'apprentissage (nouvelles transactions
    DVF disponibles au fil du temps, table `nouvelles_transactions`)
  - de déclencher un réentraînement du modèle quand les conditions sont
    réunies (volume suffisant de nouvelles données OU alerte de monitoring)

Le tout est journalisé dans une base SQLite dédiée (data/monitoring.db),
séparée de la base transactionnelle métier (toulouse_immo.db) pour ne pas
mélanger la donnée de référence et les logs opérationnels.

Usage :
    python3 scripts/07_data_collection_retrain.py --log-prediction ...
    python3 scripts/07_data_collection_retrain.py --check-retrain
"""

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MONITORING_DB = DATA / "monitoring.db"
REPORTS = BASE / "reports"

SEUIL_NOUVELLES_TRANSACTIONS = 500   # déclenche un réentraînement si atteint
SEUIL_ALERTE_MONITORING = True       # déclenche aussi si le script 06 a levé une alerte

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage      TEXT NOT NULL,
    modele_version  TEXT NOT NULL,
    quartier        TEXT,
    type_bien       TEXT,
    surface_m2      REAL,
    prix_m2_predit  REAL,
    prix_m2_reel    REAL          -- rempli a posteriori quand la vraie vente est connue
);

CREATE TABLE IF NOT EXISTS nouvelles_transactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage_ajout TEXT NOT NULL,
    source          TEXT NOT NULL,   -- ex: 'DVF_2025_T1', 'saisie_manuelle'
    payload_json    TEXT NOT NULL,   -- ligne de transaction complète (mêmes colonnes que `transactions`)
    integree        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reentrainements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage      TEXT NOT NULL,
    declencheur     TEXT NOT NULL,   -- 'volume' | 'alerte_monitoring' | 'manuel'
    n_nouvelles_donnees INTEGER,
    modele_version_avant TEXT,
    modele_version_apres TEXT,
    statut          TEXT             -- 'succes' | 'echec'
);
"""


def get_conn():
    DATA.mkdir(exist_ok=True)
    conn = sqlite3.connect(MONITORING_DB)
    conn.executescript(SCHEMA)
    return conn


def log_prediction(quartier, type_bien, surface_m2, modele_version, prix_m2_predit, prix_m2_reel=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO predictions_log (horodatage, modele_version, quartier, type_bien, "
        "surface_m2, prix_m2_predit, prix_m2_reel) VALUES (?,?,?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), modele_version, quartier, type_bien,
         surface_m2, prix_m2_predit, prix_m2_reel),
    )
    conn.commit()
    conn.close()


def collecter_nouvelle_transaction(payload: dict, source: str = "flux_dvf"):
    """Historise une nouvelle transaction en attente d'intégration au jeu
    d'entraînement (ex : nouveau millésime DVF publié trimestriellement)."""
    conn = get_conn()
    conn.execute(
        "INSERT INTO nouvelles_transactions (horodatage_ajout, source, payload_json) VALUES (?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), source, json.dumps(payload, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def compter_nouvelles_transactions_en_attente() -> int:
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM nouvelles_transactions WHERE integree=0").fetchone()[0]
    conn.close()
    return n


def derniere_alerte_monitoring() -> bool:
    log_path = REPORTS / "monitoring_log.jsonl"
    if not log_path.exists():
        return False
    with open(log_path) as f:
        lignes = [json.loads(l) for l in f if l.strip()]
    if not lignes:
        return False
    derniere = lignes[-1]
    return bool(derniere.get("alerte_performance") or derniere.get("alerte_drift"))


def integrer_nouvelles_transactions() -> int:
    """Fusionne les transactions en attente (table nouvelles_transactions de
    monitoring.db) dans la base transactionnelle de référence (toulouse_immo.db,
    table `transactions`), en complétant le contexte de quartier (revenu,
    pauvreté, démographie) depuis `dim_quartier`. C'est cette étape qui rend le
    réentraînement réellement effectif : sans elle, le pipeline s'exécuterait
    à nouveau sur des données inchangées."""
    conn_mon = get_conn()
    en_attente = pd.read_sql_query(
        "SELECT id, payload_json FROM nouvelles_transactions WHERE integree=0", conn_mon)
    if en_attente.empty:
        conn_mon.close()
        return 0

    conn_tx = sqlite3.connect(DATA / "toulouse_immo.db")
    dim = pd.read_sql_query("SELECT * FROM dim_quartier", conn_tx)
    max_id = conn_tx.execute("SELECT COALESCE(MAX(id_mutation), 0) FROM transactions").fetchone()[0]

    inserees = 0
    for i, row in en_attente.iterrows():
        p = json.loads(row["payload_json"])
        ctx = dim[dim.quartier == p.get("quartier")]
        if ctx.empty:
            continue  # quartier inconnu : donnée écartée (contrôle qualité)
        ctx = ctx.iloc[0]
        date_mut = p.get("date_mutation", datetime.now().strftime("%Y-%m-%d"))
        annee, mois = int(date_mut[:4]), int(date_mut[5:7])
        max_id += 1
        conn_tx.execute(
            "INSERT INTO transactions (id_mutation, date_mutation, annee, mois_mutation, trimestre, "
            "code_postal, quartier, type_bien, type_vente, surface_m2, nb_pieces, prix_vente, prix_m2, "
            "revenu_median_mensuel, taux_pauvrete_pct, population_2019, part_jeunes_pct, part_seniors_pct) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (max_id, date_mut, annee, mois, f"{annee}Q{(mois-1)//3+1}", str(ctx.code_postal), p["quartier"],
             p["type_bien"], "Vente", float(p["surface_m2"]), int(p.get("nb_pieces", 2)),
             float(p["prix_m2"]) * float(p["surface_m2"]), float(p["prix_m2"]),
             float(ctx.revenu_median_mensuel), float(ctx.taux_pauvrete_pct), int(ctx.population_2019),
             float(ctx.part_jeunes_pct), float(ctx.part_seniors_pct)),
        )
        inserees += 1
    conn_tx.commit()
    conn_tx.close()
    conn_mon.close()
    print(f"[INTÉGRATION] {inserees} nouvelles transactions fusionnées dans toulouse_immo.db "
          f"({len(en_attente) - inserees} écartées : quartier inconnu).")
    return inserees


def declencher_reentrainement(declencheur: str):
    """Intègre les nouvelles données puis relance la chaîne complète
    (feature engineering -> sélection -> entraînement -> optimisation ->
    sauvegarde), en respectant les mêmes étapes que l'entraînement initial
    (idempotence du pipeline)."""
    with open(BASE / "models" / "latest.json") as f:
        version_avant = json.load(f)["version"]

    integrer_nouvelles_transactions()

    print(f"[RÉENTRAÎNEMENT] Déclencheur : {declencheur}")
    etapes = [
        "01_feature_engineering.py", "02_feature_selection.py",
        "03_train_models.py", "04_optimize_hyperparameters.py", "05_save_model.py",
    ]
    statut = "succes"
    for script in etapes:
        print(f"  -> {script}")
        r = subprocess.run([sys.executable, str(BASE / "scripts" / script)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"     ÉCHEC : {r.stderr[-500:]}")
            statut = "echec"
            break

    version_apres = version_avant
    if statut == "succes":
        with open(BASE / "models" / "latest.json") as f:
            version_apres = json.load(f)["version"]
        conn = get_conn()
        conn.execute("UPDATE nouvelles_transactions SET integree=1 WHERE integree=0")
        conn.commit()
        conn.close()

    conn = get_conn()
    conn.execute(
        "INSERT INTO reentrainements (horodatage, declencheur, n_nouvelles_donnees, "
        "modele_version_avant, modele_version_apres, statut) VALUES (?,?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), declencheur,
         compter_nouvelles_transactions_en_attente(), version_avant, version_apres, statut),
    )
    conn.commit()
    conn.close()
    print(f"[RÉENTRAÎNEMENT] Statut : {statut}  (version {version_avant} -> {version_apres})")
    return statut == "succes"


def verifier_conditions_reentrainement():
    n_attente = compter_nouvelles_transactions_en_attente()
    alerte = derniere_alerte_monitoring()
    print(f"Nouvelles transactions en attente : {n_attente} (seuil : {SEUIL_NOUVELLES_TRANSACTIONS})")
    print(f"Alerte de monitoring active : {alerte}")

    if n_attente >= SEUIL_NOUVELLES_TRANSACTIONS:
        declencher_reentrainement("volume")
    elif alerte and SEUIL_ALERTE_MONITORING:
        declencher_reentrainement("alerte_monitoring")
    else:
        print("Aucune condition de réentraînement remplie : le modèle en production est conservé.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-retrain", action="store_true", help="Vérifie les conditions et réentraîne si besoin")
    ap.add_argument("--simuler-flux", type=int, default=0, help="Simule N nouvelles transactions collectées")
    args = ap.parse_args()

    if args.simuler_flux:
        for i in range(args.simuler_flux):
            collecter_nouvelle_transaction(
                {"quartier": "Rangueil / Saouzelong", "type_bien": "Appartement",
                 "surface_m2": 48, "prix_m2": 4200, "date_mutation": "2025-02-01"},
                source="flux_dvf_2025_T1",
            )
        print(f"[OK] {args.simuler_flux} nouvelles transactions simulées et journalisées.")

    if args.check_retrain:
        verifier_conditions_reentrainement()


if __name__ == "__main__":
    main()
