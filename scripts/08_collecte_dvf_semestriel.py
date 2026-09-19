"""
08_collecte_dvf_semestriel.py
================================
Bloc 5 — C5.3.4 : Automatisation du cycle de vie du système ML.

Complète 07_data_collection_retrain.py en branchant enfin une VRAIE source de
données sur la table `nouvelles_transactions` (jusqu'ici, elle n'était
alimentée qu'artificiellement via `--simuler-flux`).

Principe : les Demandes de Valeurs Foncières (DVF) ne sont publiées par
l'État que deux fois par an (avril et octobre, cf. data.gouv.fr). Il n'existe
donc pas de "flux temps réel" possible pour ce type de donnée — le bon
rythme d'exécution de ce script est justement semestriel, après chaque
publication.

Source : fichier DVF géolocalisé par commune, publié par la mission Etalab
(https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/),
au format :
    https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/communes/{departement}/{code_insee}.csv
Toulouse = commune INSEE 31555, département 31. Ce fichier couvre toute la
ville (tous ses codes postaux, tous ses arrondissements), on filtre ensuite
sur les 6 quartiers couverts par le modèle.

NOTE (à mentionner en soutenance) : le pattern d'URL a été vérifié
manuellement (curl sur l'index /geo-dvf/latest/csv/2024/communes/31/, qui a
confirmé le nom exact du fichier de Toulouse : 31555.csv, sans compression
gzip contrairement à une première version de ce script). Le téléchargement
et le filtrage restent malgré tout à valider de bout en bout avec un
lancement réel (colonnes DVF, volumétrie, dédoublonnage), l'environnement de
préparation de ce dossier n'ayant pas d'accès réseau sortant direct
(cf. 00_generate_calibrated_dataset.py) ; le test d'URL ci-dessus a été fait
depuis le poste personnel du candidat, en dehors de cet environnement.

Usage prévu (à relancer après chaque publication DVF, ~avril et ~octobre) :
    python3 scripts/08_collecte_dvf_semestriel.py --annee 2026
"""

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MONITORING_DB = DATA / "monitoring.db"

# Même mapping code postal -> quartier que api/main.py et
# 00_generate_calibrated_dataset.py, pour rester cohérent avec les
# indicatrices "q_<quartier>" attendues par le modèle. Dupliqué ici
# volontairement : chaque script du projet reste autonome et ne dépend pas
# d'imports croisés entre fichiers numérotés (dont les noms, commençant par
# un chiffre, ne sont de toute façon pas importables tels quels en Python).
QUARTIERS_CP = {
    "31000": "Capitole / Carmes",
    "31100": "Bagatelle / Reynerie",
    "31200": "Borderouge / Croix-Daurade",
    "31300": "Saint-Cyprien / Arènes",
    "31400": "Rangueil / Saouzelong",
    "31500": "Côte Pavée / Jolimont",
}

DEPARTEMENT = "31"
CODE_INSEE_TOULOUSE = "31555"

SCHEMA = """
CREATE TABLE IF NOT EXISTS nouvelles_transactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage_ajout TEXT NOT NULL,
    source          TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    integree        INTEGER DEFAULT 0
);
"""


def get_conn():
    DATA.mkdir(exist_ok=True)
    conn = sqlite3.connect(MONITORING_DB)
    conn.executescript(SCHEMA)
    return conn


def collecter_nouvelle_transaction(conn, payload: dict, source: str):
    conn.execute(
        "INSERT INTO nouvelles_transactions (horodatage_ajout, source, payload_json) VALUES (?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), source, json.dumps(payload, ensure_ascii=False)),
    )


def url_fichier_dvf(annee: int) -> str:
    return (
        f"https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/communes/"
        f"{DEPARTEMENT}/{CODE_INSEE_TOULOUSE}.csv"
    )


def telecharger_dvf(annee: int) -> pd.DataFrame:
    """Télécharge et charge le fichier DVF géolocalisé de Toulouse pour
    l'année demandée. pandas gère directement la décompression gzip et le
    téléchargement HTTP, pas besoin d'étape intermédiaire."""
    url = url_fichier_dvf(annee)
    colonnes = [
        "id_mutation", "date_mutation", "valeur_fonciere", "code_postal",
        "type_local", "surface_reelle_bati", "nombre_pieces_principales",
        "longitude", "latitude",
    ]
    return pd.read_csv(url, usecols=lambda c: c in colonnes, dtype={"code_postal": str})


def date_mutation_la_plus_recente_connue() -> str:
    """Dernière date déjà présente dans la base d'entraînement : tout ce qui
    est postérieur est considéré comme une donnée réellement nouvelle."""
    conn = sqlite3.connect(DATA / "toulouse_immo.db")
    try:
        (max_date,) = conn.execute("SELECT MAX(date_mutation) FROM transactions").fetchone()
    finally:
        conn.close()
    return max_date or "1900-01-01"


def filtrer_et_convertir(df: pd.DataFrame, depuis_le: str) -> list[dict]:
    df = df.dropna(subset=["valeur_fonciere", "surface_reelle_bati", "code_postal", "type_local"])
    df = df[df["code_postal"].isin(QUARTIERS_CP)]
    df = df[df["type_local"].isin(["Appartement", "Maison"])]
    df = df[df["surface_reelle_bati"] > 9]          # même garde-fou que TransactionInput (api/main.py)

    # Calcul du prix au m2 AVANT le filtre : certaines mutations DVF portent une
    # valeur_fonciere non nulle mais dérisoire (rectificatifs administratifs,
    # partages avec soulte symbolique...). Une fois arrondi à 1 décimale, un tel
    # prix_m2 peut tomber exactement à 0.0, ce qui fait échouer la contrainte
    # CHECK prix_vente > 0 de la table `transactions` au moment de l'intégration
    # (07_data_collection_retrain.py) — repéré lors d'un test avec de vraies
    # données DVF 2025 (un simple filtre `valeur_fonciere > 0` ne suffisait pas
    # à l'écarter). On filtre donc sur le prix_m2 réellement calculé, pas sur la
    # valeur brute.
    df = df.assign(prix_m2_calc=(df["valeur_fonciere"] / df["surface_reelle_bati"]).round(1))
    df = df[df["prix_m2_calc"] > 0]
    df = df[df["date_mutation"] > depuis_le]         # ne garder que le vraiment nouveau

    payloads = []
    for _, row in df.iterrows():
        payload = {
            "quartier": QUARTIERS_CP[row["code_postal"]],
            "type_bien": row["type_local"],
            "surface_m2": float(row["surface_reelle_bati"]),
            "nb_pieces": int(row["nombre_pieces_principales"]) if pd.notna(row.get("nombre_pieces_principales")) else 2,
            "prix_m2": float(row["prix_m2_calc"]),
            "date_mutation": str(row["date_mutation"])[:10],
        }
        # lat/lon réelles de la mutation, si présentes dans le fichier DVF :
        # permettent à l'intégration (07_data_collection_retrain.py) de
        # calculer une vraie distance au transport le plus proche, au lieu
        # de laisser la colonne vide (bug repéré lors d'un test réel : des
        # colonnes NULL faisaient planter la corrélation/RFE de
        # 02_feature_selection.py avec un ValueError "Input X contains NaN").
        lon, lat = row.get("longitude"), row.get("latitude")
        if pd.notna(lon) and pd.notna(lat):
            payload["lon"] = float(lon)
            payload["lat"] = float(lat)
        payloads.append(payload)
    return payloads


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annee", type=int, default=datetime.now().year,
                     help="Millésime DVF à récupérer (année de la publication)")
    args = ap.parse_args()

    print(f"[1/3] Téléchargement du fichier DVF {args.annee} pour Toulouse (INSEE {CODE_INSEE_TOULOUSE})...")
    print(f"       URL : {url_fichier_dvf(args.annee)}")
    df = telecharger_dvf(args.annee)
    print(f"       {len(df)} lignes brutes reçues (avant filtrage).")

    depuis_le = date_mutation_la_plus_recente_connue()
    print(f"[2/3] Filtrage : quartiers couverts, Appartement/Maison, postérieur au {depuis_le}...")
    nouvelles = filtrer_et_convertir(df, depuis_le)
    print(f"       {len(nouvelles)} transactions réellement nouvelles identifiées.")

    print("[3/3] Insertion dans nouvelles_transactions (monitoring.db)...")
    conn = get_conn()
    for payload in nouvelles:
        collecter_nouvelle_transaction(conn, payload, source=f"DVF_{args.annee}")
    conn.commit()
    conn.close()

    print(f"\n[OK] {len(nouvelles)} nouvelles transactions journalisées.")
    print("     Prochaine étape : python3 scripts/07_data_collection_retrain.py --check-retrain")


if __name__ == "__main__":
    main()
