"""
01_feature_engineering.py
===========================
Bloc 5 — C5.2.1 : Construction du jeu de données exploitable pour le ML

Lit directement `data/toulouse_immo.db` (table `transactions`), exactement
comme correlations_b2.py / kruskal_b2.py du Bloc 1/2 : ce script est donc
100% compatible avec la vraie base produite par 03_chargement.py du candidat.
Il suffit de remplacer data/toulouse_immo.db par la vraie base pour que tout
le pipeline Bloc 5 tourne sur les vraies données, sans aucune autre modif.

Variable dépendante (cible)   : prix_m2 (€ / m²)
Variables prédictives brutes  : quartier, type_bien, surface_m2, nb_pieces,
                                 revenu_median_mensuel, taux_pauvrete_pct,
                                 population_2019, part_jeunes_pct,
                                 part_seniors_pct, mois_mutation, trimestre
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
DB = DATA_DIR / "toulouse_immo.db"


def load_raw_dataset() -> pd.DataFrame:
    if not DB.exists():
        raise SystemExit(
            f"Base introuvable : {DB}\n"
            f"Lancer d'abord : python3 scripts/00_generate_calibrated_dataset.py\n"
            f"(ou déposer la vraie base toulouse_immo.db produite par le Bloc 1/2)."
        )
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    print(f"[INFO] {len(df)} transactions chargées depuis {DB.name} (table transactions).")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_maison"] = (df["type_bien"] == "Maison").astype(int)

    before = len(df)
    df = df[(df.surface_m2 >= 9) & (df.surface_m2 <= 400)]
    df = df[(df.prix_m2 >= 500) & (df.prix_m2 <= 40000)]
    df = df[df.nb_pieces.between(1, 10)]
    df = df.dropna(subset=["prix_m2", "surface_m2", "quartier", "type_bien"])
    removed = before - len(df)
    if removed:
        print(f"[INFO] {removed} lignes écartées lors du contrôle qualité (valeurs aberrantes / manquantes).")

    df = pd.get_dummies(df, columns=["quartier"], prefix="q", drop_first=False)
    return df


def main():
    raw = load_raw_dataset()
    df = build_features(raw)

    target = "prix_m2"
    feature_cols = [
        "surface_m2", "nb_pieces", "is_maison",
        "revenu_median_mensuel", "taux_pauvrete_pct",
        "population_2019", "part_jeunes_pct", "part_seniors_pct",
        "mois_mutation", "distance_transport_m", "est_metro_proche",
    ] + [c for c in df.columns if c.startswith("q_")]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    DATA_DIR.mkdir(exist_ok=True)
    X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(DATA_DIR / "y_test.csv", index=False)

    meta = {
        "n_lignes_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "target": target,
    }
    with open(DATA_DIR / "feature_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Jeu de données ML prêt : {len(df)} lignes, {len(feature_cols)} variables prédictives.")
    print(f"[OK] Split : {len(X_train)} train / {len(X_test)} test (80/20, random_state=42).")
    print(f"[OK] Variables : {feature_cols}")


if __name__ == "__main__":
    main()
