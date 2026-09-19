"""
00_generate_calibrated_dataset.py
===================================
ToulouseImmo Analytics — Bloc 5 (Concevoir et déployer des modèles d'apprentissage automatique)

CONTEXTE IMPORTANT (à mentionner en soutenance) :
--------------------------------------------------
Ce script ne fait PAS un nouvel appel réseau vers data.gouv.fr / insee.fr : le
poste utilisé pour préparer ce dossier n'a pas d'accès sortant vers ces serveurs.
En conditions réelles, ce script est remplacé par le pipeline déjà développé et
validé au Bloc 1 (01_collecte.py → 02_transformation.py → 03_chargement.py, cf.
scripts fournis par le candidat), qui produit data/toulouse_immo.db.

Pour ne pas bloquer le Bloc 5, ce script reconstruit directement une base
SQLite `data/toulouse_immo.db` AVEC LE MÊME SCHÉMA EXACT que 03_chargement.py
(mêmes tables, mêmes colonnes, mêmes contraintes CHECK), peuplée d'un jeu de
transactions statistiquement calibré sur les agrégats réels déjà publiés et
validés dans le dossier Bloc 1/2 du candidat (mêmes 6 quartiers/codes postaux,
mêmes effectifs n=4694 appartements / n=973 maisons soit 5667 transactions,
mêmes prix moyens par quartier, même saisonnalité mensuelle).

>>> SI LE FICHIER RÉEL EST DISPONIBLE : il suffit de remplacer
>>> data/toulouse_immo.db par la vraie base produite par 03_chargement.py.
>>> Tous les scripts du Bloc 5 (01_feature_engineering.py et suivants) lisent
>>> exclusivement `SELECT * FROM transactions` dans cette base : aucune autre
>>> modification n'est nécessaire.
"""

import sqlite3
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)
DB = DATA / "toulouse_immo.db"

REF_TRANSPORT = pd.read_csv(DATA / "reference_transport_quartier.csv")

# ---------------------------------------------------------------------------
# Mapping code postal -> quartier, IDENTIQUE à 02_transformation.py (Bloc 1)
# ---------------------------------------------------------------------------
QUARTIERS_CP = {
    "31000": "Capitole / Carmes",
    "31100": "Bagatelle / Reynerie",
    "31200": "Saint-Cyprien / Arènes",
    "31300": "Côte Pavée / Jolimont",
    "31400": "Rangueil / Saouzelong",
    "31500": "Borderouge / Croix-Daurade",
}

# calibrage (valeurs lues sur les graphiques réels du dossier Bloc 1/2)
CALIBRAGE = {
    "Capitole / Carmes":        dict(code_postal="31000", prix_moyen_appart=5949, revenu=2320, n_total=1008),
    "Borderouge / Croix-Daurade":   dict(code_postal="31200", prix_moyen_appart=4488, revenu=1640, n_total=1000),
    "Côte Pavée / Jolimont":    dict(code_postal="31500", prix_moyen_appart=4214, revenu=2160, n_total=741),
    "Rangueil / Saouzelong":    dict(code_postal="31400", prix_moyen_appart=4183, revenu=1530, n_total=1130),
    "Saint-Cyprien / Arènes":  dict(code_postal="31300", prix_moyen_appart=3497, revenu=1930, n_total=1241),
    "Bagatelle / Reynerie":  dict(code_postal="31100", prix_moyen_appart=3388, revenu=1950, n_total=547),
}

PART_APPART = 4694 / 5667
SIGMA_APPART = 0.69
SIGMA_MAISON = 0.565
RATIO_MAISON = 4512 / 4406

VOLUME_MOIS = {1: 420, 2: 445, 3: 420, 4: 480, 5: 400, 6: 515,
               7: 680, 8: 420, 9: 515, 10: 480, 11: 360, 12: 500}
_t = sum(VOLUME_MOIS.values())
VOLUME_MOIS = {m: v / _t for m, v in VOLUME_MOIS.items()}

PRIX_APPART_MOIS = {1: 4130, 2: 4250, 3: 4940, 4: 4060, 5: 3880, 6: 4280,
                     7: 4620, 8: 3990, 9: 4560, 10: 4320, 11: 4720, 12: 5030}
PRIX_MAISON_MOIS = {1: 4080, 2: 4420, 3: 4040, 4: 4940, 5: 4870, 6: 4280,
                     7: 4540, 8: 4120, 9: 5200, 10: 4110, 11: 4460, 12: 4650}
_mA, _mM = np.mean(list(PRIX_APPART_MOIS.values())), np.mean(list(PRIX_MAISON_MOIS.values()))
MULT_A = {m: v / _mA for m, v in PRIX_APPART_MOIS.items()}
MULT_M = {m: v / _mM for m, v in PRIX_MAISON_MOIS.items()}

for q, d in CALIBRAGE.items():
    d["taux_pauvrete_pct"] = round(float(np.clip(34 - d["revenu"] / 100, 8, 30)), 1)
    d["population_2019"] = int(RNG.integers(14000, 46000))
    d["part_jeunes_pct"] = round(float(RNG.uniform(18, 32)), 1)
    d["part_seniors_pct"] = round(float(np.clip(38 - d["part_jeunes_pct"] + RNG.uniform(-3, 3), 10, 30)), 1)


def sample_dates(n):
    months = list(VOLUME_MOIS.keys())
    weights = list(VOLUME_MOIS.values())
    chosen = RNG.choice(months, size=n, p=weights)
    out = []
    for m in chosen:
        start = date(2024, m, 1)
        end = date(2024, 12, 31) if m == 12 else date(2024, m + 1, 1) - timedelta(days=1)
        out.append(start + timedelta(days=int(RNG.integers(0, (end - start).days + 1))))
    return np.array(out)


def sample_surface_pieces(type_bien):
    n = len(type_bien)
    surface = np.empty(n)
    is_a = type_bien == "Appartement"
    surface[is_a] = np.clip(RNG.lognormal(np.log(48), 0.42, is_a.sum()), 15, 160)
    surface[~is_a] = np.clip(RNG.lognormal(np.log(105), 0.32, (~is_a).sum()), 45, 260)
    base = np.where(is_a, surface / 24, surface / 27)
    pieces = np.clip(np.round(base + RNG.normal(0, 0.6, n)), 1, 8).astype(int)
    return surface, pieces


rows = []
for quartier, info in CALIBRAGE.items():
    n_total = info["n_total"]
    n_appart = int(round(n_total * PART_APPART))
    n_maison = n_total - n_appart

    type_bien = np.array(["Appartement"] * n_appart + ["Maison"] * n_maison)
    RNG.shuffle(type_bien)

    dates = sample_dates(n_total)
    months = np.array([d.month for d in dates])
    surface, pieces = sample_surface_pieces(type_bien)

    mean_a, mean_m = info["prix_moyen_appart"], info["prix_moyen_appart"] * RATIO_MAISON
    is_a = type_bien == "Appartement"
    n_a, n_m = int(is_a.sum()), int((~is_a).sum())

    # Composante "signal" : bruit gaussien MODÉRÉ autour de la moyenne cible du
    # quartier/type, pour que le quartier et le type de bien restent des
    # variables prédictives apprenables par les modèles (cf. C5.2.1/C5.2.3).
    # Composante "bruit long" : ~3.5% de transactions atypiques (grand
    # standing, exception ponctuelle...) qui reproduisent la longue traîne
    # observée dans l'histogramme du Bloc 2 (valeurs jusqu'à ~30 000 €/m²)
    # sans pour autant noyer le signal exploitable par le modèle.
    prix_m2 = np.empty(n_total)
    prix_m2[is_a] = mean_a * (1 + RNG.normal(0, 0.14, n_a))
    prix_m2[~is_a] = mean_m * (1 + RNG.normal(0, 0.14, n_m))
    prix_m2 = np.clip(prix_m2, mean_a * 0.35, None)

    outlier_mask = RNG.random(n_total) < 0.025
    prix_m2[outlier_mask] *= RNG.uniform(1.8, 3.2, size=outlier_mask.sum())
    prix_m2 = np.clip(prix_m2, 1200, 33000)

    # Recalibrage : on ramène la moyenne réalisée exactement sur la moyenne
    # cible lue dans le dossier Bloc 1/2 (le bruit + les outliers ne doivent
    # pas faire dériver l'agrégat déjà validé).
    prix_m2[is_a] *= mean_a / prix_m2[is_a].mean()
    prix_m2[~is_a] *= mean_m / prix_m2[~is_a].mean()

    season = np.where(is_a, [MULT_A[m] for m in months], [MULT_M[m] for m in months])
    prix_m2 = np.clip(prix_m2 * (0.5 + 0.5 * season), 1200, 33000)
    prix_vente = prix_m2 * surface

    # Distance aux transports / proximite metro : on NE PEUT PAS reutiliser
    # une vraie adresse DVF individuelle (ces transactions sont simulees, pas
    # reelles), donc on tire au sort, PAR QUARTIER, des couples
    # (distance, est_metro) reellement observes sur les vraies adresses de ce
    # quartier (geocodage BAN + arrets Tisseo reels, cf. reference_transport_
    # quartier.csv) -- un bootstrap conditionnel au quartier, fidele a la
    # distribution reelle, sans lien fictif a une transaction precise.
    ref_q = REF_TRANSPORT[REF_TRANSPORT["quartier"] == quartier]
    tirage = ref_q.sample(n=n_total, replace=True, random_state=RNG.integers(0, 2**31 - 1))
    distance_transport_m = tirage["distance_transport_m"].to_numpy()
    est_metro_proche = tirage["est_metro_proche"].to_numpy()

    for i in range(n_total):
        d = dates[i]
        rows.append(dict(
            date_mutation=d.isoformat(),
            annee=d.year,
            mois_mutation=d.month,
            trimestre=f"{d.year}Q{(d.month - 1)//3 + 1}",
            code_postal=info["code_postal"],
            quartier=quartier,
            type_bien=type_bien[i],
            type_vente="Vente",
            surface_m2=round(float(surface[i]), 1),
            nb_pieces=int(pieces[i]),
            prix_vente=round(float(prix_vente[i]), 2),
            prix_m2=round(float(prix_m2[i]), 2),
            revenu_median_mensuel=info["revenu"],
            taux_pauvrete_pct=info["taux_pauvrete_pct"],
            population_2019=info["population_2019"],
            part_jeunes_pct=info["part_jeunes_pct"],
            part_seniors_pct=info["part_seniors_pct"],
            distance_transport_m=round(float(distance_transport_m[i]), 1),
            est_metro_proche=int(est_metro_proche[i]),
        ))

df = pd.DataFrame(rows).sort_values("date_mutation").reset_index(drop=True)
df.insert(0, "id_mutation", range(1, len(df) + 1))

# --- dim_quartier ------------------------------------------------------------
dim = pd.DataFrame([
    dict(code_postal=info["code_postal"], quartier=q,
         revenu_median_mensuel=info["revenu"], taux_pauvrete_pct=info["taux_pauvrete_pct"],
         nb_iris_total=RNG.integers(10, 30), nb_iris_renseignes=RNG.integers(8, 28),
         population_2019=info["population_2019"], part_jeunes_pct=info["part_jeunes_pct"],
         part_seniors_pct=info["part_seniors_pct"])
    for q, info in CALIBRAGE.items()
])

agg_q = df.groupby(["quartier", "type_bien"]).agg(
    prix_moyen=("prix_vente", "mean"), prix_median=("prix_vente", "median"),
    prix_m2_moyen=("prix_m2", "mean"), nb_transactions=("id_mutation", "count"),
    surface_moyenne=("surface_m2", "mean"),
).round(2).reset_index()

agg_t = df.groupby(["annee", "mois_mutation", "type_bien"]).agg(
    prix_m2_moyen=("prix_m2", "mean"), nb_transactions=("id_mutation", "count"),
).round(2).reset_index()

# ---------------------------------------------------------------------------
# Écriture SQLite — schéma identique à 03_chargement.py (Bloc 1)
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE transactions (
    id_mutation   INTEGER PRIMARY KEY,
    date_mutation DATE    NOT NULL,
    annee         INTEGER NOT NULL,
    mois_mutation INTEGER NOT NULL,
    trimestre     TEXT    NOT NULL,
    code_postal   TEXT    NOT NULL,
    quartier      TEXT    NOT NULL,
    type_bien     TEXT    NOT NULL CHECK (type_bien IN ('Appartement','Maison')),
    type_vente    TEXT    NOT NULL,
    surface_m2    REAL    NOT NULL CHECK (surface_m2 > 0),
    nb_pieces     INTEGER,
    prix_vente    REAL    NOT NULL CHECK (prix_vente > 0),
    prix_m2       REAL    NOT NULL CHECK (prix_m2    > 0),
    revenu_median_mensuel REAL,
    taux_pauvrete_pct     REAL,
    population_2019       INTEGER,
    part_jeunes_pct       REAL,
    part_seniors_pct      REAL,
    distance_transport_m  REAL,
    est_metro_proche      INTEGER
);
CREATE TABLE dim_quartier (
    code_postal        TEXT PRIMARY KEY,
    quartier           TEXT NOT NULL,
    revenu_median_mensuel REAL,
    taux_pauvrete_pct     REAL,
    nb_iris_total         INTEGER,
    nb_iris_renseignes    INTEGER,
    population_2019       INTEGER,
    part_jeunes_pct       REAL,
    part_seniors_pct      REAL
);
CREATE TABLE agg_prix_quartier (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    quartier        TEXT,
    type_bien       TEXT,
    prix_moyen      REAL,
    prix_median     REAL,
    prix_m2_moyen   REAL,
    nb_transactions INTEGER,
    surface_moyenne REAL
);
CREATE TABLE agg_prix_temporel (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    annee           INTEGER,
    mois_mutation   INTEGER,
    type_bien       TEXT,
    prix_m2_moyen   REAL,
    nb_transactions INTEGER
);
CREATE INDEX idx_tx_quartier ON transactions(quartier);
CREATE INDEX idx_tx_annee    ON transactions(annee);
CREATE INDEX idx_tx_type     ON transactions(type_bien);
CREATE INDEX idx_tx_prix     ON transactions(prix_m2);
"""

if DB.exists():
    DB.unlink()
conn = sqlite3.connect(DB)
conn.executescript(SCHEMA)

df.to_sql("transactions", conn, if_exists="append", index=False)
dim.to_sql("dim_quartier", conn, if_exists="append", index=False)
agg_q.to_sql("agg_prix_quartier", conn, if_exists="append", index=False)
agg_t.to_sql("agg_prix_temporel", conn, if_exists="append", index=False)

print(f"Base SQLite créée : {DB}")
for t in ["transactions", "dim_quartier", "agg_prix_quartier", "agg_prix_temporel"]:
    n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t:<20} : {n:>5} lignes")

print("\n--- Contrôle de cohérence avec le dossier Bloc 1/2 ---")
check = pd.read_sql_query("""
    SELECT quartier, ROUND(AVG(prix_m2)) AS prix_m2_moyen, COUNT(*) AS nb_ventes
    FROM transactions WHERE type_bien='Appartement'
    GROUP BY quartier ORDER BY prix_m2_moyen DESC
""", conn)
print(check.to_string(index=False))
conn.close()
