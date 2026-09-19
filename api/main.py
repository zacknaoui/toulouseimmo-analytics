"""
api/main.py
=============
Bloc 5 — C5.3.2 : Déploiement du modèle via API + intégration dans le
processus CI/CD.

API FastAPI qui expose le modèle de prédiction du prix au m² :
  - GET  /health    : vérification que l'API et le modèle sont opérationnels
  - GET  /model-info: métadonnées de la version de modèle chargée
  - POST /predict    : prédiction du prix au m² à partir d'une ADRESSE
  - GET  /           : interface web de démonstration (formulaire + résultat)

Le modèle est chargé une seule fois au démarrage (via models/latest.json,
cf. 05_save_model.py) : aucun rechargement disque à chaque requête.

Extension (Bloc 5, itération 2) : le bien est désormais décrit par son
adresse exacte plutôt que par une sélection manuelle de quartier. L'adresse
est géocodée à la volée (API Base Adresse Nationale, gratuite, sans clé) et
sert à calculer deux variables prédictives supplémentaires :
  - distance_transport_m : distance (m) à l'arrêt Tisséo le plus proche
  - est_metro_proche     : l'arrêt le plus proche est-il une station de métro

Minimisation (RGPD) : l'adresse n'est utilisée qu'en mémoire, pour cette
seule requête, afin de calculer ces deux variables dérivées. Elle n'est
jamais journalisée ni persistée.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.neighbors import BallTree
from starlette.middleware.base import BaseHTTPMiddleware

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
DATA = BASE / "data"
DB = DATA / "toulouse_immo.db"
STATIC = Path(__file__).resolve().parent / "static"

BAN_URL = "https://api-adresse.data.gouv.fr/search/"

app = FastAPI(
    title="ToulouseImmo Analytics — API de prédiction du prix au m²",
    description="Bloc 5 RNCP39586 — Modèle d'apprentissage automatique (Data Scientist)",
    version="1.1",
)


class NoCacheMiddleware(BaseHTTPMiddleware):
    """Empêche le navigateur de garder en cache une ancienne version de
    l'interface statique (index.html / JS) : sans ça, un rafraîchissement
    "normal" peut réafficher un bug déjà corrigé côté serveur — piège vécu
    pendant les tests de cette démo."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response


app.add_middleware(NoCacheMiddleware)

# Rattachement code postal → quartier — identique à 02_transformation.py,
# pour que le quartier déduit de l'adresse corresponde exactement aux
# indicatrices "q_<quartier>" attendues par le modèle.
QUARTIERS_CP = {
    "31000": "Capitole / Carmes",
    "31100": "Bagatelle / Reynerie",
    "31200": "Borderouge / Croix-Daurade",
    "31300": "Saint-Cyprien / Arènes",
    "31400": "Rangueil / Saouzelong",
    "31500": "Côte Pavée / Jolimont",
}

# Garde-fou de robustesse (C5.3.3 — monitoring) : le modèle a un R² test
# modeste sur données réelles DVF (~0.37 même après enrichissement). Pour
# certaines combinaisons de features, l'arbre XGBoost peut extrapoler vers
# des valeurs physiquement impossibles (prix au m² négatif ou aberrant). On
# borne donc la prédiction à la plage réellement observée dans les
# transactions DVF 2024 (avec une marge), plutôt que de renvoyer un chiffre
# non-sens à l'utilisateur final.
PRIX_M2_PLANCHER = 1200.0
PRIX_M2_PLAFOND = 35000.0

_state = {}


def _load_model():
    with open(MODELS / "latest.json") as f:
        pointeur = json.load(f)
    version_dir = MODELS / pointeur["version"]
    with open(version_dir / "metadata.json") as f:
        meta = json.load(f)
    modele = joblib.load(version_dir / "model.joblib")
    scaler_path = version_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    _state.update(modele=modele, scaler=scaler, meta=meta)


def _load_moyennes_quartier():
    """Moyennes socio-économiques réelles par quartier, calculées depuis la
    base — plutôt que des valeurs codées en dur côté client, pour rester
    cohérent si la base évolue (nouvelle collecte, nouveau millésime)."""
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT quartier, AVG(revenu_median_mensuel) AS revenu, "
        "AVG(taux_pauvrete_pct) AS pauvrete, AVG(population_2019) AS population, "
        "AVG(part_jeunes_pct) AS jeunes, AVG(part_seniors_pct) AS seniors "
        "FROM transactions GROUP BY quartier",
        conn,
    )
    conn.close()
    _state["moyennes_quartier"] = df.set_index("quartier").to_dict(orient="index")


def _load_arrets_transport():
    """Charge les arrêts Tisséo et construit un BallTree (métrique
    haversine) pour retrouver l'arrêt le plus proche d'une adresse en
    O(log n), sans dépendance réseau à l'inférence."""
    df = pd.read_csv(DATA / "arrets_tisseo.csv", sep=";", encoding="utf-8-sig")
    coords = df["Geo Point"].str.split(",", expand=True).astype(float)
    df["lat"] = coords[0]
    df["lon"] = coords[1]
    df["mode"] = df["CONC_MODE"].fillna("")
    df["est_metro"] = df["mode"].str.contains("metro", case=False, na=False)
    df = df.dropna(subset=["lat", "lon"])
    tree = BallTree(np.radians(df[["lat", "lon"]].to_numpy()), metric="haversine")
    _state["arrets"] = df
    _state["arbre_arrets"] = tree


@app.on_event("startup")
def startup():
    _load_model()
    _load_moyennes_quartier()
    _load_arrets_transport()


def _geocoder(adresse: str) -> dict:
    try:
        resp = requests.get(BAN_URL, params={"q": adresse, "limit": 1}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise HTTPException(503, f"Service de géocodage indisponible : {e}")

    features = data.get("features") or []
    if not features:
        raise HTTPException(422, "Adresse introuvable. Vérifiez la saisie (numéro, rue, ville).")

    props = features[0]["properties"]
    lon, lat = features[0]["geometry"]["coordinates"]
    if props.get("score", 0) < 0.3:
        raise HTTPException(422, "Adresse trop imprécise pour être géolocalisée fiablement.")

    return {
        "lat": lat,
        "lon": lon,
        "postcode": props.get("postcode"),
        "label": props.get("label"),
        "score": props.get("score"),
    }


def _distance_transport(lat: float, lon: float) -> dict:
    d_rad, idx = _state["arbre_arrets"].query(np.radians([[lat, lon]]), k=1)
    distance_m = float(d_rad[0][0] * 6_371_000)
    arret = _state["arrets"].iloc[idx[0][0]]
    return {
        "distance_m": round(distance_m, 0),
        "est_metro": bool(arret["est_metro"]),
        "mode": arret["mode"],
        "nom": arret["NOM_ARRET"],
        "lat": float(arret["lat"]),
        "lon": float(arret["lon"]),
    }


class TransactionInput(BaseModel):
    adresse: str = Field(..., description="Adresse complète du bien, ex: '12 rue de Metz, Toulouse'")
    type_bien: str = Field(..., description="'Appartement' ou 'Maison'")
    surface_m2: float = Field(..., gt=9, le=400)
    nb_pieces: int = Field(..., ge=1, le=10)
    mois_mutation: int = Field(..., ge=1, le=12)


class PredictionOutput(BaseModel):
    prix_m2_predit: float
    prix_m2_brut: float
    prix_vente_estime: float
    modele_version: str
    intervalle_indicatif_m2: list[float]
    plafonne: bool = False
    adresse_normalisee: str
    quartier_detecte: str
    distance_transport_m: float
    type_arret_proche: str
    est_metro_proche: bool
    lat: float
    lon: float
    arret_lat: float
    arret_lon: float
    arret_nom: str


@app.get("/health")
def health():
    return {"status": "ok", "modele_charge": "modele" in _state}


@app.get("/model-info")
def model_info():
    if "meta" not in _state:
        raise HTTPException(503, "Modèle non chargé")
    return _state["meta"]


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: TransactionInput):
    if "modele" not in _state:
        raise HTTPException(503, "Modèle non chargé")
    if payload.type_bien not in ("Appartement", "Maison"):
        raise HTTPException(422, "type_bien doit être 'Appartement' ou 'Maison'")

    geo = _geocoder(payload.adresse)
    quartier = QUARTIERS_CP.get(str(geo["postcode"]))
    if quartier is None:
        raise HTTPException(
            422,
            f"Adresse hors du périmètre couvert par le modèle (code postal {geo['postcode']}). "
            f"Le modèle ne couvre que Toulouse intra-muros (6 quartiers).",
        )

    arret_info = _distance_transport(geo["lat"], geo["lon"])
    distance_m, est_metro, mode_arret = arret_info["distance_m"], arret_info["est_metro"], arret_info["mode"]
    moyennes = _state["moyennes_quartier"][quartier]

    features = _state["meta"]["features_attendues"]
    row = {f: 0 for f in features}
    row["surface_m2"] = payload.surface_m2
    row["nb_pieces"] = payload.nb_pieces
    row["is_maison"] = 1 if payload.type_bien == "Maison" else 0
    row["mois_mutation"] = payload.mois_mutation
    row["revenu_median_mensuel"] = moyennes["revenu"]
    row["taux_pauvrete_pct"] = moyennes["pauvrete"]
    row["population_2019"] = moyennes["population"]
    row["part_jeunes_pct"] = moyennes["jeunes"]
    row["part_seniors_pct"] = moyennes["seniors"]
    row["distance_transport_m"] = distance_m
    row["est_metro_proche"] = int(est_metro)
    q_col = f"q_{quartier}"
    if q_col in row:
        row[q_col] = 1

    X = pd.DataFrame([row])[features]
    if _state["scaler"] is not None:
        X = pd.DataFrame(_state["scaler"].transform(X), columns=features)

    pred_brut = float(_state["modele"].predict(X)[0])
    pred = min(max(pred_brut, PRIX_M2_PLANCHER), PRIX_M2_PLAFOND)
    plafonne = abs(pred - pred_brut) > 1e-6
    mae_test = _state["meta"]["metriques_test"]["mae_test"]

    return PredictionOutput(
        prix_m2_predit=round(pred, 0),
        prix_m2_brut=round(pred_brut, 1),
        prix_vente_estime=round(pred * payload.surface_m2, 0),
        modele_version=_state["meta"]["version"],
        intervalle_indicatif_m2=[
            round(max(pred - mae_test, PRIX_M2_PLANCHER), 0),
            round(min(pred + mae_test, PRIX_M2_PLAFOND), 0),
        ],
        plafonne=plafonne,
        adresse_normalisee=geo["label"],
        quartier_detecte=quartier,
        distance_transport_m=distance_m,
        type_arret_proche=mode_arret,
        est_metro_proche=est_metro,
        lat=geo["lat"],
        lon=geo["lon"],
        arret_lat=arret_info["lat"],
        arret_lon=arret_info["lon"],
        arret_nom=str(arret_info["nom"]),
    )


# Interface web de démonstration : servie en dernier pour ne jamais masquer
# les routes API ci-dessus (health / model-info / predict).
if STATIC.exists():
    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
