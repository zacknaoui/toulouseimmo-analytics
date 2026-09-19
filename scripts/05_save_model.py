"""
05_save_model.py
===================
Bloc 5 — C5.3.1 : Sauvegarde du modèle d'apprentissage automatique

Librairies utilisées : joblib (sérialisation), json (métadonnées de version).

Modalités de sauvegarde :
  - Sérialisation binaire du modèle final (joblib, format optimisé pour les
    objets scikit-learn / numpy — plus rapide et plus compact que pickle).
  - Versioning par horodatage + hash court des hyperparamètres, avec un
    fichier `latest.json` qui pointe toujours vers la version courante
    (permet un rollback trivial en changeant juste ce pointeur).
  - Métadonnées associées (features attendues, métriques, date, source des
    données) pour garantir la traçabilité et la réutilisation correcte du
    modèle (schéma d'entrée attendu par l'API du script 06).

Modalités de réutilisation : voir api/main.py (script 06) qui recharge le
modèle avec `joblib.load()` à partir de `models/latest.json`.
"""

import json
import math
import pickle
import hashlib
from datetime import datetime
from pathlib import Path

import joblib


def _json_safe(v):
    """Neutralise les valeurs non représentables en JSON strict (NaN/Inf),
    par ex. le défaut missing=nan de XGBoost, qui casserait la réponse de
    l'API FastAPI (json standard) lors de GET /model-info."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v if isinstance(v, (int, float, str, bool)) else str(v)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

with open(DATA / "_meilleur_modele_tmp.pkl", "rb") as f:
    best = pickle.load(f)

nom, modele, scaler, features = best["nom"], best["modele"], best["scaler"], best["features"]

with open(BASE / "reports" / "optimisation.json") as f:
    opt = json.load(f)
metriques = [r for r in opt["comparaison_apres"] if r["modele"] == nom][0]

horodatage = datetime.now().strftime("%Y%m%d-%H%M%S")
hyper_str = json.dumps(getattr(modele, "get_params", lambda: {})(), sort_keys=True, default=str)
version_hash = hashlib.sha1(hyper_str.encode()).hexdigest()[:8]
version = f"{horodatage}-{version_hash}"

version_dir = MODELS / version
version_dir.mkdir(exist_ok=True)

joblib.dump(modele, version_dir / "model.joblib")
if scaler is not None:
    joblib.dump(scaler, version_dir / "scaler.joblib")

metadata = {
    "version": version,
    "date_entrainement": datetime.now().isoformat(timespec="seconds"),
    "modele": nom,
    "features_attendues": features,
    "necessite_scaler": scaler is not None,
    "target": "prix_m2",
    "metriques_test": {k: v for k, v in metriques.items() if k != "modele"},
    "hyperparametres": {k: _json_safe(v)
                         for k, v in getattr(modele, "get_params", lambda: {})().items()},
}
with open(version_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# Pointeur vers la version courante (utilisé par l'API et le monitoring)
with open(MODELS / "latest.json", "w") as f:
    json.dump({"version": version, "path": str(version_dir / "model.joblib")}, f, indent=2)

print(f"[OK] Modèle sauvegardé : {version_dir}/model.joblib")
print(f"[OK] Version : {version}")
print(f"[OK] Métadonnées : {version_dir}/metadata.json")
print(f"[OK] Pointeur 'latest' mis à jour : {MODELS}/latest.json")
print(f"\nMétriques du modèle sauvegardé : {metadata['metriques_test']}")
