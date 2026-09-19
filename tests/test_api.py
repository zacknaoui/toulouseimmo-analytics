"""
tests/test_api.py
====================
Tests exécutés automatiquement par le pipeline CI/CD (cf. .github/workflows/ci-cd.yml)
avant toute mise en production du modèle : c'est cette étape qui rend le
déploiement automatisable en toute confiance.

Le géocodage (API Base Adresse Nationale) est simulé ici (monkeypatch) : un
test unitaire ne doit pas dépendre d'un service réseau externe (fragile,
lent, parfois indisponible en environnement CI) — seul le comportement de
l'API elle-même est testé. Le géocodage réel est validé manuellement (en
recette / démo), pas en test unitaire.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
import api.main as main_module
from api.main import app

# Le context manager déclenche les événements startup/shutdown de l'API
# (chargement du modèle), indispensable pour tester /predict correctement.
client = TestClient(app)
client.__enter__()

GEO_CAPITOLE = {"lat": 43.6045, "lon": 1.4442, "postcode": "31000",
                "label": "3 Place du Capitole 31000 Toulouse", "score": 0.95}
GEO_BAGATELLE = {"lat": 43.638, "lon": 1.454, "postcode": "31100",
                      "label": "10 Chemin de Bagatelle 31100 Toulouse", "score": 0.9}
GEO_HORS_PERIMETRE = {"lat": 48.8566, "lon": 2.3522, "postcode": "75001",
                       "label": "1 Rue de Rivoli 75001 Paris", "score": 0.9}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["modele_charge"] is True


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body
    assert "metriques_test" in body


def test_predict_valid():
    payload = {
        "adresse": "3 Place du Capitole, Toulouse",
        "type_bien": "Appartement",
        "surface_m2": 55,
        "nb_pieces": 3,
        "mois_mutation": 6,
    }
    with patch.object(main_module, "_geocoder", return_value=GEO_CAPITOLE):
        r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prix_m2_predit"] > 0
    assert body["prix_vente_estime"] > 0
    assert len(body["intervalle_indicatif_m2"]) == 2
    assert body["quartier_detecte"] == "Capitole / Carmes"


def test_predict_adresse_hors_perimetre():
    payload = {
        "adresse": "1 Rue de Rivoli, Paris",
        "type_bien": "Appartement",
        "surface_m2": 55,
        "nb_pieces": 3,
        "mois_mutation": 6,
    }
    with patch.object(main_module, "_geocoder", return_value=GEO_HORS_PERIMETRE):
        r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_surface_hors_bornes():
    payload = {
        "adresse": "3 Place du Capitole, Toulouse",
        "type_bien": "Appartement",
        "surface_m2": 5,  # trop petite : contrainte gt=9
        "nb_pieces": 3,
        "mois_mutation": 6,
    }
    with patch.object(main_module, "_geocoder", return_value=GEO_CAPITOLE):
        r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_capitole_plus_cher_que_bagatelle():
    """Vérifie que le modèle a bien appris l'effet quartier (cohérence métier)."""
    base = dict(type_bien="Appartement", surface_m2=50, nb_pieces=2, mois_mutation=6)
    with patch.object(main_module, "_geocoder", return_value=GEO_CAPITOLE):
        p_capitole = client.post("/predict", json={**base, "adresse": "Capitole, Toulouse"}).json()
    with patch.object(main_module, "_geocoder", return_value=GEO_BAGATELLE):
        p_bagatelle = client.post("/predict", json={**base, "adresse": "Bagatelle, Toulouse"}).json()
    assert p_capitole["prix_m2_predit"] > p_bagatelle["prix_m2_predit"]
