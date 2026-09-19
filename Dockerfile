# ToulouseImmo Analytics — Bloc 5 : image de déploiement de l'API de prédiction
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

# --workers 2 : deux processus pour absorber la charge sans multiplier les
# ressources ; le rechargement du modèle en mémoire (05_save_model.py) rend
# chaque worker autonome, sans dépendance à un cache partagé.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
