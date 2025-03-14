# Utiliser Python 3.11 comme base
FROM python:3.11-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    libsqlite3-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances d'abord pour profiter du cache des couches Docker
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application
COPY . .

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Définir les variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ENABLE_CORS=false

# Commande pour démarrer l'application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]