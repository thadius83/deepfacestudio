"""Front‑end global settings."""
import os

# URL where the FastAPI service lives
# Use localhost in devcontainer, service name in docker-compose
API_URL = os.getenv("API_URL", "http://localhost:3900")
