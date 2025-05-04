"""Front‑end global settings."""
import os

# URL where the FastAPI service lives (docker‑compose uses service name)
API_URL = os.getenv("API_URL", "http://deepface-api:3900")
