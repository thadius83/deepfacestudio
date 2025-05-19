"""
Central configuration for the DeepFace backend.
Modify here rather than scattering constants through the code.
"""
from pathlib import Path

# Face‑detector and recognition model
#MODEL_NAME = "ArcFace"
MODEL_NAME = "Facenet"  # ArcFace is not available in the current version of deepface
#MODEL_NAME = "OpenFace"
DETECTOR_BACKEND = "retinaface"

# Threshold for deciding "same person"
IDENTITY_THRESHOLD = 0.30  # Default 70% confidence (1.0 - 0.3 = 0.7)

# Where reference images are stored (mapped as a volume in docker‑compose)
REFERENCE_DIR = Path("/data/reference_db")

# Accepted image MIME sub‑types
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".webp")
