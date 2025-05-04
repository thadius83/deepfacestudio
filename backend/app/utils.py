"""
Utility functions for the DeepFace API.
"""
from fastapi import HTTPException
from deepface import DeepFace
import numpy as np
import cv2
import threading
from . import config

# ---------- model cache (load once per process) ----------
_models_lock = threading.Lock()
_models = {}  # {model_name: keras.Model}

def get_model(name: str):
    """Get or create a DeepFace model, with thread-safe caching."""
    with _models_lock:
        if name not in _models:
            _models[name] = DeepFace.build_model(name)
        return _models[name]

def bgr_from_upload(data: bytes):
    """Convert raw bytes → OpenCV‑BGR ndarray, with PIL fallback for broader format support."""
    # First try OpenCV's decoder (fast for JPG, PNG)
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    if bgr is None:
        try:
            # Fall back to PIL which supports more formats (like WebP)
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(data)).convert("RGB")
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise HTTPException(415, f"Unsupported or corrupt image: {str(e)}")
    
    if bgr is None or bgr.size == 0:
        raise HTTPException(415, "Failed to decode image (empty or corrupt)")
        
    return bgr

def to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    return obj
