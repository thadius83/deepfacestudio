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
    """Convert raw bytes → OpenCV‑BGR ndarray, with PIL fallback for broader format support.
    
    This function ensures consistent BGR output with proper color channel handling to prevent
    blue/purple tinting or other color matrix issues.
    """
    # First try OpenCV's decoder (fast for JPG, PNG)
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    if bgr is None:
        try:
            # Fall back to PIL which supports more formats (like WebP)
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(data)).convert("RGB")
            
            # Convert from PIL's RGB format to numpy array, then to OpenCV's BGR
            rgb_array = np.array(img)
            bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise HTTPException(415, f"Unsupported or corrupt image: {str(e)}")
    
    # Validate and normalize the image format
    if bgr is None or bgr.size == 0:
        raise HTTPException(415, "Failed to decode image (empty or corrupt)")
    
    # Ensure correct data type (convert from float to uint8 if needed)
    if bgr.dtype == np.float64 or bgr.dtype == np.float32:
        bgr = (bgr * 255).astype(np.uint8)
        
    # Ensure correct color format
    if len(bgr.shape) == 2:  # Grayscale
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.shape[2] == 4:  # RGBA
        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGBA2BGR)
    
    # Verify we have a valid 3-channel BGR image
    if len(bgr.shape) != 3 or bgr.shape[2] != 3:
        raise HTTPException(415, f"Invalid image format: {bgr.shape}")
        
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
