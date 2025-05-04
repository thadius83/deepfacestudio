"""
Reference image management for DeepFace API.
"""
from fastapi import HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import List
from deepface import DeepFace
import os
import glob
import shutil
import pickle
import uuid
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from . import config
from .utils import bgr_from_upload

def save_reference(label: str, img_bytes: bytes) -> str:
    """Persist reference image and return relative path."""
    # Create the label directory if it doesn't exist
    ref_dir = config.REFERENCE_DIR / label
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert image bytes to an image for processing
    img = bgr_from_upload(img_bytes)
    
    # Extract and preprocess the face to ensure quality
    try:
        # Try to extract face with enforce_detection=True for best quality
        face_objs = DeepFace.extract_faces(
            img, 
            detector_backend=config.DETECTOR_BACKEND,
            enforce_detection=True
        )
        if not face_objs:
            # Fall back to non-enforced detection if needed
            face_objs = DeepFace.extract_faces(
                img, 
                detector_backend=config.DETECTOR_BACKEND,
                enforce_detection=False
            )
            if not face_objs:
                raise ValueError("No face detected in reference image")
        
        # Use first detected face
        face_img = face_objs[0]["face"]
        
        # DeepFace returns normalized floating point images (range 0-1)
        # We need to convert to uint8 range (0-255) for proper saving
        if face_img.dtype == np.float64 or face_img.dtype == np.float32:
            face_img = (face_img * 255).astype(np.uint8)
        
        # Create a filename with consistent format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]  # Use shorter ID for readability
        fname = f"{label}_{timestamp}_{unique_id}.jpg"
        
        # Save the face image - ensuring it's valid for JPEG format
        if len(face_img.shape) != 3 or face_img.shape[2] != 3:
            if len(face_img.shape) == 2:  # Grayscale
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
            elif face_img.shape[2] == 4:  # RGBA
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2BGR)
                
        # Save the image
        success = cv2.imwrite(str(ref_dir / fname), face_img)
        if not success:
            raise ValueError(f"Failed to save image to {fname}")
        
        return f"{label}/{fname}"
    except Exception as e:
        # Fall back to saving the original image if face extraction fails
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        fname = f"{label}_{timestamp}_{unique_id}_original.jpg"
        cv2.imwrite(str(ref_dir / fname), img)
        return f"{label}/{fname}"

def rebuild_reference_db(force=False):
    """Rebuild the reference database if needed."""
    try:
        # Verify we have reference photos before rebuilding
        ref_photos_exist = False
        for item in os.listdir(config.REFERENCE_DIR):
            if os.path.isdir(os.path.join(config.REFERENCE_DIR, item)) and not item.startswith('.') and not item.startswith('ds_model'):
                ref_photos_exist = True
                break
        
        if not ref_photos_exist:
            return False
            
        # Let DeepFace manage the PKL file by calling find() with a sample image
        temp_img_path = "/tmp/temp_rebuild_face.jpg"
        blank_img = np.ones((100, 100, 3), dtype=np.uint8) * 200  # Light gray
        # Add facial features (basic circles for eyes and mouth)
        cv2.circle(blank_img, (35, 40), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(blank_img, (65, 40), 10, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(blank_img, (50, 70), (20, 10), 0, 0, 180, (0, 0, 0), -1)  # Mouth
        cv2.imwrite(temp_img_path, blank_img)
        
        # Generate expected pickle filename according to DeepFace format
        file_parts = [
            "ds", 
            "model", 
            config.MODEL_NAME, 
            "detector", 
            config.DETECTOR_BACKEND,
            "aligned", # Always use aligned
            "normalization", 
            "base",  # Use base normalization
            "expand", 
            "0"  # No expand percentage
        ]
        file_name = "_".join(file_parts) + ".pkl"
        file_name = file_name.replace("-", "").lower()
        
        pkl_path = os.path.join(str(config.REFERENCE_DIR), file_name)
            
        # If force rebuild is requested, delete the PKL file
        if force and os.path.exists(pkl_path):
            try:
                os.remove(pkl_path)
            except Exception:
                pass
        
        try:
            # Explicitly build the model first for consistency
            _ = DeepFace.build_model(config.MODEL_NAME)
            
            # Call find with refresh_database=True to update the representations
            result = DeepFace.find(
                img_path=temp_img_path,
                db_path=str(config.REFERENCE_DIR),
                model_name=config.MODEL_NAME,
                detector_backend=config.DETECTOR_BACKEND,
                enforce_detection=False,
                distance_metric="cosine",
                threshold=config.IDENTITY_THRESHOLD,
                silent=True,  # Disable debugging output
                refresh_database=True
            )
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
        
        # Verify the representation file exists
        if os.path.exists(pkl_path):
            return True
        else:
            # Check if any PKL file was created with a different naming convention
            all_files = os.listdir(config.REFERENCE_DIR)
            pkl_files = [f for f in all_files if f.endswith('.pkl')]
            return bool(pkl_files)
    except Exception:
        return False
