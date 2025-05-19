#!/usr/bin/env python3
"""
IMDB Celebrity Image Importer – Gender-Aware Version

• Uses FaceNet embeddings with RetinaFace detection
• Performs gender verification to ensure image matches celebrity gender
• Adaptively handles invalid MATLAB coordinates by scaling or using whole image
• Iterates entries up to `--limit`, filters by face_score
• Uses DeepFace with proper image format handling
• Saves the coarse crop as-is for verified images
• Creates per-celebrity subfolders, up to `--max_per_celeb`
• Logs processing details to `processing_log.csv`
"""
import os
import logging
import argparse
import unicodedata
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.io
import cv2
from deepface import DeepFace

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH            = Path("/workspace/imdb")
IMDB_ROOT            = DATA_PATH / "imdb_crop"
MAT_FILE             = IMDB_ROOT / "imdb.mat"
OUTPUT_PATH          = Path("/data/reference_db")
LOG_FILE             = OUTPUT_PATH / "processing_log.csv"
DEBUG_SAVE           = False
DEBUG_PATH           = OUTPUT_PATH / "debug"

MODEL_NAME           = "Facenet"
DETECTOR_BACKEND     = "retinaface"
MAX_IMAGES_PER_CELEB = 100  # Increased from 20 to allow more images per celebrity
FACE_SCORE_THRESHOLD = 3.0
CROP_MARGIN          = 0.55   # Padding around MATLAB bbox (0 to disable)
EXPAND_COORDINATES   = 0.0    # Expansion of raw MATLAB coordinates before processing (0 to disable)
DETECT_FACE          = True   # If False, skip face detection gating
GENDER_CHECK         = True   # If True, verify gender against MATLAB data
GENDER_THRESHOLD     = 0.60   # Minimum confidence for gender verification (lowered from 0.81)
MIN_CROP_SIZE        = 50     # Minimum size (width/height) for a valid face crop
MIN_IMAGE_SIZE       = 250    # Don't crop images smaller than this size (use full image)
TEMP_DIR             = Path("/tmp/deepface_temp")  # Temp directory for image files

# Gender constants (from MATLAB file)
GENDER_FEMALE = 0
GENDER_MALE = 1

# Aspect ratio limits for face crops
ASPECT_RATIO_MIN = 0.6  # Allow more rectangular crops
ASPECT_RATIO_MAX = 1.67  # Allow more rectangular crops

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _scalar(x):
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        if x.size == 1:
            return _scalar(x.item())
        return [_scalar(el) for el in x]
    if isinstance(x, bytes):
        return x.decode()
    return x


def _resolve_path(p):
    if isinstance(p, (str, bytes)):
        return p.decode() if isinstance(p, bytes) else p
    if isinstance(p, np.ndarray):
        return _resolve_path(p[0]) if p.size else ""
    return str(p)


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    txt = unicodedata.normalize("NFKD", text)
    txt = ''.join(c for c in txt if c.isascii())
    return txt.strip().lower().replace(" ", "_")


def parse_bbox(loc):
    while isinstance(loc, list) and len(loc) == 1:
        loc = loc[0]
    if not isinstance(loc, (list, tuple)) or len(loc) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(loc[i]) for i in range(4)]
        return int(x1), int(y1), int(x2), int(y2)
    except:
        return None


def adaptive_crop(img, loc, margin=CROP_MARGIN, expand=EXPAND_COORDINATES, min_size=MIN_CROP_SIZE):
    """
    Adaptively crop a face based on MATLAB bbox, handling invalid coordinates gracefully.
    
    Args:
        img: Image to crop
        loc: Face location from MATLAB
        margin: Padding around the final bbox (0 to disable)
        expand: Expansion factor for raw MATLAB coordinates (0 to disable)
        min_size: Minimum size for a valid crop
    
    If coordinates are invalid:
    1. First tries to scale them to fit within the image
    2. If that's not possible, defaults to using the entire image
    
    Returns the cropped image (never None).
    """
    h, w = img.shape[:2]
    
    # Default to full image
    x1, y1, x2, y2 = 0, 0, w, h
    
    # If source image is small, don't crop it
    if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
        return img.copy()
    
    # Try to use MATLAB coordinates
    bbox = parse_bbox(loc)
    if bbox:
        mx1, my1, mx2, my2 = bbox
        
        # Apply expansion to raw coordinates if requested
        if expand > 0:
            # Calculate center and dimensions
            center_x = (mx1 + mx2) / 2
            center_y = (my1 + my2) / 2
            width = mx2 - mx1
            height = my2 - my1
            
            # Apply expansion
            new_width = width * (1 + expand)
            new_height = height * (1 + expand)
            
            # Update coordinates
            mx1 = center_x - new_width / 2
            mx2 = center_x + new_width / 2
            my1 = center_y - new_height / 2
            my2 = center_y + new_height / 2
        
        # Check if we need to scale the coordinates
        scale_needed = False
        if mx1 >= w or my1 >= h or mx2 <= 0 or my2 <= 0 or mx1 >= mx2 or my1 >= my2:
            # These coordinates are clearly outside the image or invalid
            scale_needed = True
        elif mx2 > w * 1.5 or my2 > h * 1.5:
            # Coordinates are much larger than the image - likely a scale issue
            scale_needed = True
            
        if scale_needed:
            # Try to scale the coordinates
            # First, get the source dimensions from the coordinates
            src_w = mx2 - mx1
            src_h = my2 - my1
            
            if src_w > 0 and src_h > 0:
                # Calculate a scale factor to fit within the image
                scale_x = w / src_w
                scale_y = h / src_h
                scale = min(scale_x, scale_y) * 0.9  # Use 90% to ensure some margin
                
                # Apply scaling
                center_x = (mx1 + mx2) / 2
                center_y = (my1 + my2) / 2
                half_w = src_w * scale / 2
                half_h = src_h * scale / 2
                
                mx1 = center_x - half_w
                mx2 = center_x + half_w
                my1 = center_y - half_h
                my2 = center_y + half_h
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(w-1, mx1))
        y1 = max(0, min(h-1, my1))
        x2 = max(x1+1, min(w, mx2))
        y2 = max(y1+1, min(h, my2))
        
        # Calculate crop width and height
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # Calculate aspect ratio
        if crop_height > 0:
            aspect_ratio = crop_width / crop_height
        else:
            aspect_ratio = 0
            
        # Check for small dimensions or unusual aspect ratio
        if crop_width < min_size or crop_height < min_size:
            # Fall back to full image if dimensions are too small
            x1, y1, x2, y2 = 0, 0, w, h
        elif aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
            # Fall back to full image if aspect ratio is not squarish
            x1, y1, x2, y2 = 0, 0, w, h
    
    # Apply margin (if enabled)
    if margin > 0:
        crop_w = x2 - x1
        crop_h = y2 - y1
        pad_w = int(crop_w * margin)
        pad_h = int(crop_h * margin)
        
        # Ensure all coordinates are integers
        xa = int(max(0, x1 - pad_w))
        ya = int(max(0, y1 - pad_h))
        xb = int(min(w, x2 + pad_w))
        yb = int(min(h, y2 + pad_h))
    else:
        # No margin
        xa = int(x1)
        ya = int(y1)
        xb = int(x2)
        yb = int(y2)
    
    # Create crop with copy to ensure it's not a view
    return img[ya:yb, xa:xb].copy()

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_imdb_raw(limit=None):
    logger.info("Loading IMDB metadata from %s", MAT_FILE)
    mat = scipy.io.loadmat(str(MAT_FILE))
    imdb = mat['imdb'][0, 0]
    total = imdb['full_path'][0].shape[0]
    idxs = list(range(total))
    if limit:
        idxs = idxs[:limit]
    logger.info("Will process %d entries", len(idxs))
    return imdb, idxs

# ──────────────────────────────────────────────────────────────────────────────
# FACE DETECTION AND GENDER VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def verify_face_with_file(img, img_id, min_face_ratio=0.05, min_confidence=0.9):
    """
    Use DeepFace with temporary file to avoid "Invalid image input" errors.
    Additional quality checks to ensure valid face detection.
    
    Args:
        img: Image to analyze
        img_id: ID for temporary file
        min_face_ratio: Minimum ratio of face area to image area (0-1)
        min_confidence: Minimum confidence score for face detection
        
    Returns:
        (success, error_message, face_details).
    """
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Get image dimensions for quality checks
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w
    
    # Save the image to a temporary file
    temp_path = TEMP_DIR / f"temp_{img_id}.jpg"
    try:
        # Ensure image is in proper format before saving
        if img.dtype == np.float64 or img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
            
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
        success = cv2.imwrite(str(temp_path), img)
        if not success:
            return False, "Failed to save temporary image", None
    except Exception as e:
        return False, f"Error saving temporary image: {str(e)}", None
    
    try:
        # First try with enforce_detection=True for stricter detection
        try:
            faces = DeepFace.extract_faces(
                img_path=str(temp_path),
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True
            )
            
            # Check if we got at least one face with good quality
            if faces and len(faces) > 0:
                face = faces[0]
                # Check face quality
                if 'facial_area' in face and 'confidence' in face:
                    # Handle different formats of facial_area
                    facial_area = face['facial_area']
                    if isinstance(facial_area, dict):
                        try:
                            # Try regular dictionary access first
                            x = facial_area.get('x', 0)
                            y = facial_area.get('y', 0)
                            w = facial_area.get('w', 0)
                            h = facial_area.get('h', 0)
                        except Exception:
                            # Alternative approach if values() doesn't work
                            facial_area_values = list(facial_area.values())
                            if len(facial_area_values) >= 4:
                                x, y, w, h = facial_area_values[:4]
                            else:
                                x, y, w, h = 0, 0, 0, 0
                    elif isinstance(facial_area, (list, tuple)) and len(facial_area) >= 4:
                        # List/tuple format
                        x, y, w, h = facial_area[:4]
                    else:
                        # Unknown format, use defaults
                        x, y, w, h = 0, 0, 0, 0
                        
                    face_area = w * h
                    confidence = face['confidence']
                    
                    # Face area should be a reasonable portion of the image
                    face_ratio = face_area / img_area
                
                    # Quality checks - with safeguards against invalid dimensions
                    if w <= 0 or h <= 0:
                        return False, f"Invalid face dimensions: {w}x{h}", face
                    if face_ratio < min_face_ratio:
                        # Much more permissive with small faces
                        if face_ratio < 0.01:  # Only reject extremely tiny faces
                            return False, f"Face too small: {face_ratio:.2f} of image", face
                    if confidence < min_confidence:
                        return False, f"Low confidence: {confidence:.2f}", face
                    
                    # Return success along with face details
                    return True, None, face
                
                # Handle case where facial_area or confidence is missing
                return True, None, face
        except Exception:
            # Fall back to enforce_detection=False
            try:
                faces = DeepFace.extract_faces(
                    img_path=str(temp_path),
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=True
                )
                
                # Apply stricter quality checks to non-enforced detection
                if faces and len(faces) > 0:
                    face = faces[0]
                    
                    # More stringent checks for non-enforced detection
                    if 'facial_area' in face:
                        # Handle different formats of facial_area
                        facial_area = face['facial_area']
                        if isinstance(facial_area, dict):
                            try:
                                x = facial_area.get('x', 0)
                                y = facial_area.get('y', 0)
                                w = facial_area.get('w', 0)
                                h = facial_area.get('h', 0)
                            except Exception:
                                facial_area_values = list(facial_area.values())
                                if len(facial_area_values) >= 4:
                                    x, y, w, h = facial_area_values[:4]
                                else:
                                    x, y, w, h = 0, 0, 0, 0
                        elif isinstance(facial_area, (list, tuple)) and len(facial_area) >= 4:
                            x, y, w, h = facial_area[:4]
                        else:
                            x, y, w, h = 0, 0, 0, 0
                        
                        face_area = w * h
                        face_ratio = face_area / img_area
                        
                        # Higher thresholds for non-enforced detection but still permissive
                        if w <= 0 or h <= 0:
                            return False, f"Non-enforced invalid dimensions: {w}x{h}", face
                        if face_ratio < 0.01:  # Only reject extremely tiny faces
                            return False, f"Non-enforced face too small: {face_ratio:.2f} of image", face
                            
                        # Face should have reasonable aspect ratio (not too stretched)
                        aspect_ratio = w / max(1, h)
                        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                            return False, f"Face has unusual proportions: {aspect_ratio:.2f}", face
                    
                    # Apply confidence check if available
                    if 'confidence' in face and face['confidence'] < min_confidence * 0.9:
                        return False, f"Non-enforced low confidence: {face['confidence']:.2f}", face
                    
                    return True, None, face
                else:
                    return False, "No face detected in non-enforced mode", None
            except Exception as e2:
                return False, str(e2), None
                
        # If we get here with no faces, no face was detected
        return False, "No face detected", None
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

def verify_gender(img, img_id, expected_gender, threshold=GENDER_THRESHOLD):
    """
    Verify that the detected gender matches the expected gender.
    Uses multiple approaches to improve reliability.
    
    Args:
        img: Image to analyze
        img_id: ID for temporary file
        expected_gender: GENDER_MALE (1) or GENDER_FEMALE (0)
        threshold: Minimum confidence for gender verification
    
    Returns:
        (success, confidence, error_message, detailed_results)
        where detailed_results is a dict with woman_score, man_score, detected_gender
    """
    # Initialize detailed results
    detailed_results = {
        'woman_score': 0.0,
        'man_score': 0.0,
        'detected_gender': 'unknown'
    }
    
    # Skip if expected_gender is not valid
    if expected_gender is None or np.isnan(expected_gender):
        detailed_results['expected_gender'] = 'unknown'
        return True, 1.0, None, detailed_results  # Skip verification if gender is unknown
    
    # Set expected gender string
    detailed_results['expected_gender'] = 'female' if expected_gender == GENDER_FEMALE else 'male'
        
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Save the image to a temporary file
    temp_path = TEMP_DIR / f"gender_{img_id}.jpg"
    try:
        # Ensure the image is in the right format for saving
        if img.dtype == np.float64 or img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
            
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
        success = cv2.imwrite(str(temp_path), img)
        if not success:
            return False, 0.0, "Failed to save temporary image for gender verification", detailed_results
    except Exception as e:
        return False, 0.0, f"Error saving temporary image for gender verification: {str(e)}", detailed_results
    
    try:
        # First try with retinaface detector and no enforcement
        try:
            analysis = DeepFace.analyze(
                img_path=str(temp_path),
                actions=['gender'],
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                silent=True
            )
            
            # Extract gender results
            if isinstance(analysis, list):
                analysis = analysis[0]  # Get first face
                
            detected_gender = analysis.get('gender', None)
            if detected_gender is None:
                # Try different backend as fallback
                fallback_analysis = DeepFace.analyze(
                    img_path=str(temp_path),
                    actions=['gender'],
                    detector_backend="opencv",  # Try opencv as fallback
                    enforce_detection=False,
                    silent=True
                )
                
                if isinstance(fallback_analysis, list):
                    fallback_analysis = fallback_analysis[0]
                
                detected_gender = fallback_analysis.get('gender', None)
                
                if detected_gender is None:
                    return False, 0.0, "Gender not detected with any backend", detailed_results
            
            # Get the dominant gender and scores
            if isinstance(detected_gender, dict):
                # For DeepFace v1
                woman_score = detected_gender.get('Woman', 0)
                man_score = detected_gender.get('Man', 0)
                
                # Update detailed results
                detailed_results['woman_score'] = float(woman_score)
                detailed_results['man_score'] = float(man_score)
                detailed_results['detected_gender'] = 'female' if woman_score > man_score else 'male'
                
                # Compare with expected gender
                if expected_gender == GENDER_FEMALE:
                    return woman_score >= threshold, float(woman_score), None, detailed_results
                else:
                    return man_score >= threshold, float(man_score), None, detailed_results
            else:
                # For other DeepFace versions - string result
                is_female = detected_gender.lower() == 'female'
                detailed_results['detected_gender'] = 'female' if is_female else 'male'
                
                # Set approximate scores since exact values aren't available
                if is_female:
                    detailed_results['woman_score'] = 0.9
                    detailed_results['man_score'] = 0.1
                else:
                    detailed_results['woman_score'] = 0.1
                    detailed_results['man_score'] = 0.9
                
                # Match expected gender
                matches = detected_gender.lower() == ('female' if expected_gender == GENDER_FEMALE else 'male')
                confidence = 1.0  # Confidence is high since it's a definitive classification
                return matches, confidence, None, detailed_results
                
        except Exception as e:
            try:
                # Try one more approach - direct image analysis
                direct_analysis = DeepFace.analyze(
                    img_path=img,  # Try direct image instead of file
                    actions=['gender'],
                    detector_backend="opencv",  # Try opencv as last resort
                    enforce_detection=False,
                    silent=True
                )
                
                if isinstance(direct_analysis, list):
                    direct_analysis = direct_analysis[0]
                    
                detected_gender = direct_analysis.get('gender', None)
                if detected_gender is not None:
                    is_female = detected_gender.lower() == 'female'
                    detailed_results['detected_gender'] = 'female' if is_female else 'male'
                    
                    # Set scores
                    if is_female:
                        detailed_results['woman_score'] = 0.9
                        detailed_results['man_score'] = 0.1
                    else:
                        detailed_results['woman_score'] = 0.1
                        detailed_results['man_score'] = 0.9
                    
                    # Compare with expected
                    matches = is_female == (expected_gender == GENDER_FEMALE)
                    confidence = 0.9  # Slightly lower confidence for fallback
                    return matches, confidence, None, detailed_results
                else:
                    return False, 0.0, f"Gender analysis failed in all attempts: {str(e)}", detailed_results
            except Exception as e2:
                return False, 0.0, f"Gender analysis failed: {str(e)} / {str(e2)}", detailed_results
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

def generate_embedding(img, img_id):
    """
    Generate embedding for an image using a temporary file approach.
    Returns (success, embedding or error_message).
    """
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Save the image to a temporary file
    temp_path = TEMP_DIR / f"embed_{img_id}.jpg"
    try:
        success = cv2.imwrite(str(temp_path), img)
        if not success:
            return False, "Failed to save temporary image for embedding"
    except Exception as e:
        return False, f"Error saving temporary image for embedding: {str(e)}"
    
    try:
        # Generate embedding with skip detector
        embedding = DeepFace.represent(
            img_path=str(temp_path),
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False
        )
        return True, embedding
    except Exception as e:
        return False, str(e)
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def rebuild_deepface_database():
    """
    Rebuild the DeepFace database after processing images.
    This creates the pickle file used for face recognition.
    """
    logger.info("Rebuilding DeepFace database...")
    
    # Count how many images are in the database before rebuilding
    total_images = 0
    total_celebrities = 0
    for root, dirs, files in os.walk(OUTPUT_PATH):
        if root == str(OUTPUT_PATH):
            # Count celebrities (directories at the top level)
            total_celebrities = len([d for d in dirs if not d.startswith('.')])
        else:
            # Count image files
            jpg_files = [f for f in files if f.lower().endswith('.jpg')]
            total_images += len(jpg_files)
    
    logger.info(f"Database contains {total_images} images across {total_celebrities} celebrities")
    
    try:
        # Create a dummy image for the find function
        temp_img_path = "/tmp/temp_rebuild_face.jpg"
        blank_img = np.ones((100, 100, 3), dtype=np.uint8) * 200  # Light gray
        # Add facial features for better detection
        cv2.circle(blank_img, (35, 40), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(blank_img, (65, 40), 10, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(blank_img, (50, 70), (20, 10), 0, 0, 180, (0, 0, 0), -1)  # Mouth
        cv2.imwrite(temp_img_path, blank_img)
        
        try:
            # Call find with refresh_database=True to create the representations
            result = DeepFace.find(
                img_path=temp_img_path,
                db_path=str(OUTPUT_PATH),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                distance_metric="cosine",
                silent=True,
                refresh_database=True
            )
            
            logger.info("Successfully rebuilt DeepFace database")
            
            # Verify the pickle file was created
            file_parts = [
                "ds", 
                "model", 
                MODEL_NAME.lower(), 
                "detector", 
                DETECTOR_BACKEND,
                "aligned",
                "normalization", 
                "base",
                "expand", 
                "0"
            ]
            file_name = "_".join(file_parts) + ".pkl"
            file_name = file_name.replace("-", "").lower()
            
            pkl_path = OUTPUT_PATH / file_name
            if os.path.exists(pkl_path):
                # Get file size
                file_size_bytes = os.path.getsize(pkl_path)
                file_size_mb = file_size_bytes / (1024 * 1024)
                logger.info(f"Verified pickle file: {pkl_path} (Size: {file_size_mb:.2f} MB)")
            else:
                logger.warning("Pickle file not found after database rebuild")
                
                # Check for any pickle files
                all_files = os.listdir(OUTPUT_PATH)
                pkl_files = [f for f in all_files if f.endswith('.pkl')]
                if pkl_files:
                    logger.info(f"Found alternative pickle files: {pkl_files}")
                    
        except Exception as e:
            logger.error(f"Error during DeepFace.find: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error rebuilding DeepFace database: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception:
                pass

def create_celebrity_database(limit=None, max_per_celeb=None, debug_save=DEBUG_SAVE,
                             detect_face=DETECT_FACE, gender_check=GENDER_CHECK,
                             gender_threshold=GENDER_THRESHOLD, crop_margin=CROP_MARGIN,
                             expand_coordinates=EXPAND_COORDINATES):
    """
    Create celebrity database with configurable parameters.
    
    Args:
        limit: Maximum number of entries to process
        max_per_celeb: Maximum images per celebrity
        debug_save: Whether to save debug images
        detect_face: Whether to perform face detection
        gender_check: Whether to verify gender
        gender_threshold: Minimum confidence for gender verification
        crop_margin: Margin around the face crop (0 to disable)
        expand_coordinates: Expansion of raw MATLAB coordinates (0 to disable)
    """
    # Use the provided max_per_celeb or the default
    if max_per_celeb is None:
        max_per_celeb = MAX_IMAGES_PER_CELEB
        
    # Create output directories
    imdb, idxs = load_imdb_raw(limit)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    if debug_save:
        DEBUG_PATH.mkdir(parents=True, exist_ok=True)
        (DEBUG_PATH / "full").mkdir(exist_ok=True)
        (DEBUG_PATH / "coarse").mkdir(exist_ok=True)
    
    # Create temp directory
    os.makedirs(TEMP_DIR, exist_ok=True)

    logs, counts = [], {}
    saved = 0

    for i in tqdm(idxs, unit='img'):
        entry = {'index': i}
        try:
            fp = _resolve_path(imdb['full_path'][0, i]); entry['full_path'] = fp
            name = _scalar(imdb['name'][0, i]); entry['name'] = name
            loc = _scalar(imdb['face_location'][0, i])
            fs = float(imdb['face_score'][0, i]); entry['face_score'] = fs
            sfs = _scalar(imdb['second_face_score'][0, i]); entry['second_face_score'] = sfs
            
            # Get gender from MATLAB data (0=female, 1=male, NaN=unknown, or directly 'male'/'female')
            gender_val = imdb['gender'][0, i]
            
            # Handle different possible formats of the gender data
            if isinstance(gender_val, np.ndarray) and gender_val.size > 0:
                gender_val = gender_val[0]  # Extract first element if array
                
            # Handle string type gender
            if isinstance(gender_val, str) or (isinstance(gender_val, bytes) and gender_val):
                gender_str = gender_val.decode() if isinstance(gender_val, bytes) else gender_val
                if gender_str.lower() == 'female':
                    gender = GENDER_FEMALE
                elif gender_str.lower() == 'male':
                    gender = GENDER_MALE
                else:
                    gender = None
            # Handle numeric type gender
            elif isinstance(gender_val, (int, float, np.number)):
                try:
                    gender_num = float(gender_val)
                    if not np.isnan(gender_num):
                        if gender_num == GENDER_FEMALE:
                            gender = GENDER_FEMALE
                        elif gender_num == GENDER_MALE:
                            gender = GENDER_MALE
                        else:
                            gender = None
                    else:
                        gender = None
                except (ValueError, TypeError):
                    gender = None
            else:
                gender = None
                
            # Set gender string for display
            entry['gender'] = 'female' if gender == GENDER_FEMALE else ('male' if gender == GENDER_MALE else 'unknown')
            
            status, detect_status = 'skipped', None

            if fs == -np.inf:
                status = 'no_face'
            elif fs < FACE_SCORE_THRESHOLD:
                status = 'low_score'
            else:
                label = normalize_text(name)
                if not label:
                    status = 'bad_name'
                elif counts.get(label, 0) >= max_per_celeb:
                    status = 'limit'
                else:
                    # Read the full image
                    img_path = IMDB_ROOT / fp
                    if not os.path.exists(img_path):
                        status = 'file_not_found'
                        entry.update({'status': status})
                        logs.append(entry)
                        continue
                        
                    img_full = cv2.imread(str(img_path))
                    if img_full is None:
                        status = 'read_fail'
                        entry.update({'status': status})
                        logs.append(entry)
                        continue
                    
                    # debug: save full image
                    if debug_save:
                        try:
                            cv2.imwrite(str(DEBUG_PATH / "full" / f"full_{i:06d}.jpg"), img_full)
                        except Exception:
                            pass
                    
                    # Adaptive crop based on MATLAB bbox
                    coarse = adaptive_crop(img_full, loc, margin=crop_margin, expand=expand_coordinates)
                    
                    # debug: save coarse crop
                    if debug_save:
                        try:
                            cv2.imwrite(str(DEBUG_PATH / "coarse" / f"coarse_{i:06d}.jpg"), coarse)
                        except Exception:
                            pass
                    
                    # Face verification - using temporary file approach with quality checks
                    if detect_face:
                        has_face, error_msg, face_details = verify_face_with_file(coarse, i)
                        if has_face:
                            # Face passed quality checks
                            detect_status = 'detected'
                            if face_details and 'confidence' in face_details:
                                # Add face detection confidence to the entry
                                entry['face_confidence'] = float(face_details['confidence'])
                            if face_details and 'facial_area' in face_details:
                                # Add facial area details to the entry
                                facial_area = face_details['facial_area']
                                
                                # Handle different formats of facial_area
                                if isinstance(facial_area, dict):
                                    try:
                                        x = facial_area.get('x', 0)
                                        y = facial_area.get('y', 0)
                                        w = facial_area.get('w', 0)
                                        h = facial_area.get('h', 0)
                                    except Exception:
                                        # Alternative approach if direct access doesn't work
                                        facial_area_values = list(facial_area.values())
                                        if len(facial_area_values) >= 4:
                                            x, y, w, h = facial_area_values[:4]
                                        else:
                                            x, y, w, h = 0, 0, 0, 0
                                elif isinstance(facial_area, (list, tuple)) and len(facial_area) >= 4:
                                    # List/tuple format
                                    x, y, w, h = facial_area[:4]
                                else:
                                    # Unknown format, use defaults
                                    x, y, w, h = 0, 0, 0, 0
                                    
                                entry['face_width'] = w
                                entry['face_height'] = h
                                entry['face_area'] = w * h
                                entry['face_ratio'] = (w * h) / (coarse.shape[0] * coarse.shape[1])
                        else:
                            # Face detection or quality check failed
                            status = 'no_face_refine'
                            detect_status = f'error:{error_msg}' if error_msg else 'none' 
                            
                            # Save debug info about rejected face
                            if face_details and 'facial_area' in face_details:
                                # Add facial area details to the entry
                                facial_area = face_details['facial_area']
                                
                                # Handle different formats of facial_area
                                if isinstance(facial_area, dict):
                                    try:
                                        x = facial_area.get('x', 0)
                                        y = facial_area.get('y', 0)
                                        w = facial_area.get('w', 0)
                                        h = facial_area.get('h', 0)
                                    except Exception:
                                        # Alternative approach if direct access doesn't work
                                        facial_area_values = list(facial_area.values())
                                        if len(facial_area_values) >= 4:
                                            x, y, w, h = facial_area_values[:4]
                                        else:
                                            x, y, w, h = 0, 0, 0, 0
                                elif isinstance(facial_area, (list, tuple)) and len(facial_area) >= 4:
                                    # List/tuple format
                                    x, y, w, h = facial_area[:4]
                                else:
                                    # Unknown format, use defaults
                                    x, y, w, h = 0, 0, 0, 0
                                
                                entry['rejected_face_width'] = w
                                entry['rejected_face_height'] = h
                                entry['rejected_face_area'] = w * h
                                h_img, w_img = coarse.shape[:2]
                                entry['rejected_face_ratio'] = (w * h) / (w_img * h_img)
                                
                            entry.update({'status': status, 'detect_status': detect_status})
                            logs.append(entry)
                            continue
                    
                    # Gender verification - only if gender_check is enabled and gender is known
                    if gender_check and gender is not None and not np.isnan(gender):
                        gender_match, confidence, error_msg, gender_details = verify_gender(
                            coarse, i, gender, threshold=gender_threshold
                        )
                        # Add all gender details to the entry
                        entry['gender_confidence'] = confidence
                        entry['detected_gender'] = gender_details['detected_gender']
                        entry['woman_score'] = gender_details['woman_score']
                        entry['man_score'] = gender_details['man_score']
                        entry['gender_diff'] = abs(gender_details['woman_score'] - gender_details['man_score'])
                        
                        if not gender_match:
                            status = 'gender_mismatch'
                            detect_status = f'expected:{entry["gender"]}, detected:{gender_details["detected_gender"]}, confidence:{confidence:.2f}'
                            entry.update({'status': status, 'detect_status': detect_status})
                            logs.append(entry)
                            continue
                    
                    # Generate embeddings - using temporary file approach
                    success, result = generate_embedding(coarse, i)
                    if success:
                        # Save the coarse crop as our final result
                        out_dir = OUTPUT_PATH / label
                        out_dir.mkdir(exist_ok=True)
                        fname = f"{label}_{i:06d}.jpg"
                        
                        try:
                            cv2.imwrite(str(out_dir / fname), coarse)
                            counts[label] = counts.get(label, 0) + 1
                            saved += 1
                            status = 'saved'
                            entry['saved_path'] = str(out_dir)
                            entry['saved_name'] = fname
                        except Exception as e:
                            status = 'save_fail'
                            detect_status = f'save_error:{str(e)}'
                            entry.update({'status': status, 'detect_status': detect_status})
                            logs.append(entry)
                            continue
                    else:
                        status = 'embed_fail'
                        detect_status = f'embed_error:{result}'
                        entry.update({'status': status, 'detect_status': detect_status})
                        logs.append(entry)
                        continue
            
            entry['status'] = status
            entry['detect_status'] = detect_status
            logs.append(entry)
        except Exception as e:
            # Catch any unexpected errors
            entry['status'] = 'error'
            entry['detect_status'] = f'error:{str(e)}'
            logs.append(entry)
        
        # Save logs periodically
        if i % 100 == 0 and logs:
            try:
                pd.DataFrame(logs).to_csv(LOG_FILE, index=False)
            except Exception:
                pass

    # Final save of logs
    try:
        pd.DataFrame(logs).to_csv(LOG_FILE, index=False)
    except Exception as e:
        logger.error(f"Error saving log file: {str(e)}")
    
    logger.info("Saved %d images across %d celebrities", saved, len(counts))
    logger.info("Log file: %s", LOG_FILE)
    
    # Clean up temp directory
    try:
        import shutil
        shutil.rmtree(TEMP_DIR)
    except Exception:
        pass
    
    # Rebuild the DeepFace database to create the pickle file
    rebuild_deepface_database()

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("IMDB Celeb Importer")
    parser.add_argument('--limit', type=int, help='max entries')
    parser.add_argument('--max_per_celeb', type=int,
                        default=MAX_IMAGES_PER_CELEB,
                        help='max per celeb')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug saves')
    parser.add_argument('--no-detect', action='store_true',
                        help='Skip face detection (not recommended)')
    parser.add_argument('--no-gender', action='store_true',
                        help='Skip gender verification')
    parser.add_argument('--gender-threshold', type=float,
                        default=GENDER_THRESHOLD,
                        help='Gender confidence threshold (0-1)')
    parser.add_argument('--crop-margin', type=float,
                        default=CROP_MARGIN,
                        help='Margin around face crop (0 to disable)')
    parser.add_argument('--expand-coordinates', type=float,
                        default=EXPAND_COORDINATES,
                        help='Expansion factor for MATLAB coordinates (0 to disable)')
    args = parser.parse_args()
    
    # Create celebrity database with command line argument settings
    create_celebrity_database(
        limit=args.limit,
        max_per_celeb=args.max_per_celeb,
        debug_save=args.debug,
        detect_face=not args.no_detect,
        gender_check=not args.no_gender,
        gender_threshold=args.gender_threshold,
        crop_margin=args.crop_margin,
        expand_coordinates=args.expand_coordinates
    )

if __name__ == '__main__':
    main()
