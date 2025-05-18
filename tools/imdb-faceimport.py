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
MAX_IMAGES_PER_CELEB = 10
FACE_SCORE_THRESHOLD = 3.0
CROP_MARGIN          = 0.25   # Padding around MATLAB bbox
DETECT_FACE          = True   # If False, skip face detection gating
GENDER_CHECK         = True   # If True, verify gender against MATLAB data
GENDER_THRESHOLD     = 0.70   # Minimum confidence for gender verification
TEMP_DIR             = Path("/tmp/deepface_temp")  # Temp directory for image files

# Gender constants (from MATLAB file)
GENDER_FEMALE = 0
GENDER_MALE = 1

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


def adaptive_crop(img, loc, margin=CROP_MARGIN, min_size=50):
    """
    Adaptively crop a face based on MATLAB bbox, handling invalid coordinates gracefully.
    
    If coordinates are invalid:
    1. First tries to scale them to fit within the image
    2. If that's not possible, defaults to using the entire image
    
    Returns the cropped image (never None).
    """
    h, w = img.shape[:2]
    
    # Default to full image
    x1, y1, x2, y2 = 0, 0, w, h
    
    # Try to use MATLAB coordinates
    bbox = parse_bbox(loc)
    if bbox:
        mx1, my1, mx2, my2 = bbox
        
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
        
        # Check if crop is too small
        if x2 - x1 < min_size or y2 - y1 < min_size:
            # Fall back to full image
            x1, y1, x2, y2 = 0, 0, w, h
    
    # Apply margin
    crop_w = x2 - x1
    crop_h = y2 - y1
    pad_w = int(crop_w * margin)
    pad_h = int(crop_h * margin)
    
    xa = max(0, x1 - pad_w)
    ya = max(0, y1 - pad_h)
    xb = min(w, x2 + pad_w)
    yb = min(h, y2 + pad_h)
    
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

def verify_face_with_file(img, img_id):
    """
    Use DeepFace with temporary file to avoid "Invalid image input" errors.
    Returns (success, error_message).
    """
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Save the image to a temporary file
    temp_path = TEMP_DIR / f"temp_{img_id}.jpg"
    try:
        success = cv2.imwrite(str(temp_path), img)
        if not success:
            return False, "Failed to save temporary image"
    except Exception as e:
        return False, f"Error saving temporary image: {str(e)}"
    
    try:
        # First try with enforce_detection=True
        try:
            faces = DeepFace.extract_faces(
                img_path=str(temp_path),
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True
            )
            if faces and len(faces) > 0:
                return True, None
        except Exception:
            # Fall back to enforce_detection=False
            try:
                faces = DeepFace.extract_faces(
                    img_path=str(temp_path),
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=True
                )
                if faces and len(faces) > 0:
                    return True, None
            except Exception as e2:
                return False, str(e2)
                
        # If we get here with no faces, no face was detected
        return False, "No face detected"
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

def verify_gender(img, img_id, expected_gender):
    """
    Verify that the detected gender matches the expected gender.
    
    Args:
        img: Image to analyze
        img_id: ID for temporary file
        expected_gender: GENDER_MALE (1) or GENDER_FEMALE (0)
    
    Returns:
        (success, confidence, error_message)
    """
    # Skip if expected_gender is not valid
    if expected_gender is None or np.isnan(expected_gender):
        return True, 1.0, None  # Skip verification if gender is unknown
        
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Save the image to a temporary file
    temp_path = TEMP_DIR / f"gender_{img_id}.jpg"
    try:
        success = cv2.imwrite(str(temp_path), img)
        if not success:
            return False, 0.0, "Failed to save temporary image for gender verification"
    except Exception as e:
        return False, 0.0, f"Error saving temporary image for gender verification: {str(e)}"
    
    try:
        # Analyze the image with DeepFace
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
                return False, 0.0, "Gender not detected"
                
            # Get the dominant gender
            if isinstance(detected_gender, dict):
                # For DeepFace v1
                woman_score = detected_gender.get('Woman', 0)
                man_score = detected_gender.get('Man', 0)
                
                # Compare with expected gender
                if expected_gender == GENDER_FEMALE:
                    return woman_score >= GENDER_THRESHOLD, woman_score, None
                else:
                    return man_score >= GENDER_THRESHOLD, man_score, None
            else:
                # For other DeepFace versions
                return detected_gender.lower() == ('female' if expected_gender == GENDER_FEMALE else 'male'), 1.0, None
                
        except Exception as e:
            return False, 0.0, f"Gender analysis failed: {str(e)}"
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

def create_celebrity_database(limit=None, max_per_celeb=None):
    global MAX_IMAGES_PER_CELEB
    if max_per_celeb:
        MAX_IMAGES_PER_CELEB = max_per_celeb
        
    # Create output directories
    imdb, idxs = load_imdb_raw(limit)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    if DEBUG_SAVE:
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
            
            # Get gender from MATLAB data (0=female, 1=male, NaN=unknown)
            gender = imdb['gender'][0, i]
            if isinstance(gender, np.ndarray) and gender.size > 0:
                gender = float(gender[0])
            else:
                gender = None
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
                elif counts.get(label, 0) >= MAX_IMAGES_PER_CELEB:
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
                    if DEBUG_SAVE:
                        try:
                            cv2.imwrite(str(DEBUG_PATH / "full" / f"full_{i:06d}.jpg"), img_full)
                        except Exception:
                            pass
                    
                    # Adaptive crop based on MATLAB bbox
                    coarse = adaptive_crop(img_full, loc)
                    
                    # debug: save coarse crop
                    if DEBUG_SAVE:
                        try:
                            cv2.imwrite(str(DEBUG_PATH / "coarse" / f"coarse_{i:06d}.jpg"), coarse)
                        except Exception:
                            pass
                    
                    # Face verification - using temporary file approach
                    if DETECT_FACE:
                        has_face, error_msg = verify_face_with_file(coarse, i)
                        if has_face:
                            detect_status = 'detected'
                        else:
                            status = 'no_face_refine'
                            detect_status = f'error:{error_msg}' if error_msg else 'none' 
                            entry.update({'status': status, 'detect_status': detect_status})
                            logs.append(entry)
                            continue
                    
                    # Gender verification - only if GENDER_CHECK is enabled and gender is known
                    if GENDER_CHECK and gender is not None and not np.isnan(gender):
                        gender_match, confidence, error_msg = verify_gender(coarse, i, gender)
                        entry['gender_confidence'] = confidence
                        
                        if not gender_match:
                            status = 'gender_mismatch'
                            detect_status = f'expected:{entry["gender"]}, confidence:{confidence:.2f}'
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

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global DEBUG_SAVE, DETECT_FACE, GENDER_CHECK, GENDER_THRESHOLD
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
    args = parser.parse_args()
    
    # Set global variables based on args
    DEBUG_SAVE = args.debug
    DETECT_FACE = not args.no_detect
    GENDER_CHECK = not args.no_gender
    
    # Only try to update GENDER_THRESHOLD if explicitly provided
    if args.gender_threshold:
        GENDER_THRESHOLD = args.gender_threshold
    
    create_celebrity_database(args.limit, args.max_per_celeb)

if __name__ == '__main__':
    main()
