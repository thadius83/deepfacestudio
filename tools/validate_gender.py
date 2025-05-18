#!/usr/bin/env python3
"""
IMDB Gender Classification Validation Tool

A utility for testing gender classification on the IMDB dataset.
Samples random entries from the MATLAB file and tests gender classification.

Usage:
    python validate_gender.py --samples 50 --save-images --output-dir ./debug_gender
"""
import os
import sys
import logging
import argparse
import random
from pathlib import Path
import csv
import time

import numpy as np
import pandas as pd
import scipy.io
import cv2
from deepface import DeepFace
from tqdm import tqdm

# Configuration
DATA_PATH = Path("/workspace/imdb")
IMDB_ROOT = DATA_PATH / "imdb_crop"
MAT_FILE = IMDB_ROOT / "imdb.mat"
DEFAULT_OUTPUT_DIR = DATA_PATH / ".debug_gender"
TEMP_DIR = Path("/tmp/deepface_temp")

# Gender constants
GENDER_FEMALE = 0
GENDER_MALE = 1

# DeepFace configuration
DETECTOR_BACKEND = "retinaface"
MODEL_NAME = "Facenet"
GENDER_THRESHOLD = 0.60

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def convert_to_scalar(x):
    """Convert numpy arrays to scalars."""
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        if x.size == 1:
            return convert_to_scalar(x.item())
        return [convert_to_scalar(el) for el in x]
    if isinstance(x, bytes):
        return x.decode()
    return x

def resolve_path(p):
    """Resolve path from MATLAB file."""
    if isinstance(p, (str, bytes)):
        return p.decode() if isinstance(p, bytes) else p
    if isinstance(p, np.ndarray):
        return resolve_path(p[0]) if p.size else ""
    return str(p)

def parse_bbox(loc):
    """
    Parse bounding box coordinates from MATLAB format.
    
    In MATLAB, the face_location is used as:
    img(face_location(2):face_location(4),face_location(1):face_location(3),:))
    
    MATLAB indexing is different from Python:
    - face_location(1) is column start (x-min)
    - face_location(2) is row start (y-min)
    - face_location(3) is column end (x-max)
    - face_location(4) is row end (y-max)
    
    But MATLAB indexing runs row,column while Python is y,x
    """
    while isinstance(loc, list) and len(loc) == 1:
        loc = loc[0]
    if not isinstance(loc, (list, tuple)) or len(loc) < 4:
        return None
    try:
        # Extract raw values from loc
        col_min, row_min, col_max, row_max = [float(loc[i]) for i in range(4)]
        
        # Convert to Python format (x1,y1,x2,y2)
        x1 = int(col_min)
        y1 = int(row_min) 
        x2 = int(col_max)
        y2 = int(row_max)
        
        # Safety check for invalid coordinates (x1 > x2 or y1 > y2)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Make sure we have positive width and height
        if x2 - x1 <= 0:
            x2 = x1 + 1
        if y2 - y1 <= 0:
            y2 = y1 + 1
            
        return x1, y1, x2, y2
    except Exception as e:
        logger.warning(f"Error parsing bbox: {e}, loc: {loc}")
        return None

def adaptive_crop(img, loc, margin=0.55, expand=0.0, min_size=20, min_image_size=250):
    """
    Adaptively crop a face based on MATLAB bbox, handling invalid coordinates gracefully.
    
    Args:
        img: Image to crop
        loc: Face location from MATLAB
        margin: Padding around the final bbox (0 to disable)
        expand: Expansion factor for raw MATLAB coordinates (0 to disable)
        min_size: Minimum size for a valid crop
        min_image_size: Don't crop images smaller than this size
    
    If coordinates are invalid:
    1. First tries to scale them to fit within the image
    2. If that's not possible, defaults to using the entire image
    
    Returns the cropped image and a metadata dictionary.
    """
    h, w = img.shape[:2]
    
    # Default to full image
    x1, y1, x2, y2 = 0, 0, w, h
    
    # Track metadata for diagnostics
    metadata = {
        'source': 'full_image',
        'original_width': w,
        'original_height': h,
        'matlab_valid': False,
        'raw_matlab_coords': str(loc) if loc else "None"
    }
    
    # If source image is small, don't crop it
    if w < min_image_size or h < min_image_size:
        metadata['source'] = 'small_image_no_crop'
        metadata['final_width'] = w
        metadata['final_height'] = h
        metadata['final_aspect_ratio'] = w / max(1, h)
        return img.copy(), metadata
    
    # Try to use MATLAB coordinates
    bbox = parse_bbox(loc)
    if bbox:
        mx1, my1, mx2, my2 = bbox
        metadata['matlab_valid'] = True
        metadata['matlab_x1'] = mx1
        metadata['matlab_y1'] = my1
        metadata['matlab_x2'] = mx2
        metadata['matlab_y2'] = my2
        
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
            
            metadata['expansion_applied'] = True
        
        # Check if we need to scale the coordinates
        scale_needed = False
        if mx1 >= w or my1 >= h or mx2 <= 0 or my2 <= 0 or mx1 >= mx2 or my1 >= my2:
            # These coordinates are clearly outside the image or invalid
            scale_needed = True
            metadata['scaling_reason'] = 'out_of_bounds'
        elif mx2 > w * 1.5 or my2 > h * 1.5:
            # Coordinates are much larger than the image - likely a scale issue
            scale_needed = True
            metadata['scaling_reason'] = 'too_large'
            
        if scale_needed:
            # Try to scale the coordinates
            metadata['scaling_applied'] = True
            
            # First, get the source dimensions from the coordinates
            src_w = mx2 - mx1
            src_h = my2 - my1
            
            # Save original dimensions for debugging
            metadata['original_src_width'] = src_w
            metadata['original_src_height'] = src_h
            
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
                
                metadata['scale_factor'] = scale
                metadata['scaled_width'] = mx2 - mx1
                metadata['scaled_height'] = my2 - my1
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(w-1, mx1))
        y1 = max(0, min(h-1, my1))
        x2 = max(x1+1, min(w, mx2))
        y2 = max(y1+1, min(h, my2))
        
        # Check if crop is too small or has unusual aspect ratio
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # Calculate aspect ratio (width/height) - make sure it's correct
        if crop_height > 0:
            aspect_ratio = crop_width / crop_height
        else:
            aspect_ratio = 0
            
        # Store calculated dimensions in metadata for debugging
        metadata['calculated_width'] = crop_width
        metadata['calculated_height'] = crop_height
            
        # More permissive aspect ratio thresholds
        aspect_ratio_min = 0.6  # Allow more rectangular crops
        aspect_ratio_max = 1.67  # Allow more rectangular crops
        
        if crop_width < min_size or crop_height < min_size:
            # Fall back to full image if dimensions are too small
            metadata['too_small'] = True
            metadata['crop_size'] = f"{crop_width}x{crop_height}"
            metadata['fallback_reason'] = 'dimensions_too_small'
            x1, y1, x2, y2 = 0, 0, w, h
        elif aspect_ratio < aspect_ratio_min or aspect_ratio > aspect_ratio_max:
            # Fall back to full image if aspect ratio is too far from square
            metadata['unusual_aspect_ratio'] = True
            metadata['aspect_ratio'] = aspect_ratio
            metadata['fallback_reason'] = 'non_square_crop'
            metadata['crop_size'] = f"{crop_width}x{crop_height}"
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            metadata['source'] = 'matlab_coords'
            metadata['crop_width'] = crop_width
            metadata['crop_height'] = crop_height
            metadata['aspect_ratio'] = aspect_ratio
    
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
        
        metadata['margin_applied'] = True
    else:
        # No margin
        xa = int(x1)
        ya = int(y1)
        xb = int(x2)
        yb = int(y2)
        metadata['margin_applied'] = False
    
    # Final dimensions
    metadata['final_width'] = xb - xa
    metadata['final_height'] = yb - ya
    if yb - ya > 0:  # Prevent division by zero
        metadata['final_aspect_ratio'] = (xb - xa) / (yb - ya)
    else:
        metadata['final_aspect_ratio'] = 1.0
    
    # Create crop with copy to ensure it's not a view
    try:
        cropped = img[ya:yb, xa:xb].copy()
        return cropped, metadata
    except Exception as e:
        logger.error(f"Error creating crop: {e}, dimensions: [{ya}:{yb}, {xa}:{xb}]")
        # Last resort fallback to full image
        return img.copy(), {**metadata, 'source': 'error_fallback', 'error': str(e)}

def load_imdb_data():
    """Load MATLAB file with IMDB dataset."""
    logger.info(f"Loading MATLAB file from {MAT_FILE}")
    try:
        mat = scipy.io.loadmat(str(MAT_FILE))
        imdb = mat['imdb'][0, 0]
        total_entries = imdb['full_path'][0].shape[0]
        logger.info(f"Successfully loaded {total_entries} entries from MATLAB file")
        return imdb, total_entries
    except Exception as e:
        logger.error(f"Failed to load MATLAB file: {e}")
        raise

def get_gender_from_matlab(imdb, idx):
    """Extract gender from MATLAB data, handling different formats."""
    try:
        gender_val = imdb['gender'][0, idx]
        
        # Handle array type
        if isinstance(gender_val, np.ndarray) and gender_val.size > 0:
            gender_val = gender_val[0]
            
        # Handle string type
        if isinstance(gender_val, str) or (isinstance(gender_val, bytes) and gender_val):
            gender_str = gender_val.decode() if isinstance(gender_val, bytes) else gender_val
            if gender_str.lower() == 'female':
                return GENDER_FEMALE, 'female'
            elif gender_str.lower() == 'male':
                return GENDER_MALE, 'male'
            
        # Handle numeric type
        elif isinstance(gender_val, (int, float, np.number)):
            try:
                gender_num = float(gender_val)
                if not np.isnan(gender_num):
                    if gender_num == GENDER_FEMALE:
                        return GENDER_FEMALE, 'female'
                    elif gender_num == GENDER_MALE:
                        return GENDER_MALE, 'male'
            except (ValueError, TypeError):
                pass
                
        # Unknown gender
        return None, 'unknown'
    except Exception as e:
        logger.warning(f"Error getting gender for index {idx}: {e}")
        return None, 'unknown'

def select_samples(imdb, total, num_samples, min_face_score=3.0):
    """Select random samples with valid gender information."""
    logger.info(f"Selecting {num_samples} random samples...")
    
    # Create candidate pool (3x the requested number)
    candidates = random.sample(range(total), min(total, num_samples * 5))
    valid_samples = []
    
    # Track sample information for debugging
    gender_counts = {'male': 0, 'female': 0, 'unknown': 0}
    file_exists_count = 0
    low_score_count = 0
    
    for idx in candidates:
        try:
            # Check face score
            fs = float(imdb['face_score'][0, idx])
            if fs == -np.inf or fs < min_face_score:
                low_score_count += 1
                continue
                
            # Check gender
            gender, gender_str = get_gender_from_matlab(imdb, idx)
            gender_counts[gender_str] += 1
            
            if gender is None:
                continue
                
            # Check if file exists
            file_path = resolve_path(imdb['full_path'][0, idx])
            img_path = IMDB_ROOT / file_path
            if not os.path.exists(img_path):
                continue
                
            file_exists_count += 1
            valid_samples.append(idx)
            
            # If we have enough samples, stop
            if len(valid_samples) >= num_samples:
                break
                
        except Exception as e:
            logger.warning(f"Error processing candidate {idx}: {e}")
    
    # Shuffle and limit to requested number
    random.shuffle(valid_samples)
    valid_samples = valid_samples[:num_samples]
    
    # Log sample selection stats
    logger.info(f"Gender distribution in candidates: {gender_counts}")
    logger.info(f"Files found: {file_exists_count}, Low score entries: {low_score_count}")
    logger.info(f"Selected {len(valid_samples)} valid samples for validation")
    
    # Log a few examples for debugging
    if valid_samples:
        for i, idx in enumerate(valid_samples[:3]):
            gender, gender_str = get_gender_from_matlab(imdb, idx)
            fs = float(imdb['face_score'][0, idx])
            logger.info(f"Sample {i}: Index {idx}, Gender: {gender_str}, Face Score: {fs:.2f}")
    
    return valid_samples

def detect_gender(img, expected_gender, crop_metadata=None):
    """
    Run gender detection on an image with robust error handling.
    
    Args:
        img: Image to analyze
        expected_gender: Expected gender (GENDER_MALE or GENDER_FEMALE)
        crop_metadata: Optional metadata about the cropping process
        
    Returns:
        Dictionary with results
    """
    # Initialize results
    result = {
        'expected_gender': 'female' if expected_gender == GENDER_FEMALE else 'male',
        'detected_gender': 'unknown',
        'woman_score': 0.0,
        'man_score': 0.0,
        'gender_match': False,
        'detection_error': None
    }
    
    # Add crop metadata if provided
    if crop_metadata:
        for key, value in crop_metadata.items():
            result[f'crop_{key}'] = value
    
    # Validate image dimensions - prevent OpenCV errors
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        result['detection_error'] = f"Invalid image dimensions: {w}x{h}"
        return result
    
    # Create temp directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Save image to temporary file
    temp_id = int(time.time() * 1000)
    temp_path = TEMP_DIR / f"gender_{temp_id}.jpg"
    
    try:
        # Ensure image is in proper format
        if img.dtype == np.float64 or img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
            
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
        # Save to temp file
        if not cv2.imwrite(str(temp_path), img):
            result['detection_error'] = "Failed to save temp image"
            return result
        
        # Run DeepFace analysis with error handling for cv2 resize errors
        try:
            analysis = DeepFace.analyze(
                img_path=str(temp_path),
                actions=['gender'],
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                silent=True
            )
            
            # Extract results
            if isinstance(analysis, list):
                analysis = analysis[0]
                
            gender_data = analysis.get('gender', None)
            
            # Process results based on format
            if isinstance(gender_data, dict):
                # DeepFace v1 style output
                woman_score = gender_data.get('Woman', 0)
                man_score = gender_data.get('Man', 0)
                
                result['woman_score'] = float(woman_score)
                result['man_score'] = float(man_score)
                result['detected_gender'] = 'female' if woman_score > man_score else 'male'
                
            elif gender_data is not None:
                # String output
                detected = gender_data.lower()
                result['detected_gender'] = detected
                
                # Set approximate scores
                if detected == 'female':
                    result['woman_score'] = 0.9
                    result['man_score'] = 0.1
                else:
                    result['woman_score'] = 0.1
                    result['man_score'] = 0.9
            
            # Determine if there's a match
            expected = 'female' if expected_gender == GENDER_FEMALE else 'male'
            result['gender_match'] = result['detected_gender'] == expected
            
        except cv2.error as e:
            # Handle OpenCV errors specifically
            result['detection_error'] = f"OpenCV error: {str(e)}"
            logger.warning(f"OpenCV error in gender detection: {e}")
            return result
            
        except Exception as e:
            # Try alternative approach with OpenCV detector if primary fails
            try:
                alternative_analysis = DeepFace.analyze(
                    img_path=str(temp_path),
                    actions=['gender'],
                    detector_backend="opencv",  # Try opencv as fallback
                    enforce_detection=False,
                    silent=True
                )
                
                # Extract results from alternative analysis
                if isinstance(alternative_analysis, list):
                    alternative_analysis = alternative_analysis[0]
                    
                gender_data = alternative_analysis.get('gender', None)
                
                if gender_data is not None:
                    # For string output
                    detected = gender_data.lower()
                    result['detected_gender'] = detected
                    
                    # Set approximate scores
                    if detected == 'female':
                        result['woman_score'] = 0.9
                        result['man_score'] = 0.1
                    else:
                        result['woman_score'] = 0.1
                        result['man_score'] = 0.9
                        
                    # Determine if there's a match
                    expected = 'female' if expected_gender == GENDER_FEMALE else 'male'
                    result['gender_match'] = result['detected_gender'] == expected
                else:
                    result['detection_error'] = "Gender not detected in alternative approach"
                    
            except cv2.error as e2:
                # Handle OpenCV errors in alternative approach
                result['detection_error'] = f"OpenCV error in fallback: {str(e2)}"
                logger.warning(f"OpenCV error in fallback gender detection: {e2}")
                
            except Exception as e2:
                # Handle other errors in alternative approach
                result['detection_error'] = f"Error in gender detection: {str(e)} / {str(e2)}"
                logger.warning(f"Gender detection error: {e} / {e2}")
        
    finally:
        # Clean up
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
            
    return result

def save_debug_image(output_dir, index, img, expected_gender, gender_result, original_img=None, name=None):
    """Save debug image with gender information."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with name for identification
    name_part = ""
    if name:
        # Normalize name for filename
        safe_name = name.replace(' ', '_').replace('/', '_').replace('\\', '_')[:30]
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
        name_part = f"_{safe_name}"
    
    # Save cropped image
    try:
        cv2.imwrite(f"{output_dir}/cropped_{index:06d}{name_part}.jpg", img)
    except Exception as e:
        logger.warning(f"Error saving cropped image: {e}")
        return
    
    # Save original image if provided
    if original_img is not None:
        try:
            cv2.imwrite(f"{output_dir}/original_{index:06d}{name_part}.jpg", original_img)
        except Exception as e:
            logger.warning(f"Error saving original image: {e}")
    
    # Create annotated version
    h, w = img.shape[:2]
    annotated = img.copy()
    
    # Color based on match/mismatch
    match = gender_result['gender_match']
    color = (0, 255, 0) if match else (0, 0, 255)  # Green=match, Red=mismatch
    
    # Add border
    border = max(5, min(w, h) // 20)
    annotated = cv2.copyMakeBorder(
        annotated, border, border, border, border, 
        cv2.BORDER_CONSTANT, value=color
    )
    
    # Add text with gender info
    expected = gender_result['expected_gender']
    detected = gender_result['detected_gender']
    woman_score = gender_result['woman_score']
    man_score = gender_result['man_score']
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(w, h) / 400)
    thickness = max(1, int(scale * 2))
    
    # Add expected gender
    cv2.putText(
        annotated, f"Expected: {expected}",
        (border, h + int(border * 1.5)),
        font, scale, (255, 255, 255), thickness
    )
    
    # Add detected gender with scores
    cv2.putText(
        annotated, f"Detected: {detected} (F:{woman_score:.2f}, M:{man_score:.2f})",
        (border, h + int(border * 3)),
        font, scale, (255, 255, 255), thickness
    )
    
    # Save annotated image
    try:
        cv2.imwrite(f"{output_dir}/annotated_{index:06d}{name_part}.jpg", annotated)
    except Exception as e:
        logger.warning(f"Error saving annotated debug image: {e}")

def validate_gender_classification(num_samples=50, save_images=False, output_dir=None, ignore_matlab_coords=False):
    """Run gender validation on sample images."""
    # Set up output directory
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting gender validation with {num_samples} samples...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Ignore MATLAB coordinates: {ignore_matlab_coords}")
    
    # Create image directory if needed
    if save_images:
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
    
    # Load MATLAB data
    imdb, total = load_imdb_data()
    
    # Select samples
    samples = select_samples(imdb, total, num_samples)
    
    if not samples:
        logger.error("No valid samples found. Cannot proceed with validation.")
        return
    
    # Initialize stats
    stats = {
        'total_samples': len(samples),
        'processed': 0,
        'gender_matches': 0,
        'gender_mismatches': 0,
        'detection_errors': 0,
        'female_as_male': 0,
        'male_as_female': 0
    }
    
    # Initialize results list
    results = []
    
    # Process each sample
    logger.info(f"Processing {len(samples)} samples...")
    for idx in tqdm(samples):
        result = {'index': idx}
        
        try:
            # Get metadata
            file_path = resolve_path(imdb['full_path'][0, idx])
            result['file_path'] = file_path
            
            name = convert_to_scalar(imdb['name'][0, idx])
            result['name'] = name
            
            # Get face score
            face_score = float(imdb['face_score'][0, idx])
            result['face_score'] = face_score
            
            # Get gender
            gender, gender_str = get_gender_from_matlab(imdb, idx)
            result['expected_gender'] = gender_str
            
            if gender is None:
                logger.warning(f"Sample {idx}: Missing gender information")
                continue
            
            # Get face location
            loc = convert_to_scalar(imdb['face_location'][0, idx])
            
            # Load image
            img_path = IMDB_ROOT / file_path
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Sample {idx}: Failed to load image: {img_path}")
                continue
            
            # Crop face with adaptive crop function from gimport2.py
            face_img, crop_metadata = adaptive_crop(
                img, 
                loc, 
                margin=0.25,
                expand=0.0,
                min_size=20
            )
            
            # If using full image was requested, override the adaptive crop
            if ignore_matlab_coords:
                face_img = img.copy()
                crop_metadata = {
                    'source': 'full_image',
                    'original_width': img.shape[1],
                    'original_height': img.shape[0],
                    'final_width': img.shape[1],
                    'final_height': img.shape[0],
                    'final_aspect_ratio': img.shape[1] / max(1, img.shape[0])
                }
            
            # Check for valid image dimensions to avoid OpenCV errors
            if face_img.shape[0] <= 0 or face_img.shape[1] <= 0:
                logger.warning(f"Sample {idx}: Invalid face crop dimensions: {face_img.shape}")
                result['error'] = f"Invalid face crop dimensions: {face_img.shape}"
                results.append(result)
                continue
            
            # Run gender detection with crop metadata
            gender_result = detect_gender(face_img, gender, crop_metadata)
            
            # Add results to result dictionary
            result.update(gender_result)
            
            # Update statistics
            stats['processed'] += 1
            
            if gender_result['detection_error']:
                stats['detection_errors'] += 1
            elif gender_result['gender_match']:
                stats['gender_matches'] += 1
            else:
                stats['gender_mismatches'] += 1
                
                # Track gender-specific mismatches
                if gender == GENDER_FEMALE:
                    stats['female_as_male'] += 1
                else:
                    stats['male_as_female'] += 1
            
            # Save debug images if requested
            if save_images and (not gender_result['gender_match'] or save_images == 'all'):
                save_debug_image(
                    images_dir,
                    idx,
                    face_img,
                    gender,
                    gender_result,
                    original_img=img,  # Pass the original image
                    name=name  # Pass the celebrity name
                )
                
            # Add to results list
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            result['error'] = str(e)
            results.append(result)
    
    # Calculate rates
    if stats['processed'] > 0:
        stats['match_rate'] = stats['gender_matches'] / stats['processed']
        stats['mismatch_rate'] = stats['gender_mismatches'] / stats['processed']
        stats['error_rate'] = stats['detection_errors'] / stats['processed']
    
    # Save results to CSV
    results_file = output_dir / "validation_results.csv"
    logger.info(f"Saving results to {results_file}")
    pd.DataFrame(results).to_csv(results_file, index=False)
    
    # Save stats to CSV
    stats_file = output_dir / "validation_stats.csv"
    logger.info(f"Saving statistics to {stats_file}")
    with open(stats_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in stats.items():
            writer.writerow([key, value])
    
    # Print results to console
    print("\n" + "="*50)
    print("GENDER VALIDATION RESULTS")
    print("="*50)
    print(f"Total samples processed: {stats['processed']}")
    print(f"Gender matches: {stats['gender_matches']}")
    print(f"Gender mismatches: {stats['gender_mismatches']}")
    print(f"Detection errors: {stats['detection_errors']}")
    print(f"Female detected as male: {stats['female_as_male']}")
    print(f"Male detected as female: {stats['male_as_female']}")
    
    if stats['processed'] > 0:
        print(f"Match rate: {stats['match_rate']:.2%}")
        print(f"Mismatch rate: {stats['mismatch_rate']:.2%}")
    
    print(f"Results saved to: {results_file}")
    print(f"Statistics saved to: {stats_file}")
    print("="*50)
    
    return stats

def analyze_matlab_coords(imdb, samples):
    """Analyze MATLAB coordinates to find issues"""
    print("\n" + "="*80)
    print("MATLAB COORDINATE ANALYSIS")
    print("="*80)
    
    problematic = 0
    tiny = 0
    weird_ratio = 0
    
    for idx in samples:
        try:
            loc = convert_to_scalar(imdb['face_location'][0, idx])
            bbox = parse_bbox(loc)
            
            if bbox:
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                if width <= 1 or height <= 1:
                    problematic += 1
                    tiny += 1
                    print(f"Sample {idx}: Tiny dimension - {width}x{height} - Raw: {loc}")
                elif width > 0 and height > 0:
                    aspect = width / height
                    if aspect < 0.4 or aspect > 2.5:
                        problematic += 1
                        weird_ratio += 1
                        print(f"Sample {idx}: Odd aspect ratio - {aspect:.2f} ({width}x{height}) - Raw: {loc}")
        except Exception as e:
            print(f"Error analyzing sample {idx}: {e}")
    
    total = len(samples)
    print("-"*80)
    print(f"Total samples: {total}")
    print(f"Problematic coordinates: {problematic} ({problematic/total:.1%})")
    print(f"Tiny dimensions: {tiny} ({tiny/total:.1%})")
    print(f"Weird aspect ratios: {weird_ratio} ({weird_ratio/total:.1%})")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="IMDB Gender Classification Validator")
    
    parser.add_argument('--samples', type=int, default=50,
                      help='Number of samples to validate')
    
    parser.add_argument('--save-images', action='store_true',
                      help='Save images of mismatches')
    
    parser.add_argument('--save-all-images', action='store_true',
                      help='Save all processed images')
    
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for results')
                      
    parser.add_argument('--ignore-matlab-coords', action='store_true',
                      help='Ignore MATLAB coordinates and use the full image')
                      
    parser.add_argument('--analyze-coords', action='store_true',
                      help='Analyze MATLAB coordinates for issues')
    
    args = parser.parse_args()
    
    # Process save_images parameter
    save_images = 'all' if args.save_all_images else args.save_images
    
    # Load MATLAB data
    imdb, total = load_imdb_data()
    
    # Select samples
    samples = select_samples(imdb, total, args.samples)
    
    # Analyze coordinates if requested
    if args.analyze_coords:
        analyze_matlab_coords(imdb, samples)
    
    # Run validation
    validate_gender_classification(
        num_samples=args.samples,
        save_images=save_images,
        output_dir=args.output_dir,
        ignore_matlab_coords=args.ignore_matlab_coords
    )

if __name__ == '__main__':
    main()
