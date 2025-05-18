#!/usr/bin/env python3
"""
IMDB-WIKI Dataset Inspector - Nested Array Version

This script examines the structure of the IMDB-WIKI .mat file and extracts data
from its nested array structure.
"""

import os
import sys
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

# Path to the .mat file
MAT_FILE = "/workspace/imdb/imdb_crop/imdb.mat"
IMDB_ROOT = "/workspace/imdb/imdb_crop"

def extract_array_value(arr):
    """Safely extract value from potentially nested numpy arrays."""
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.size == 0:
        return None
    if arr.size == 1:
        return extract_array_value(arr.item()) if hasattr(arr, 'item') else arr.item()
    return [extract_array_value(x) for x in arr]

def inspect_mat_file(mat_file):
    """Examine the structure of the .mat file."""
    print(f"Loading {mat_file}...")
    mat = scipy.io.loadmat(mat_file)
    
    # Print the top-level keys
    print("\nTop-level keys in the .mat file:")
    for key in mat.keys():
        if not key.startswith('__'):  # Skip MATLAB internal keys
            print(f"  - {key}: {type(mat[key])}")
    
    # Examine the 'imdb' structure
    if 'imdb' in mat:
        imdb = mat['imdb']
        print(f"\nStructure of 'imdb':")
        print(f"  Type: {type(imdb)}")
        print(f"  Shape: {imdb.shape}")
        
        # Navigate the nested structure
        imdb_data = imdb[0, 0]
        
        # Print field names
        field_names = imdb_data.dtype.names
        print(f"\nField names: {field_names}")
        
        # Print shape of each field
        print("\nField shapes:")
        for field in field_names:
            field_data = imdb_data[field]
            print(f"  - {field}: {field_data.shape}")
            
            # Print first element of each field for debugging
            if field_data.size > 0:
                first_elem = field_data[0, 0] if field_data.shape[1] > 0 else None
                if isinstance(first_elem, np.ndarray) and first_elem.size > 0:
                    print(f"    First element: {first_elem[0] if first_elem.size > 0 else None}")
                else:
                    print(f"    First element: {first_elem}")
        
        # Check if field shapes make sense
        print("\nConsistency check:")
        main_length = None
        for field in ['full_path', 'name', 'dob']:
            if field in field_names:
                field_data = imdb_data[field]
                if field_data.size > 0:
                    length = field_data.shape[1]
                    print(f"  - {field}: {length} entries")
                    if main_length is None:
                        main_length = length
                    elif main_length != length:
                        print(f"    WARNING: Length mismatch with previous fields!")
                        
        # Create sample dataframe
        print("\nCreating sample dataframe (5 entries)...")
        
        # Initialize dataframe dictionary
        data = {}
        sample_size = 5
        
        # Extract data for each field
        for field in field_names:
            field_data = imdb_data[field]
            
            # Handle different field shapes
            if field_data.shape[0] > 0 and field_data.shape[1] > 0:
                # Get sample size elements
                sample_data = []
                for i in range(min(sample_size, field_data.shape[1])):
                    value = field_data[0, i]
                    
                    # Extract value from nested arrays
                    if isinstance(value, np.ndarray):
                        if value.size == 0:
                            extracted = None
                        elif value.size == 1:
                            if isinstance(value[0], np.ndarray):
                                if value[0].size > 0:
                                    extracted = value[0][0] if isinstance(value[0][0], (str, bytes, int, float)) else str(value[0][0])
                                else:
                                    extracted = None
                            else:
                                extracted = value[0]
                        else:
                            extracted = [v[0] if isinstance(v, np.ndarray) and v.size > 0 else v for v in value]
                    else:
                        extracted = value
                    
                    sample_data.append(extracted)
                
                data[field] = sample_data
        
        # Create dataframe
        df = pd.DataFrame(data)
        print(df)
        
        # Test loading images
        print("\nTesting image loading (3 samples)...")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            
            # Get path
            if 'full_path' in df.columns:
                path = row['full_path']
                print(f"\nSample {i+1}:")
                print(f"  Path: {path}")
                
                if isinstance(path, str):
                    # Direct path string
                    img_path = os.path.join(IMDB_ROOT, path)
                    print(f"  Full path: {img_path}")
                    print(f"  File exists: {os.path.exists(img_path)}")
                    
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            print(f"  Image loaded - Size: {img.shape}")
                            
                            # Try to extract face
                            if 'face_location' in df.columns:
                                face_loc = row['face_location']
                                if isinstance(face_loc, list) and len(face_loc) >= 4:
                                    x1, y1, x2, y2 = [int(v) for v in face_loc[:4]]
                                    print(f"  Face location: ({x1}, {y1}) to ({x2}, {y2})")
                                    
                                    # Check if face is full image
                                    height, width = img.shape[:2]
                                    is_full_image = (x1 == 0 and y1 == 0 and x2 == width and y2 == height)
                                    print(f"  Face is full image: {is_full_image}")
                                    
                                    # Extract face
                                    try:
                                        face = img[y1:y2, x1:x2]
                                        print(f"  Face extracted - Size: {face.shape}")
                                    except Exception as e:
                                        print(f"  Error extracting face: {e}")
                        else:
                            print("  Failed to load image")

def load_and_process_data(mat_file, limit=10):
    """Load the .mat file and process it into a usable format."""
    print(f"Loading and processing {mat_file}...")
    
    # Load the mat file
    mat = scipy.io.loadmat(mat_file)
    imdb_data = mat['imdb'][0, 0]
    
    # Get field names
    field_names = imdb_data.dtype.names
    
    # Initialize dictionary to hold the data
    processed_data = {}
    
    # Determine number of entries
    num_entries = imdb_data['full_path'].shape[1]
    print(f"Total entries: {num_entries}")
    
    # Limit the number of entries to process
    process_count = min(limit, num_entries)
    print(f"Processing {process_count} entries...")
    
    # Process each field
    for field in field_names:
        field_data = imdb_data[field]
        
        # Special handling for different fields
        if field in ['celeb_names']:
            # This field might have a different shape
            processed_data[field] = []
            for i in range(min(process_count, field_data.shape[1])):
                value = field_data[0, i]
                if isinstance(value, np.ndarray) and value.size > 0:
                    processed_data[field].append(value[0])
                else:
                    processed_data[field].append(None)
        else:
            # Regular fields
            processed_data[field] = []
            for i in range(min(process_count, field_data.shape[1])):
                value = field_data[0, i]
                
                # Extract value from nested arrays
                if isinstance(value, np.ndarray):
                    if value.size == 0:
                        processed_data[field].append(None)
                    elif value.size == 1:
                        processed_data[field].append(value[0])
                    else:
                        processed_data[field].append(value.tolist())
                else:
                    processed_data[field].append(value)
    
    # Create dataframe
    df = pd.DataFrame(processed_data)
    
    # Apply filters
    print("\nApplying filters...")
    original_count = len(df)
    
    # Remove entries without faces
    df = df[df['face_score'] != float('-inf')]
    print(f"After removing entries without faces: {len(df)} (removed {original_count - len(df)})")
    
    # Remove entries with multiple faces
    df = df[df['second_face_score'].isna()]
    print(f"After removing entries with multiple faces: {len(df)} (removed {original_count - len(df)})")
    
    # Remove entries with low face scores
    df = df[df['face_score'] >= 3.0]
    print(f"After removing entries with low face scores: {len(df)} (removed {original_count - len(df)})")
    
    return df

def test_image_loading(df, num_samples=3):
    """Test loading images and extracting faces."""
    print(f"\nTesting image loading for {num_samples} random samples...")
    
    # Select random samples
    sample_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        row = df.iloc[idx]
        
        # Get celebrity name
        celeb_name = row['name'] if 'name' in df.columns else "Unknown"
        print(f"\nSample {i+1} - {celeb_name}:")
        
        # Get path
        if 'full_path' in df.columns:
            path = row['full_path']
            if isinstance(path, list) and len(path) > 0:
                path = path[0]
            
            print(f"  Path: {path}")
            
            # Construct full path
            img_path = os.path.join(IMDB_ROOT, path)
            print(f"  Full path: {img_path}")
            print(f"  File exists: {os.path.exists(img_path)}")
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    print(f"  Image loaded - Size: {img.shape}")
                    
                    # Try to extract face
                    if 'face_location' in df.columns:
                        face_loc = row['face_location']
                        if isinstance(face_loc, list) and len(face_loc) >= 4:
                            x1, y1, x2, y2 = [int(v) for v in face_loc[:4]]
                            print(f"  Face location: ({x1}, {y1}) to ({x2}, {y2})")
                            
                            # Check if face is full image
                            height, width = img.shape[:2]
                            is_full_image = (x1 == 0 and y1 == 0 and x2 == width and y2 == height)
                            print(f"  Face is full image: {is_full_image}")
                            
                            # Extract face
                            try:
                                face = img[y1:y2, x1:x2]
                                print(f"  Face extracted - Size: {face.shape}")
                                
                                # Calculate face percentage
                                face_area = (x2 - x1) * (y2 - y1)
                                img_area = width * height
                                face_pct = (face_area / img_area) * 100
                                print(f"  Face is {face_pct:.2f}% of the image")
                                
                                # Test if we need to use 'skip' detector
                                print("\n  Testing DeepFace with different approaches:")
                                try:
                                    from deepface import DeepFace
                                    
                                    # Approach 1: Full image with 'retinaface' detector
                                    try:
                                        print("  Approach 1: Full image with 'retinaface' detector")
                                        faces = DeepFace.extract_faces(
                                            img_path=img,
                                            detector_backend='retinaface',
                                            enforce_detection=False
                                        )
                                        print(f"    Detected {len(faces)} faces")
                                    except Exception as e:
                                        print(f"    Error: {e}")
                                    
                                    # Approach 2: Full image with 'skip' detector
                                    try:
                                        print("  Approach 2: Full image with 'skip' detector")
                                        vec = DeepFace.represent(
                                            img_path=img,
                                            model_name='VGG-Face',
                                            detector_backend='skip'
                                        )
                                        print(f"    Got vector of length {len(vec)}")
                                    except Exception as e:
                                        print(f"    Error: {e}")
                                    
                                    # Approach 3: Face crop with 'skip' detector
                                    try:
                                        print("  Approach 3: Face crop with 'skip' detector")
                                        vec = DeepFace.represent(
                                            img_path=face,
                                            model_name='VGG-Face',
                                            detector_backend='skip'
                                        )
                                        print(f"    Got vector of length {len(vec)}")
                                    except Exception as e:
                                        print(f"    Error: {e}")
                                except ImportError:
                                    print("  DeepFace not available for testing")
                            except Exception as e:
                                print(f"  Error extracting face: {e}")
                else:
                    print("  Failed to load image")

if __name__ == "__main__":
    # Basic inspection of the file structure
    inspect_mat_file(MAT_FILE)
    
    # Load and process data
    df = load_and_process_data(MAT_FILE, limit=100)
    
    # Test image loading
    test_image_loading(df, num_samples=3)
