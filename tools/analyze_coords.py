#!/usr/bin/env python3
"""
IMDB MATLAB Coordinate Analyzer

Analyzes the face_location coordinates in the IMDB dataset to detect problematic values.
"""
import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import scipy.io

# Configuration
DATA_PATH = Path("/workspace/imdb")
IMDB_ROOT = DATA_PATH / "imdb_crop"
MAT_FILE = IMDB_ROOT / "imdb.mat"

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

def parse_bbox(loc):
    """
    Parse bounding box coordinates from MATLAB format.
    """
    while isinstance(loc, list) and len(loc) == 1:
        loc = loc[0]
    if not isinstance(loc, (list, tuple)) or len(loc) < 4:
        return None
    try:
        # Extract values directly - they're already in the right order
        x1, y1, x2, y2 = [int(float(loc[i])) for i in range(4)]
        return x1, y1, x2, y2
    except Exception as e:
        print(f"Error parsing bbox: {e}, loc: {loc}")
        return None

def analyze_matlab_coordinates(num_samples=100):
    print(f"Loading MATLAB file from {MAT_FILE}")
    mat = scipy.io.loadmat(str(MAT_FILE))
    imdb = mat['imdb'][0, 0]
    total_entries = imdb['full_path'][0].shape[0]
    print(f"Successfully loaded {total_entries} entries")
    
    # Select random samples
    samples = random.sample(range(total_entries), min(total_entries, num_samples))
    
    print("\n" + "="*80)
    print("MATLAB COORDINATE ANALYSIS")
    print("="*80)
    
    # Stats counters
    problematic = 0
    tiny_width = 0
    tiny_height = 0
    weird_ratio = 0
    invalid = 0
    reversed_coords = 0  # x1 > x2 or y1 > y2
    
    # Dimension distribution
    width_dist = {}
    height_dist = {}
    
    for idx in samples:
        try:
            # Get face location
            loc = convert_to_scalar(imdb['face_location'][0, idx])
            file_path = convert_to_scalar(imdb['full_path'][0, idx])
            fs = float(imdb['face_score'][0, idx])
            
            # Raw coordinates
            print(f"Sample {idx} ({file_path}): Raw Coords: {loc}, Face Score: {fs:.2f}")
            
            # Parse bbox
            bbox = parse_bbox(loc)
            if bbox:
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                # Track dimensions
                width_key = width if width < 20 else (width // 10) * 10
                height_key = height if height < 20 else (height // 10) * 10
                width_dist[width_key] = width_dist.get(width_key, 0) + 1
                height_dist[height_key] = height_dist.get(height_key, 0) + 1
                
                # Check for issues
                if x1 > x2 or y1 > y2:
                    reversed_coords += 1
                    problematic += 1
                    print(f"  ⚠️ Reversed coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                
                if width <= 1:
                    tiny_width += 1
                    problematic += 1
                    print(f"  ⚠️ Tiny width: {width}px")
                    
                if height <= 1:
                    tiny_height += 1
                    problematic += 1
                    print(f"  ⚠️ Tiny height: {height}px")
                
                if width > 0 and height > 0:
                    aspect = width / height
                    if aspect < 0.4 or aspect > 2.5:
                        weird_ratio += 1
                        problematic += 1
                        print(f"  ⚠️ Odd aspect ratio: {aspect:.2f} ({width}x{height})")
                
                print(f"  Parsed: x1={x1}, y1={y1}, x2={x2}, y2={y2} ({width}x{height})")
            else:
                invalid += 1
                problematic += 1
                print(f"  ⚠️ Invalid coordinates")
        except Exception as e:
            print(f"Error analyzing sample {idx}: {e}")
    
    # Summary
    total = len(samples)
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    print(f"Total samples analyzed: {total}")
    print(f"Problematic coordinates: {problematic} ({problematic/total:.1%})")
    print(f"Invalid coordinates: {invalid} ({invalid/total:.1%})")
    print(f"Tiny width (≤1px): {tiny_width} ({tiny_width/total:.1%})")
    print(f"Tiny height (≤1px): {tiny_height} ({tiny_height/total:.1%})")
    print(f"Weird aspect ratios: {weird_ratio} ({weird_ratio/total:.1%})")
    print(f"Reversed coordinates: {reversed_coords} ({reversed_coords/total:.1%})")
    
    # Dimension distribution
    print("\nWidth distribution:")
    for width, count in sorted(width_dist.items()):
        print(f"  {width}px: {count} ({count/total:.1%})")
        
    print("\nHeight distribution:")
    for height, count in sorted(height_dist.items()):
        print(f"  {height}px: {count} ({count/total:.1%})")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="IMDB MATLAB Coordinate Analyzer")
    
    parser.add_argument('--samples', type=int, default=100,
                      help='Number of samples to analyze')
    
    args = parser.parse_args()
    
    analyze_matlab_coordinates(args.samples)

if __name__ == '__main__':
    main()
