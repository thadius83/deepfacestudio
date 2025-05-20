#!/usr/bin/env python3
"""
Database-enabled face embedding generator for DeepFace Suite.

This script analyzes face images to generate and store embeddings in a PostgreSQL database
with pgvector support. Features:
1. Parallel processing to utilize multiple CPU cores
2. Batch processing to manage memory usage
3. Recursive folder processing
4. Storage and retrieval of embeddings in a PostgreSQL/pgvector database
5. Comprehensive logging

Usage:
  python generate_embeddings_db.py --folder /path/to/face/folder
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Add parent directory to path to allow importing from backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import backend modules
from backend.app.config import MODEL_NAME, DETECTOR_BACKEND, ALLOWED_EXT
from backend.app.db import bulk_upsert_embeddings, count_embeddings

# Reuse functions from the existing script
from advanced_outlier_detection import (
    get_embedding, 
    count_images, 
    generate_embeddings_parallel,
    find_image_folders
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def store_embeddings_in_db(
    embeddings: Dict[str, np.ndarray], 
    batch_size: int = 1000
) -> int:
    """
    Store embeddings in the database in batches.
    
    Args:
        embeddings: Dictionary mapping image paths to embeddings
        batch_size: Number of embeddings to store in each batch
        
    Returns:
        Number of embeddings stored
    """
    if not embeddings:
        return 0
    
    logger.info(f"Storing {len(embeddings)} embeddings in database (batch size: {batch_size})...")
    
    # Prepare rows for insertion
    rows = []
    for filepath, embedding in embeddings.items():
        folder_label = Path(filepath).parent.name
        # Convert numpy array to list for psycopg2
        embedding_list = embedding.astype(np.float32).tolist()
        rows.append((filepath, folder_label, embedding_list))
    
    # Store embeddings in batches
    total_stored = 0
    with tqdm(total=len(rows), desc="Storing embeddings") as pbar:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            bulk_upsert_embeddings(batch)
            total_stored += len(batch)
            pbar.update(len(batch))
    
    logger.info(f"Stored {total_stored} embeddings in database")
    return total_stored

def process_folder(
    folder_path: str,
    model_name: str = MODEL_NAME,
    detector_backend: str = DETECTOR_BACKEND,
    batch_size: int = 2000,
    max_workers: int = 5,
    db_batch_size: int = 1000
) -> Tuple[int, int]:
    """
    Process a folder of images, generating and storing embeddings.
    
    Args:
        folder_path: Path to folder containing images
        model_name: Model name for face recognition
        detector_backend: Detector backend for face detection
        batch_size: Number of images to process in parallel
        max_workers: Number of worker processes
        db_batch_size: Number of embeddings to store in each database batch
        
    Returns:
        Tuple of (processed_images, stored_embeddings)
    """
    folder_path = Path(folder_path)
    logger.info(f"Processing folder: {folder_path}")
    
    # List image files
    image_files = []
    for ext in ALLOWED_EXT:
        ext = ext.strip(".")  # Remove leading dot if present
        image_files.extend(list(folder_path.glob(f"*.{ext}")))
    
    if not image_files:
        logger.warning(f"No images found in {folder_path}")
        return 0, 0
    
    image_paths = [str(f) for f in image_files]
    logger.info(f"Found {len(image_paths)} images in {folder_path}")
    
    # Generate embeddings
    embeddings = generate_embeddings_parallel(
        image_paths,
        model_name=model_name,
        detector_backend=detector_backend,
        batch_size=batch_size,
        max_workers=max_workers
    )
    
    # Store embeddings
    stored = store_embeddings_in_db(embeddings, batch_size=db_batch_size)
    
    return len(image_paths), stored

def process_recursive(
    base_folder: str,
    model_name: str = MODEL_NAME,
    detector_backend: str = DETECTOR_BACKEND,
    batch_size: int = 2000,
    max_workers: int = 5,
    db_batch_size: int = 1000,
    min_images: int = 5
) -> Tuple[int, int, int]:
    """
    Process all folders recursively, generating and storing embeddings.
    
    Args:
        base_folder: Base folder to process
        model_name: Model name for face recognition
        detector_backend: Detector backend for face detection
        batch_size: Number of images to process in parallel
        max_workers: Number of worker processes
        db_batch_size: Number of embeddings to store in each database batch
        min_images: Minimum number of images required to process a folder
        
    Returns:
        Tuple of (total_folders, total_images, total_stored)
    """
    # Find all folders with enough images
    folders = find_image_folders(base_folder, min_images=min_images, recursive=True)
    
    if not folders:
        logger.error(f"No folders with at least {min_images} images found in {base_folder}")
        return 0, 0, 0
    
    logger.info(f"Found {len(folders)} folders with at least {min_images} images")
    
    # Process each folder
    total_images = 0
    total_stored = 0
    
    for folder in folders:
        processed, stored = process_folder(
            folder_path=folder,
            model_name=model_name,
            detector_backend=detector_backend,
            batch_size=batch_size,
            max_workers=max_workers,
            db_batch_size=db_batch_size
        )
        
        total_images += processed
        total_stored += stored
    
    return len(folders), total_images, total_stored

def main():
    parser = argparse.ArgumentParser(description="Generate and store face embeddings in a PostgreSQL database")
    
    # Required arguments
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing face images")
    
    # Face recognition options
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model to use for embeddings")
    parser.add_argument("--detector", type=str, default=DETECTOR_BACKEND, help="Face detector to use")
    
    # Performance options
    parser.add_argument("--batch-size", type=int, default=2000, 
                        help="Number of images to process in each batch (default: 2000)")
    parser.add_argument("--workers", type=int, default=5, 
                        help="Maximum number of worker processes (default: 5)")
    parser.add_argument("--db-batch-size", type=int, default=1000,
                        help="Number of embeddings to store in each database batch (default: 1000)")
    
    # Recursive options
    parser.add_argument("--recursive", action="store_true", 
                        help="Process all subfolders recursively")
    parser.add_argument("--min-images", type=int, default=5, 
                        help="Minimum number of images required to process a folder (default: 5)")
    
    args = parser.parse_args()
    
    # Make sure folder exists
    if not os.path.isdir(args.folder):
        logger.error(f"Folder not found: {args.folder}")
        return 1
    
    # Start timer
    start_time = time.time()
    
    # Process folder(s)
    if args.recursive:
        folders, total_images, total_stored = process_recursive(
            base_folder=args.folder,
            model_name=args.model,
            detector_backend=args.detector,
            batch_size=args.batch_size,
            max_workers=args.workers,
            db_batch_size=args.db_batch_size,
            min_images=args.min_images
        )
        
        # Print summary
        logger.info("\n=== Summary ===")
        logger.info(f"Processed {folders} folders")
        logger.info(f"Total images: {total_images}")
        logger.info(f"Faces detected and stored: {total_stored}")
        logger.info(f"Total embeddings in database: {count_embeddings()}")
    else:
        processed, stored = process_folder(
            folder_path=args.folder,
            model_name=args.model,
            detector_backend=args.detector,
            batch_size=args.batch_size,
            max_workers=args.workers,
            db_batch_size=args.db_batch_size
        )
        
        # Print summary
        logger.info("\n=== Summary ===")
        logger.info(f"Total images: {processed}")
        logger.info(f"Faces detected and stored: {stored}")
        logger.info(f"Total embeddings in database: {count_embeddings()}")
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
