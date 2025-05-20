#!/usr/bin/env python3
"""
High-performance embedding generator for large datasets with pgvector storage.

Key optimizations:
1. Single global worker pool that persists across all folders
2. Skip images that already have embeddings in the database
3. Work-stealing scheduler with balanced chunk allocation
4. Comprehensive progress tracking
5. H100-optimized parameters by default

Usage examples:
    # Process all images in a single folder
    python tools/db_generate_embeddings.py --folder /data/faces

    # Process recursively with 16 workers
    python tools/db_generate_embeddings.py --folder /data/faces --recursive --workers 16
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from multiprocessing import Pool, cpu_count, Queue

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.app.config import MODEL_NAME, DETECTOR_BACKEND, ALLOWED_EXT
from backend.app.db import bulk_upsert_embeddings, fetch_embeddings
from advanced_outlier_detection import get_embedding, find_image_folders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def worker_process_image(args: Tuple[str, str, str]) -> Tuple[str, Optional[np.ndarray]]:
    """Worker function for processing a single image file."""
    filepath, model_name, detector_backend = args
    try:
        path, embedding = get_embedding((filepath, model_name, detector_backend, False))
        return path, embedding
    except Exception as e:
        logger.warning(f"Error processing {filepath}: {str(e)}")
        return filepath, None

def find_all_image_files(base_folder: str, recursive: bool = False, min_images: int = 5) -> List[str]:
    """Find all valid image files to process."""
    all_files = []
    start_time = time.time()
    
    # Get folders to process
    if recursive:
        folders = find_image_folders(base_folder, min_images=min_images, recursive=True)
        logger.info(f"Found {len(folders)} folders with at least {min_images} images")
    else:
        folders = [base_folder]
    
    # Scan for image files
    for folder in folders:
        for ext in ALLOWED_EXT:
            ext = ext.strip(".")  # Remove leading dot if present
            all_files.extend(str(p) for p in Path(folder).glob(f"*.{ext}"))
    
    elapsed = time.time() - start_time
    logger.info(f"Found {len(all_files)} images in {elapsed:.2f} seconds")
    return all_files

def get_existing_embeddings() -> Set[str]:
    """Get set of file paths that already have embeddings in the database."""
    start_time = time.time()
    logger.info("Querying database for existing embeddings...")
    
    embeddings = fetch_embeddings()
    existing_paths = set(embeddings.keys())
    
    elapsed = time.time() - start_time
    logger.info(f"Found {len(existing_paths)} existing embeddings in {elapsed:.2f} seconds")
    return existing_paths

def format_time(seconds):
    """Format seconds into a human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{int(minutes)}m {int(seconds % 60)}s"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)}h {int(minutes)}m"


def process_images(
    all_files: List[str],
    existing_paths: Set[str],
    model_name: str,
    detector_backend: str,
    workers: int,
    chunk_size: int,
    db_batch_size: int
) -> int:
    """
    Process all image files using a global worker pool.
    
    Args:
        all_files: List of image files to process
        existing_paths: Set of file paths already in the database
        model_name: Model name for face recognition
        detector_backend: Detector backend for face detection
        workers: Number of worker processes
        chunk_size: Number of images to process in each worker chunk
        db_batch_size: Number of embeddings to store in each database batch
        
    Returns:
        Number of embeddings stored
    """
    # Filter out images that already have embeddings
    files_to_process = [f for f in all_files if f not in existing_paths]
    total_files = len(files_to_process)
    
    if total_files == 0:
        logger.info("All images already have embeddings in the database")
        return 0
    
    # Create phase progress bar
    print("=== Progress Tracking ===")
    # Phase 1 complete
    print(f"Scanning:   [{'█' * 25}] 100.0% ({len(all_files):,} images found)")
    # Phase 2 complete
    skipped = len(all_files) - total_files
    skip_percent = (skipped / len(all_files) * 100) if all_files else 0
    print(f"Filtering:  [{'█' * 25}] 100.0% ({skipped:,} existing embeddings skipped, {skip_percent:.1f}%)")
    # Phase 3 starting
    print(f"Processing: [{'░' * 25}] 0.0% (0/{total_files:,}) @ 0.0 img/s")
    # Phase 4 starting  
    print(f"DB Storage: [{'░' * 25}] 0.0% (0/{total_files:,}) @ 0.0 img/s")
    
    # Folder tracking
    folders_processed = set()
    current_folders = {}
    
    logger.info(f"Processing {total_files:,} new images with {workers} workers")
    
    # Create task list
    tasks = [(filepath, model_name, detector_backend) for filepath in files_to_process]
    
    # Progress tracking variables
    start_time = time.time()
    last_update = start_time
    processing_count = 0
    db_count = 0
    
    # Process images using global worker pool
    results_buffer = []
    total_stored = 0
    
    with Pool(processes=workers) as pool:
        with tqdm(total=total_files, desc="Overall Progress", unit="img") as pbar:
            # Process images in chunks
            for filepath, embedding in pool.imap_unordered(worker_process_image, tasks, chunksize=chunk_size):
                processing_count += 1
                folder = Path(filepath).parent.name
                
                # Track folders
                if folder not in folders_processed:
                    current_folders[folder] = current_folders.get(folder, 0) + 1
                
                # Progress updates
                now = time.time()
                if now - last_update > 2 or processing_count == total_files:  # Update every 2 seconds
                    elapsed = now - start_time
                    processing_rate = processing_count / elapsed if elapsed > 0 else 0
                    db_rate = db_count / elapsed if elapsed > 0 else 0
                    
                    # ETA calculation
                    remaining_processing = total_files - processing_count
                    eta_processing = remaining_processing / processing_rate if processing_rate > 0 else 0
                    
                    # Clear previous lines (3 lines: Processing, DB Storage, Folder status)
                    print("\033[3A\033[K", end="")
                    
                    # Update Processing bar
                    proc_percent = processing_count / total_files * 100
                    proc_blocks = int(proc_percent / 4)  # 25 blocks = 100%
                    proc_bar = "█" * proc_blocks + "░" * (25 - proc_blocks)
                    print(f"Processing: [{proc_bar}] {proc_percent:.1f}% ({processing_count:,}/{total_files:,}) @ {processing_rate:.1f} img/s ETA: {format_time(eta_processing)}")
                    
                    # Update DB Storage bar
                    db_percent = db_count / total_files * 100
                    db_blocks = int(db_percent / 4)  # 25 blocks = 100%
                    db_bar = "█" * db_blocks + "░" * (25 - db_blocks)
                    print(f"DB Storage: [{db_bar}] {db_percent:.1f}% ({db_count:,}/{total_files:,}) @ {db_rate:.1f} img/s")
                    
                    # Folder status (keep last 3 completed, show current count)
                    if folders_processed:
                        recent_complete = list(folders_processed)[-3:] if len(folders_processed) > 3 else list(folders_processed)
                        folder_status = ", ".join(recent_complete)
                        print(f"Folders: Completed {len(folders_processed)}, Active: {len(current_folders)} ({folder_status}...)")
                    else:
                        print(f"Folders: Active: {len(current_folders)}")
                    
                    last_update = now
                
                # Process embedding if valid
                if embedding is not None:
                    folder_label = Path(filepath).parent.name
                    embedding_list = embedding.astype(np.float32).tolist()
                    results_buffer.append((filepath, folder_label, embedding_list))
                    
                    # Batch insert to database
                    if len(results_buffer) >= db_batch_size:
                        bulk_upsert_embeddings(results_buffer)
                        db_count += len(results_buffer)
                        total_stored += len(results_buffer)
                        results_buffer = []
                        
                        # Update folder tracking - folders with no pending items are "completed"
                        # This is an approximation since we process in random order
                        for f_path, _, _ in results_buffer:
                            f_folder = Path(f_path).parent.name
                            if f_folder in current_folders:
                                current_folders[f_folder] -= 1
                                if current_folders[f_folder] <= 0:
                                    folders_processed.add(f_folder)
                                    del current_folders[f_folder]
                
                pbar.update(1)
    
    # Store any remaining embeddings
    if results_buffer:
        bulk_upsert_embeddings(results_buffer)
        total_stored += len(results_buffer)
    
    # Clear progress bars at the end
    print(f"Processing: [{'█' * 25}] 100.0% ({total_files:,}/{total_files:,}) COMPLETE")
    print(f"DB Storage: [{'█' * 25}] 100.0% ({total_stored:,}/{total_files:,}) COMPLETE")
    
    return total_stored

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate and store face embeddings with optimized performance")
    
    # Required arguments
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing face images")
    
    # Options
    parser.add_argument("--recursive", action="store_true", help="Process all subfolders recursively")
    parser.add_argument("--min-images", type=int, default=5, help="Minimum number of images required to process a folder")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model to use for embeddings")
    parser.add_argument("--detector", type=str, default=DETECTOR_BACKEND, help="Face detector to use")
    
    # Performance options
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of images per worker chunk")
    parser.add_argument("--db-batch-size", type=int, default=1000, help="Number of embeddings per database batch")
    
    args = parser.parse_args()
    
    # Validate folder
    if not os.path.isdir(args.folder):
        logger.error(f"Folder not found: {args.folder}")
        return 1
    
    # Start timer
    start_time = time.time()
    
    # Find images and check which ones need processing
    all_files = find_all_image_files(args.folder, args.recursive, args.min_images)
    existing_paths = get_existing_embeddings()
    
    # Process images and store embeddings
    total_stored = process_images(
        all_files,
        existing_paths,
        args.model,
        args.detector,
        args.workers,
        args.chunk_size,
        args.db_batch_size
    )
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"Stored {total_stored} new embeddings in {elapsed_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
