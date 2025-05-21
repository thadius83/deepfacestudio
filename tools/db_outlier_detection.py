#!/usr/bin/env python3
"""
Database-enabled face image outlier detection tool for DeepFace Suite.

This script analyzes face images stored in a PostgreSQL/pgvector database to identify outliers.
Features:
1. Fast retrieval of embeddings from PostgreSQL/pgvector database
2. Dimensionality reduction with PCA
3. Hierarchical clustering for efficient outlier detection
4. Outlier deletion capability
5. Comprehensive logging and reporting

Usage:
  python db_outlier_detection.py --folder-label celebrity_A --output /path/to/review/folder
"""

import os
import sys
import csv
import shutil
import argparse
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Set
from collections import defaultdict

# For visualizations
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# scikit-learn for dimensionality reduction and clustering
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Add parent directory to path to allow importing from backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.app.config import MODEL_NAME, DETECTOR_BACKEND, ALLOWED_EXT
from backend.app.db import (
    fetch_embeddings, 
    update_outlier_results,
    count_embeddings,
    get_folder_labels
)

# Reuse functions from the existing script for visualization and file operations
from advanced_outlier_detection import (
    apply_dimensionality_reduction,
    perform_hierarchical_clustering,
    identify_outliers_from_clusters,
    visualize_clusters,
    copy_outliers,
    delete_outliers,
    generate_report,
    write_outliers_list,
    generate_activity_log
)

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def analyze_embeddings_from_db(
    folder_label: str,
    output_dir: str = None,
    outlier_threshold: float = 2.0,
    variance_retained: float = 0.95,
    n_clusters: Optional[int] = None,
    visualization: bool = True,
    should_delete_outliers: bool = False,
    force_delete: bool = False,
    preserve_structure: bool = True,
    recompute: bool = False,
    dry_run: bool = False
) -> Tuple[Dict[str, bool], List[str]]:
    """
    Analyze embeddings from the database to find potential outliers.
    
    Args:
        folder_label: Folder label to analyze
        output_dir: Directory to save results and outliers
        outlier_threshold: Number of standard deviations above mean to consider as outlier
        variance_retained: Proportion of variance to retain in PCA
        n_clusters: Number of clusters for hierarchical clustering (None for auto)
        visualization: Whether to generate visualizations
        should_delete_outliers: Whether to delete outliers after copying
        force_delete: Whether to skip confirmation prompt for deletion
        preserve_structure: Whether to preserve folder structure in output
        recompute: Whether to recompute outliers even if already marked
        
    Returns:
        Dictionary mapping each file to boolean indicating if it's an outlier,
        List of deleted files
    """
    start_time = time.time()
    stats = {}
    
    # 1. Set up paths
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"db_outliers_{folder_label}_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Analyzing folder label: {folder_label}")
    logger.info(f"Output directory: {output_path}")
    
    # Set up file logger for this analysis
    file_handler = logging.FileHandler(output_path / "process.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 2. Fetch embeddings from database
    logger.info(f"Fetching embeddings for folder label: {folder_label}")
    embeddings = fetch_embeddings(folder_label=folder_label)
    
    if not embeddings:
        logger.error(f"No embeddings found for folder label: {folder_label}")
        logger.removeHandler(file_handler)
        return {}, []
    
    logger.info(f"Found {len(embeddings)} embeddings")
    stats["total_images"] = len(embeddings)
    
    if len(embeddings) < 2:
        logger.error("Not enough valid images with faces detected (need at least 2)")
        logger.removeHandler(file_handler)
        return {}, []
    
    # 3. Apply dimensionality reduction
    reduced_embeddings, pca_model = apply_dimensionality_reduction(
        embeddings,
        variance_retained=variance_retained
    )
    
    # 4. Perform hierarchical clustering
    clustering, clusters = perform_hierarchical_clustering(
        reduced_embeddings,
        n_clusters=n_clusters
    )
    
    stats["clusters"] = len(set(clusters.values()))
    
    # 5. Identify outliers
    outliers = identify_outliers_from_clusters(
        reduced_embeddings,
        clusters,
        threshold=outlier_threshold
    )
    
    outlier_count = sum(1 for is_outlier in outliers.values() if is_outlier)
    stats["outliers"] = outlier_count
    
    # 6. Store outlier results in database
    results = []
    for path, cluster_id in clusters.items():
        # Calculate distance to cluster centroid
        distance = np.linalg.norm(
            reduced_embeddings[path] - np.mean(
                [reduced_embeddings[p] for p, c in clusters.items() if c == cluster_id],
                axis=0,
            )
        )
        is_outlier = outliers.get(path, False)

        # Explicitly cast NumPy types to built-in Python types to avoid psycopg2 "can't adapt" error
        results.append(
            (
                path,
                int(cluster_id),         # ensure Python int
                float(distance),         # ensure Python float
                bool(is_outlier),        # ensure Python bool
            )
        )
    
    updated = update_outlier_results(results)
    logger.info(f"Updated {updated} rows in database with outlier results")
    
    # 7. Prepare outlier list and handle missing files
    outlier_files = [f for f, is_outlier in outliers.items() if is_outlier]
    missing_files = [f for f in outlier_files if not os.path.exists(f)]

    # Delete embeddings that reference files already gone
    if missing_files:
        from backend.app.db import delete_embeddings
        removed = delete_embeddings(missing_files)
        logger.warning(f"{len(missing_files)} outlier files already missing on disk. "
                       f"Removed {removed} embeddings that pointed to them.")

        # Keep only existing files for further copying / deletion
        outlier_files = [f for f in outlier_files if os.path.exists(f)]

    if outlier_files:
        write_outliers_list(outlier_files, str(output_path))

    # 8. Copy outliers to review folder (only those that still exist)
    review_folder = output_path / "outliers"
    outliers_subset = {p: True for p in outlier_files}
    
    if dry_run:
        logger.info(f"DRY-RUN: Would copy {len(outlier_files)} outlier files to {review_folder}")
        copied_files = outlier_files  # Pretend we copied them for reporting purposes
    else:
        copied_files = copy_outliers(
            outliers_subset,
            str(review_folder),
            preserve_structure=preserve_structure,
        )
    
    # 9. Delete outliers if requested
    deleted_files = []
    if should_delete_outliers and copied_files:
        if dry_run:
            logger.info(f"DRY-RUN: Would delete {len(copied_files)} outlier files")
            # No deleted_files list when in dry run mode
        else:
            deleted_files = delete_outliers(copied_files, force=force_delete)

            # Also remove their embeddings from the database to keep things in sync
            if deleted_files:
                from backend.app.db import delete_embeddings  # local import to avoid circular deps
                removed = delete_embeddings(deleted_files)
                logger.info(f"Removed {removed} embeddings from database")
    
    # 10. Generate report
    generate_report(
        reduced_embeddings, 
        clusters, 
        outliers, 
        str(output_path)
    )
    
    # 11. Generate visualizations
    if visualization:
        logger.info("Generating visualizations...")
        visualize_clusters(reduced_embeddings, clusters, outliers, str(output_path))
    
    # 12. Generate activity log
    elapsed_time = time.time() - start_time
    generate_activity_log(
        folder_path=folder_label,  # Using folder_label instead of path
        output_dir=str(output_path),
        outliers=outliers,
        copied_files=copied_files,
        deleted_files=deleted_files,
        elapsed_time=elapsed_time,
        stats=stats
    )
    
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    logger.removeHandler(file_handler)
    
    return outliers, deleted_files

def process_all_folders(
    output_dir: str,
    outlier_threshold: float = 2.0,
    variance_retained: float = 0.95,
    n_clusters: Optional[int] = None,
    visualization: bool = True,
    should_delete_outliers: bool = False,
    force_delete: bool = False,
    preserve_structure: bool = True,
    recompute: bool = False,
    min_images: int = 5,
    dry_run: bool = False
) -> Dict[str, Dict[str, bool]]:
    """
    Process all folder labels in the database.
    
    Args:
        output_dir: Directory to save results and outliers
        outlier_threshold: Number of standard deviations above mean to consider as outlier
        variance_retained: Proportion of variance to retain in PCA
        n_clusters: Number of clusters for hierarchical clustering (None for auto)
        visualization: Whether to generate visualizations
        should_delete_outliers: Whether to delete outliers after copying
        force_delete: Whether to skip confirmation prompt for deletion
        preserve_structure: Whether to preserve folder structure in output
        recompute: Whether to recompute outliers even if already marked
        min_images: Minimum number of images required to process a folder
        
    Returns:
        Dictionary mapping folder labels to dictionaries of outliers
    """
    # Get all folder labels
    folder_labels = get_folder_labels()
    
    if not folder_labels:
        logger.error("No folder labels found in database")
        return {}
    
    # Filter folders with enough images
    valid_folders = []
    for label in folder_labels:
        count = count_embeddings(label)
        if count >= min_images:
            valid_folders.append(label)
    
    if not valid_folders:
        logger.error(f"No folders with at least {min_images} images found")
        return {}
    
    logger.info(f"Found {len(valid_folders)} folders with at least {min_images} images")
    
    # Create main output directory
    main_output_path = Path(output_dir)
    main_output_path.mkdir(parents=True, exist_ok=True)
    
    # Create summary report file
    summary_file = main_output_path / "summary_report.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Folder Label', 'Total Images', 'Outliers', 'Deleted'])
    
    # Process each folder
    all_outliers = {}
    all_deleted = []
    
    for label in valid_folders:
        logger.info(f"\nProcessing folder label: {label}")
        
        # Process this folder
        folder_output = main_output_path / label
        
        outliers, deleted = analyze_embeddings_from_db(
            folder_label=label,
            output_dir=str(folder_output),
            outlier_threshold=outlier_threshold,
            variance_retained=variance_retained,
            n_clusters=n_clusters,
            visualization=visualization,
            should_delete_outliers=should_delete_outliers,
            force_delete=force_delete,
            preserve_structure=preserve_structure,
            recompute=recompute,
            dry_run=dry_run
        )
        
        # Record results
        all_outliers[label] = outliers
        all_deleted.extend(deleted)
        
        # Update summary report
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            outlier_count = sum(1 for is_outlier in outliers.values() if is_outlier)
            writer.writerow([
                label,
                len(outliers),
                outlier_count,
                len(deleted)
            ])
    
    # Write master list of all outliers
    all_outlier_files = []
    for folder_outliers in all_outliers.values():
        all_outlier_files.extend([f for f, is_outlier in folder_outliers.items() if is_outlier])
    
    if all_outlier_files:
        master_list_path = write_outliers_list(all_outlier_files, str(main_output_path))
        logger.info(f"Master list of all outliers saved to {master_list_path}")
    
    # Print summary
    total_images = sum(len(outliers) for outliers in all_outliers.values())
    total_outliers = sum(sum(1 for is_outlier in outliers.values() if is_outlier) 
                          for outliers in all_outliers.values())
    
    logger.info(f"\n=== Overall Summary ===")
    logger.info(f"Processed {len(valid_folders)} folders")
    logger.info(f"Total images: {total_images}")
    percentage = (total_outliers/total_images*100) if total_images > 0 else 0
    logger.info(f"Total outliers: {total_outliers} ({percentage:.1f}%)")
    logger.info(f"Total deleted: {len(all_deleted)}")
    
    return all_outliers

def list_folder_labels():
    """List all available folder labels in the database with their image counts."""
    folder_labels = get_folder_labels()
    
    if not folder_labels:
        logger.info("No folder labels found in database")
        return
    
    logger.info("\n=== Available Folder Labels ===")
    logger.info(f"{'Folder Label':<30} | {'Images':<10}")
    logger.info("-" * 43)
    
    for label in sorted(folder_labels):
        count = count_embeddings(label)
        logger.info(f"{label:<30} | {count:<10}")

def main():
    parser = argparse.ArgumentParser(description="Database-enabled face image outlier detection tool")
    
    # Required arguments
    parser.add_argument("--folder-label", type=str, 
                        help="Folder label to analyze (omit to process all folders)")
    
    # Output options
    parser.add_argument("--output", type=str, default=None, help="Directory to save results and outliers")
    
    # Outlier detection options
    parser.add_argument("--threshold", type=float, default=2.0, 
                        help="Number of standard deviations above mean to consider as outlier")
    parser.add_argument("--variance", type=float, default=0.95, 
                        help="Proportion of variance to retain in PCA (0.0-1.0)")
    parser.add_argument("--clusters", type=int, default=None, 
                        help="Number of clusters (default: auto-determine)")
    
    # Visualization options
    parser.add_argument("--no-viz", action="store_false", dest="visualization", 
                        help="Disable visualization generation")
    
    # Deletion options
    parser.add_argument("--delete", action="store_true", 
                        help="Delete outliers from source after copying to review folder")
    parser.add_argument("--force-delete", action="store_true", dest="force", 
                        help="Skip confirmation prompt when deleting outliers")
    
    # Analysis options
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze and report outliers but don't copy or delete any files")
    
    # DB options
    parser.add_argument("--recompute", action="store_true", 
                        help="Recompute outliers even if already marked")
    parser.add_argument("--list-folders", action="store_true", 
                        help="List all available folder labels in the database")
    parser.add_argument("--min-images", type=int, default=5, 
                        help="Minimum number of images required to process a folder (default: 5)")
    
    args = parser.parse_args()
    
    # List folder labels if requested
    if args.list_folders:
        list_folder_labels()
        return 0
    
    # Start timer
    start_time = time.time()
    
    # Process folder(s)
    if args.folder_label:
        # Check if folder label exists
        count = count_embeddings(args.folder_label)
        if count < args.min_images:
            logger.error(f"Not enough images for folder label {args.folder_label} (found {count}, need at least {args.min_images})")
            return 1
        
        # Process single folder
        analyze_embeddings_from_db(
            folder_label=args.folder_label,
            output_dir=args.output,
            outlier_threshold=args.threshold,
            variance_retained=args.variance,
            n_clusters=args.clusters,
            visualization=args.visualization,
            should_delete_outliers=args.delete,
            force_delete=args.force,
            recompute=args.recompute,
            dry_run=args.dry_run
        )
    else:
        # Process all folders
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"outliers_all_folders_{timestamp}"
        else:
            output_dir = args.output
        
        process_all_folders(
            output_dir=output_dir,
            outlier_threshold=args.threshold,
            variance_retained=args.variance,
            n_clusters=args.clusters,
            visualization=args.visualization,
            should_delete_outliers=args.delete,
            force_delete=args.force,
            recompute=args.recompute,
            min_images=args.min_images,
            dry_run=args.dry_run
        )
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
