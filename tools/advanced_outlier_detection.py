#!/usr/bin/env python3
"""
Advanced face dataset outlier detection tool for DeepFace Suite.

This script analyzes face images to identify potential outliers (mislabeled images)
with advanced features:
1. Parallel processing to utilize multiple CPU cores
2. Dimensionality reduction to speed up comparisons
3. Batch processing to manage memory usage
4. Hierarchical clustering for efficient outlier detection
5. Recursive folder processing
6. Outlier deletion capability
7. Comprehensive logging

Usage:
  python advanced_outlier_detection.py --folder /path/to/face/folder --output /path/to/review/folder
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
import multiprocessing
from multiprocessing import Pool, cpu_count

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
from deepface import DeepFace

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_embedding(args):
    """
    Worker function for parallel processing to generate face embedding for an image.
    
    Args:
        args: Tuple of (image_path, model_name, detector_backend, enforce_detection)
        
    Returns:
        Tuple of (image_path, embedding) or (image_path, None) if no face detected
    """
    image_path, model_name, detector_backend, enforce_detection = args
    
    try:
        # Generate embedding
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=True
        )
        
        if not embedding_objs:
            return image_path, None
            
        # Get the first face embedding
        embedding = embedding_objs[0]["embedding"]
        return image_path, np.array(embedding)
        
    except Exception as e:
        logger.warning(f"Error processing {image_path}: {str(e)}")
        return image_path, None

def count_images(folder_path):
    """
    Count number of valid images in a folder.
    
    Args:
        folder_path: Path to the folder
        
    Returns:
        Number of valid images
    """
    folder_path = Path(folder_path)
    count = 0
    
    for ext in ALLOWED_EXT:
        ext = ext.strip(".")  # Remove leading dot if present
        count += len(list(folder_path.glob(f"*.{ext}")))
    
    return count

def generate_embeddings_parallel(
    image_paths: List[str],
    model_name: str = MODEL_NAME,
    detector_backend: str = DETECTOR_BACKEND,
    batch_size: int = 20,
    max_workers: int = 4
) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for all images in parallel using multiprocessing.
    
    Args:
        image_paths: List of image paths
        model_name: Model to use for embeddings
        detector_backend: Face detector to use
        batch_size: Number of images to process in each batch
        max_workers: Maximum number of worker processes (default: 5)
        
    Returns:
        Dictionary mapping image paths to their embeddings
    """
    # Limit number of workers to avoid system overload
    if max_workers is None or max_workers <= 0:
        max_workers = min(4, max(1, cpu_count() - 1))  # Default: min of 5 or (cores - 1)
    else:
        max_workers = min(max_workers, cpu_count())  # Don't exceed available cores
    
    logger.info(f"Generating embeddings using {max_workers} workers in batches of {batch_size}...")
    
    # Prepare arguments for parallel processing
    args_list = [(path, model_name, detector_backend, False) for path in image_paths]
    
    # Process in batches to avoid memory issues
    embeddings = {}
    total_batches = (len(args_list) + batch_size - 1) // batch_size
    
    with tqdm(total=len(args_list), desc="Generating embeddings") as pbar:
        for i in range(0, len(args_list), batch_size):
            batch_args = args_list[i:i+batch_size]
            
            # Process batch in parallel
            with Pool(processes=max_workers) as pool:
                batch_results = pool.map(get_embedding, batch_args)
            
            # Add successful results to embeddings dict
            for path, embedding in batch_results:
                if embedding is not None:
                    embeddings[path] = embedding
            
            pbar.update(len(batch_args))
    
    logger.info(f"Generated embeddings for {len(embeddings)}/{len(image_paths)} images")
    return embeddings

def apply_dimensionality_reduction(
    embeddings: Dict[str, np.ndarray],
    variance_retained: float = 0.95
) -> Tuple[Dict[str, np.ndarray], PCA]:
    """
    Apply PCA dimensionality reduction to embeddings.
    
    Args:
        embeddings: Dictionary mapping image paths to their embeddings
        variance_retained: Proportion of variance to retain (0.0-1.0)
        
    Returns:
        Dictionary of reduced embeddings, PCA model
    """
    if not embeddings:
        return {}, None
    
    # Stack embeddings into a matrix
    image_paths = list(embeddings.keys())
    embedding_matrix = np.stack([embeddings[path] for path in image_paths])
    
    # Apply PCA
    original_dim = embedding_matrix.shape[1]
    pca = PCA(n_components=variance_retained, svd_solver='full')
    reduced_matrix = pca.fit_transform(embedding_matrix)
    
    # Create dictionary of reduced embeddings
    reduced_dim = reduced_matrix.shape[1]
    reduced_embeddings = {path: reduced_matrix[i] for i, path in enumerate(image_paths)}
    
    logger.info(f"Reduced embedding dimensions from {original_dim} to {reduced_dim} "
                f"(retained {variance_retained:.0%} variance)")
    
    return reduced_embeddings, pca

def perform_hierarchical_clustering(
    embeddings: Dict[str, np.ndarray],
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None
) -> Tuple[AgglomerativeClustering, Dict[str, int]]:
    """
    Perform hierarchical clustering on embeddings.
    
    Args:
        embeddings: Dictionary mapping image paths to their embeddings
        n_clusters: Number of clusters to create (None for auto)
        distance_threshold: Distance threshold for clustering (None for auto)
        
    Returns:
        Clustering model, dictionary mapping image paths to cluster labels
    """
    if not embeddings:
        return None, {}
    
    # Stack embeddings into a matrix
    image_paths = list(embeddings.keys())
    embedding_matrix = np.stack([embeddings[path] for path in image_paths])
    
    # Auto-determine parameters if not provided
    if n_clusters is None and distance_threshold is None:
        # Start with a reasonable number of clusters based on dataset size
        n_clusters = max(2, min(20, len(image_paths) // 5))
    
    # Perform clustering
    if n_clusters is not None:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',
            linkage='ward'
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='euclidean',
            linkage='ward'
        )
    
    cluster_labels = clustering.fit_predict(embedding_matrix)
    
    # Map image paths to cluster labels
    clusters = {path: label for path, label in zip(image_paths, cluster_labels)}
    
    n_clusters = len(set(cluster_labels))
    logger.info(f"Clustered images into {n_clusters} clusters")
    
    return clustering, clusters

def identify_outliers_from_clusters(
    embeddings: Dict[str, np.ndarray],
    clusters: Dict[str, int],
    threshold: float = 2.0
) -> Dict[str, bool]:
    """
    Identify outliers based on distance to cluster centroid.
    
    Args:
        embeddings: Dictionary mapping image paths to their embeddings
        clusters: Dictionary mapping image paths to cluster labels
        threshold: Number of standard deviations above mean to consider as outlier
        
    Returns:
        Dictionary mapping each file to boolean indicating if it's an outlier
    """
    if not embeddings or not clusters:
        return {}
    
    # Group embeddings by cluster
    cluster_embeddings = defaultdict(list)
    cluster_paths = defaultdict(list)
    
    for path, label in clusters.items():
        cluster_embeddings[label].append(embeddings[path])
        cluster_paths[label].append(path)
    
    # Calculate centroid for each cluster
    centroids = {}
    for label, embs in cluster_embeddings.items():
        centroids[label] = np.mean(embs, axis=0)
    
    # Calculate distance to centroid for each image
    distances = {}
    for path, label in clusters.items():
        centroid = centroids[label]
        embedding = embeddings[path]
        distance = np.linalg.norm(embedding - centroid)
        distances[path] = distance
    
    # Identify outliers in each cluster
    outliers = {}
    
    for label, paths in cluster_paths.items():
        # Get distances for this cluster
        cluster_distances = [distances[path] for path in paths]
        
        # Calculate mean and standard deviation
        mean_distance = np.mean(cluster_distances)
        std_distance = np.std(cluster_distances)
        
        # Set threshold for outliers
        outlier_threshold = mean_distance + (threshold * std_distance)
        
        # Identify outliers
        for path in paths:
            outliers[path] = distances[path] > outlier_threshold
    
    # Handle small clusters (size 1 or 2)
    for label, paths in cluster_paths.items():
        if len(paths) <= 2:
            # If the cluster is very small, check if it's far from other clusters
            this_centroid = centroids[label]
            other_centroids = [cent for l, cent in centroids.items() if l != label]
            
            if other_centroids:
                # Calculate minimum distance to other centroids
                min_distance = min(np.linalg.norm(this_centroid - cent) for cent in other_centroids)
                
                # If the entire small cluster is distant from others, mark all as outliers
                if min_distance > np.mean([distances[p] for label, ps in cluster_paths.items() 
                                          for p in ps if len(ps) > 2]):
                    for path in paths:
                        outliers[path] = True
    
    outlier_count = sum(1 for is_outlier in outliers.values() if is_outlier)
    logger.info(f"Found {outlier_count} outliers from {len(embeddings)} images")
    
    return outliers

def visualize_clusters(
    embeddings: Dict[str, np.ndarray],
    clusters: Dict[str, int],
    outliers: Dict[str, bool],
    output_dir: str
) -> None:
    """
    Generate visualization of clusters and outliers.
    
    Args:
        embeddings: Dictionary mapping image paths to their embeddings
        clusters: Dictionary mapping image paths to cluster labels
        outliers: Dictionary mapping each file to boolean indicating if it's an outlier
        output_dir: Directory to save visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # If embeddings are high-dimensional, use PCA to reduce to 2D for visualization
    if next(iter(embeddings.values())).shape[0] > 2:
        # Prepare data for visualization
        paths = list(embeddings.keys())
        X = np.stack([embeddings[path] for path in paths])
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # Create dictionary of 2D coordinates
        coords = {path: X_2d[i] for i, path in enumerate(paths)}
    else:
        # Use first two dimensions if already 2D or lower
        coords = embeddings
    
    # Create scatterplot
    plt.figure(figsize=(12, 10))
    
    # Extract data for plotting
    x_coords = []
    y_coords = []
    colors = []
    markers = []
    
    for path, (x, y) in coords.items():
        x_coords.append(x)
        y_coords.append(y)
        colors.append(clusters[path])
        markers.append('x' if outliers.get(path, False) else 'o')
    
    # Plot normal points and outliers separately
    for marker in ['o', 'x']:
        mask = [m == marker for m in markers]
        if any(mask):
            plt.scatter(
                [x for x, m in zip(x_coords, mask) if m],
                [y for y, m in zip(y_coords, mask) if m],
                c=[c for c, m in zip(colors, mask) if m],
                marker=marker,
                alpha=0.7,
                s=100 if marker == 'x' else 50,
                cmap='tab20'
            )
    
    plt.title("Cluster Visualization (X marks outliers)")
    plt.colorbar(label="Cluster")
    plt.xlabel(f"Principal Component 1")
    plt.ylabel(f"Principal Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "clusters.png", dpi=300)
    plt.close()
    
    # Create dendrogram if not too many points
    if len(embeddings) <= 100:  # Limit for readable dendrogram
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import pdist
        
        plt.figure(figsize=(15, 10))
        
        # Compute hierarchical clustering
        paths = list(embeddings.keys())
        X = np.stack([embeddings[path] for path in paths])
        Z = hierarchy.linkage(X, method='ward')
        
        # Define leaf labels as shortened file names
        labels = [os.path.basename(path) for path in paths]
        if max(len(label) for label in labels) > 20:
            labels = [label[:17] + '...' if len(label) > 20 else label for label in labels]
        
        # Plot dendrogram
        hierarchy.dendrogram(
            Z,
            labels=labels,
            leaf_rotation=90.,
            leaf_font_size=10.,
        )
        
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Images')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(output_path / "dendrogram.png", dpi=300)
        plt.close()

def copy_outliers(
    outliers: Dict[str, bool],
    output_dir: str,
    preserve_structure: bool = True,
    base_folder: str = None
) -> List[str]:
    """
    Copy outlier images to review folder.
    
    Args:
        outliers: Dictionary mapping each file to boolean indicating if it's an outlier
        output_dir: Directory to copy outliers to
        preserve_structure: Whether to preserve folder structure in output
        base_folder: Base folder path for relative path calculation
        
    Returns:
        List of copied outlier file paths
    """
    outlier_files = [f for f, is_outlier in outliers.items() if is_outlier]
    
    if not outlier_files:
        logger.info("No outliers found.")
        return []
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy outlier files to output directory
    logger.info(f"Copying {len(outlier_files)} outliers to {output_dir}...")
    copied_files = []
    
    for file in outlier_files:
        src_path = Path(file)
        
        if preserve_structure and base_folder:
            # Calculate relative path from base folder
            try:
                rel_path = src_path.relative_to(base_folder)
                # Use parent folder name as the target directory
                parent_dir = rel_path.parent
                target_dir = output_path / parent_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                dst_path = target_dir / src_path.name
            except ValueError:
                # Fallback if file is not under base_folder
                dst_path = output_path / src_path.name
        else:
            # Just use the filename
            dst_path = output_path / src_path.name
        
        # Copy file
        try:
            shutil.copy2(file, dst_path)
            copied_files.append(str(file))
            logger.debug(f"Copied {file} to {dst_path}")
        except Exception as e:
            logger.error(f"Error copying {file}: {str(e)}")
    
    logger.info(f"Copied {len(copied_files)} outlier files to {output_dir}")
    return copied_files

def delete_outliers(
    outlier_files: List[str],
    force: bool = False
) -> List[str]:
    """
    Delete outlier files from their original location.
    
    Args:
        outlier_files: List of outlier file paths to delete
        force: Whether to skip confirmation prompt
        
    Returns:
        List of successfully deleted files
    """
    if not outlier_files:
        return []
    
    deleted_files = []
    
    # Confirm deletion unless force is True
    if not force:
        print(f"\nPreparing to delete {len(outlier_files)} outlier files.")
        confirmation = input("Are you sure you want to delete these files? (y/N): ").lower()
        
        if confirmation != 'y':
            logger.info("Deletion cancelled by user.")
            return []
    
    # Delete files
    logger.info(f"Deleting {len(outlier_files)} outlier files...")
    
    for file in outlier_files:
        try:
            os.remove(file)
            deleted_files.append(file)
            logger.debug(f"Deleted {file}")
        except Exception as e:
            logger.error(f"Error deleting {file}: {str(e)}")
    
    logger.info(f"Successfully deleted {len(deleted_files)} outlier files")
    return deleted_files

def generate_report(
    embeddings: Dict[str, np.ndarray],
    clusters: Dict[str, int],
    outliers: Dict[str, bool],
    output_dir: str,
    base_folder: str = None
) -> None:
    """
    Generate a CSV report of the results.
    
    Args:
        embeddings: Dictionary mapping image paths to their embeddings
        clusters: Dictionary mapping image paths to cluster labels
        outliers: Dictionary mapping each file to boolean indicating if it's an outlier
        output_dir: Directory to save report
        base_folder: Base folder path for relative path calculation
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / "outlier_report.csv"
    
    # Calculate distance to cluster centroid
    centroids = {}
    for cluster_id in set(clusters.values()):
        cluster_embeddings = [embeddings[path] for path, label in clusters.items() if label == cluster_id]
        centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
    
    distances = {}
    for path, embedding in embeddings.items():
        cluster_id = clusters[path]
        centroid = centroids[cluster_id]
        distances[path] = np.linalg.norm(embedding - centroid)
    
    # Count cluster sizes
    cluster_sizes = {}
    for cluster_id in set(clusters.values()):
        cluster_sizes[cluster_id] = len([1 for label in clusters.values() if label == cluster_id])
    
    # Sort files by outlier status, then by distance (descending)
    sorted_files = sorted(
        embeddings.keys(),
        key=lambda x: (0 if outliers.get(x, False) else 1, -distances.get(x, 0))
    )
    
    with open(report_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'File', 'Source Folder', 'Filename', 'Cluster', 'Cluster Size', 
            'Distance to Centroid', 'Is Outlier'
        ])
        
        for file in sorted_files:
            path = Path(file)
            source_folder = str(path.parent)
            filename = path.name
            
            # Add relative path if base_folder provided
            rel_path = ""
            if base_folder:
                try:
                    rel_path = str(path.relative_to(base_folder))
                except ValueError:
                    rel_path = str(path)
            
            cluster_id = clusters[file]
            writer.writerow([
                file,
                source_folder,
                filename,
                cluster_id,
                cluster_sizes[cluster_id],
                f"{distances[file]:.6f}",
                "Yes" if outliers.get(file, False) else "No"
            ])
    
    logger.info(f"Report saved to {report_file}")

def write_outliers_list(
    outlier_files: List[str],
    output_dir: str
) -> str:
    """
    Write list of outlier files to a simple text file for batch deletion.
    
    Args:
        outlier_files: List of outlier file paths
        output_dir: Directory to save the file
        
    Returns:
        Path to the created file
    """
    output_path = Path(output_dir)
    list_file = output_path / "outliers_to_delete.txt"
    
    with open(list_file, 'w') as f:
        for file in outlier_files:
            f.write(f"{file}\n")
    
    logger.info(f"List of outliers saved to {list_file}")
    return str(list_file)

def generate_activity_log(
    folder_path: str,
    output_dir: str,
    outliers: Dict[str, bool],
    copied_files: List[str],
    deleted_files: List[str],
    elapsed_time: float,
    stats: Dict = None
) -> None:
    """
    Generate activity log with detailed information about the process.
    
    Args:
        folder_path: Path to the analyzed folder
        output_dir: Output directory
        outliers: Dictionary mapping each file to boolean indicating if it's an outlier
        copied_files: List of copied files
        deleted_files: List of deleted files
        elapsed_time: Total execution time in seconds
        stats: Additional statistics to include
    """
    output_path = Path(output_dir)
    log_file = output_path / "activity_log.txt"
    
    outlier_files = [f for f, is_outlier in outliers.items() if is_outlier]
    
    with open(log_file, 'w') as f:
        f.write(f"=== DeepFace Outlier Detection Activity Log ===\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Folder: {folder_path}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        
        f.write(f"--- Summary Statistics ---\n")
        f.write(f"Total execution time: {elapsed_time:.2f} seconds\n")
        
        # Add provided stats
        if stats:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        # Add outlier stats
        total_images = len(outliers)
        outlier_count = len(outlier_files)
        f.write(f"Total images processed: {total_images}\n")
        f.write(f"Outliers detected: {outlier_count} ({outlier_count/total_images*100:.1f}%)\n")
        f.write(f"Files copied: {len(copied_files)}\n")
        f.write(f"Files deleted: {len(deleted_files)}\n\n")
        
        # List all outliers
        if outlier_files:
            f.write(f"--- Outlier Files ---\n")
            for file in outlier_files:
                f.write(f"{file}\n")
        
        # List deleted files if any
        if deleted_files:
            f.write(f"\n--- Deleted Files ---\n")
            for file in deleted_files:
                f.write(f"{file}\n")
    
    logger.info(f"Activity log saved to {log_file}")

def analyze_face_dataset(
    folder_path: str,
    output_dir: str = None,
    model_name: str = MODEL_NAME,
    detector_backend: str = DETECTOR_BACKEND,
    outlier_threshold: float = 2.0,
    batch_size: int = 20,
    max_workers: int = 4,
    variance_retained: float = 0.95,
    n_clusters: Optional[int] = None,
    visualization: bool = True,
    should_delete_outliers: bool = False,
    force_delete: bool = False,
    preserve_structure: bool = True,
    base_folder: str = None,
) -> Tuple[Dict[str, bool], List[str]]:
    """
    Analyze a face dataset to find potential outliers using optimized methods.
    
    Args:
        folder_path: Path to folder containing face images
        output_dir: Directory to save results and outliers
        model_name: Model to use for embeddings
        detector_backend: Face detector to use
        outlier_threshold: Number of standard deviations above mean to consider as outlier
        batch_size: Number of images to process in each batch
        max_workers: Maximum number of worker processes
        variance_retained: Proportion of variance to retain in PCA
        n_clusters: Number of clusters for hierarchical clustering (None for auto)
        visualization: Whether to generate visualizations
        delete_outliers: Whether to delete outliers after copying
        force_delete: Whether to skip confirmation prompt for deletion
        preserve_structure: Whether to preserve folder structure in output
        base_folder: Base folder path for relative path calculation
        
    Returns:
        Dictionary mapping each file to boolean indicating if it's an outlier,
        List of deleted files
    """
    start_time = time.time()
    stats = {}
    
    # 1. Set up paths
    folder_path = Path(folder_path)
    label_name = folder_path.name
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"fast_outliers_{label_name}_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Analyzing dataset: {folder_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Model: {model_name}, Detector: {detector_backend}")
    
    # Set up file logger for this analysis
    file_handler = logging.FileHandler(output_path / "process.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 2. List all images in the folder
    image_files = []
    for ext in ALLOWED_EXT:
        ext = ext.strip(".")  # Remove leading dot if present
        image_files.extend(list(folder_path.glob(f"*.{ext}")))
    
    if not image_files:
        logger.error(f"No images found in {folder_path}")
        logger.removeHandler(file_handler)
        return {}, []
    
    image_paths = [str(f) for f in image_files]
    logger.info(f"Found {len(image_paths)} images")
    stats["total_images"] = len(image_paths)
    
    # 3. Generate embeddings in parallel with batch processing
    embeddings = generate_embeddings_parallel(
        image_paths,
        model_name=model_name,
        detector_backend=detector_backend,
        batch_size=batch_size,
        max_workers=max_workers
    )
    
    stats["processed_images"] = len(embeddings)
    
    if len(embeddings) < 2:
        logger.error("Not enough valid images with faces detected (need at least 2)")
        logger.removeHandler(file_handler)
        return {}, []
    
    # 4. Apply dimensionality reduction
    reduced_embeddings, pca_model = apply_dimensionality_reduction(
        embeddings,
        variance_retained=variance_retained
    )
    
    # 5. Perform hierarchical clustering
    clustering, clusters = perform_hierarchical_clustering(
        reduced_embeddings,
        n_clusters=n_clusters
    )
    
    stats["clusters"] = len(set(clusters.values()))
    
    # 6. Identify outliers
    outliers = identify_outliers_from_clusters(
        reduced_embeddings,
        clusters,
        threshold=outlier_threshold
    )
    
    outlier_count = sum(1 for is_outlier in outliers.values() if is_outlier)
    stats["outliers"] = outlier_count
    
    # 6.5 Write outlier list file for batch deletion
    outlier_files = [f for f, is_outlier in outliers.items() if is_outlier]
    if outlier_files:
        write_outliers_list(outlier_files, str(output_path))
    
    # 7. Copy outliers to review folder
    review_folder = output_path / "outliers"
    copied_files = copy_outliers(
        outliers, 
        str(review_folder), 
        preserve_structure=preserve_structure,
        base_folder=base_folder
    )
    
    # 8. Delete outliers if requested
    deleted_files = []
    if should_delete_outliers and copied_files:
        deleted_files = delete_outliers(copied_files, force=force_delete)
    
    # 9. Generate report
    generate_report(
        reduced_embeddings, 
        clusters, 
        outliers, 
        str(output_path),
        base_folder=base_folder
    )
    
    # 10. Generate visualizations
    if visualization:
        logger.info("Generating visualizations...")
        visualize_clusters(reduced_embeddings, clusters, outliers, str(output_path))
    
    # 11. Generate activity log
    elapsed_time = time.time() - start_time
    generate_activity_log(
        folder_path=str(folder_path),
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

def find_image_folders(
    base_folder: str,
    min_images: int = 4,
    recursive: bool = False
) -> List[str]:
    """
    Find folders containing image files.
    
    Args:
        base_folder: Base folder to search in
        min_images: Minimum number of images required for a folder to be included
        recursive: Whether to search recursively
        
    Returns:
        List of folder paths containing enough images
    """
    valid_folders = []
    base_path = Path(base_folder)
    
    # Function to check if a folder has enough images
    def has_enough_images(folder_path):
        image_count = count_images(folder_path)
        return image_count >= min_images
    
    if recursive:
        # Walk through all subdirectories
        for root, dirs, files in os.walk(base_folder):
            if has_enough_images(root):
                valid_folders.append(root)
    else:
        # Just check the base folder
        if has_enough_images(base_folder):
            valid_folders.append(base_folder)
    
    return valid_folders

def process_recursive(
    base_folder: str,
    output_dir: str,
    min_images: int = 5,
    should_delete_outliers: bool = False,
    force_delete: bool = False,
    **kwargs
) -> Dict[str, Dict[str, bool]]:
    """
    Process all folders recursively and collect outliers.
    
    Args:
        base_folder: Base folder to process
        output_dir: Base output directory
        min_images: Minimum number of images required to process a folder
        delete_outliers: Whether to delete outliers after copying
        force_delete: Whether to skip confirmation prompt for deletion
        **kwargs: Additional arguments for analyze_face_dataset
        
    Returns:
        Dictionary mapping folder paths to dictionaries of outliers
    """
    # Find all folders with enough images
    folders = find_image_folders(base_folder, min_images=min_images, recursive=True)
    
    if not folders:
        logger.error(f"No folders with at least {min_images} images found")
        return {}
    
    logger.info(f"Found {len(folders)} folders with at least {min_images} images")
    
    # Create main output directory
    main_output_path = Path(output_dir)
    main_output_path.mkdir(parents=True, exist_ok=True)
    
    # Create summary report file
    summary_file = main_output_path / "summary_report.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Folder', 'Total Images', 'Valid Images', 'Outliers', 'Deleted'])
    
    # Process each folder
    all_outliers = {}
    all_deleted = []
    all_stats = {}
    
    for folder in folders:
        logger.info(f"\nProcessing folder: {folder}")
        
        # Calculate relative path for output structure
        rel_path = Path(folder).relative_to(Path(base_folder)) if Path(folder) != Path(base_folder) else ""
        folder_output = main_output_path / rel_path if rel_path else main_output_path / "main"
        
        # Process this folder
        outliers, deleted = analyze_face_dataset(
            folder_path=folder,
            output_dir=str(folder_output),
            should_delete_outliers=should_delete_outliers,
            force_delete=force_delete,
            base_folder=base_folder,
            **kwargs
        )
        
        # Record results
        all_outliers[folder] = outliers
        all_deleted.extend(deleted)
        
        # Update summary report
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            outlier_count = sum(1 for is_outlier in outliers.values() if is_outlier)
            writer.writerow([
                folder,
                len(outliers),
                len(outliers),
                outlier_count,
                len(deleted)
            ])
        
        # Collect stats for overall summary
        folder_stats = {
            'total_images': len(outliers),
            'outliers': outlier_count,
            'deleted': len(deleted)
        }
        all_stats[folder] = folder_stats
    
    # Generate overall summary
    total_images = sum(stats['total_images'] for stats in all_stats.values())
    total_outliers = sum(stats['outliers'] for stats in all_stats.values())
    total_deleted = sum(stats['deleted'] for stats in all_stats.values())
    
    # Write master list of all outliers
    all_outlier_files = []
    for folder_outliers in all_outliers.values():
        all_outlier_files.extend([f for f, is_outlier in folder_outliers.items() if is_outlier])
    
    if all_outlier_files:
        master_list_path = write_outliers_list(all_outlier_files, str(main_output_path))
        logger.info(f"Master list of all outliers saved to {master_list_path}")
    
    logger.info(f"\n=== Overall Summary ===")
    logger.info(f"Processed {len(folders)} folders")
    logger.info(f"Total images: {total_images}")
    percentage = (total_outliers/total_images*100) if total_images > 0 else 0
    logger.info(f"Total outliers: {total_outliers} ({percentage:.1f}%)")
    logger.info(f"Total deleted: {total_deleted}")
    
    return all_outliers

def main():
    parser = argparse.ArgumentParser(description="Advanced face dataset outlier detection tool")
    
    # Required arguments
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing face images")
    
    # Output options
    parser.add_argument("--output", type=str, default=None, help="Directory to save results and outliers")
    
    # Face recognition options
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model to use for embeddings")
    parser.add_argument("--detector", type=str, default=DETECTOR_BACKEND, help="Face detector to use")
    parser.add_argument("--threshold", type=float, default=2.0, 
                        help="Number of standard deviations above mean to consider as outlier")
    
    # Performance options
    parser.add_argument("--batch-size", type=int, default=20, 
                        help="Number of images to process in each batch (default: 20)")
    parser.add_argument("--workers", type=int, default=4, 
                        help="Maximum number of worker processes (default: 4)")
    parser.add_argument("--variance", type=float, default=0.95, 
                        help="Proportion of variance to retain in PCA (0.0-1.0)")
    parser.add_argument("--clusters", type=int, default=None, 
                        help="Number of clusters (default: auto-determine)")
    
    # Visualization options
    parser.add_argument("--no-viz", action="store_false", dest="visualization", 
                        help="Disable visualization generation")
    
    # Safety options
    parser.add_argument("--min-images", type=int, default=5, 
                        help="Minimum number of images required in folder to run analysis (default: 5)")
    
    # Deletion options
    parser.add_argument("--delete", action="store_true", 
                        help="Delete outliers from source after copying to review folder")
    parser.add_argument("--force-delete", action="store_true", dest="force",
                        help="Skip confirmation prompt when deleting outliers")
    
    # Recursive options
    parser.add_argument("--recursive", action="store_true", 
                        help="Process all subfolders recursively")
    
    args = parser.parse_args()
    
    # Make sure folder exists
    if not os.path.isdir(args.folder):
        logger.error(f"Folder not found: {args.folder}")
        return 1
    
    # Process folder(s) based on recursive flag
    if args.recursive:
        # For recursive mode, find valid subfolders first
        valid_folders = find_image_folders(args.folder, min_images=args.min_images, recursive=True)
        if not valid_folders:
            logger.error(f"No folders with at least {args.min_images} images found in {args.folder}")
            return 1
        
        logger.info(f"Found {len(valid_folders)} folders with at least {args.min_images} images")
        process_recursive(
            base_folder=args.folder,
            output_dir=args.output or "outliers_recursive_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            min_images=args.min_images,
            should_delete_outliers=args.delete,
            force_delete=args.force,
            model_name=args.model,
            detector_backend=args.detector,
            outlier_threshold=args.threshold,
            batch_size=args.batch_size,
            max_workers=args.workers,
            variance_retained=args.variance,
            n_clusters=args.clusters,
            visualization=args.visualization
        )
    else:
        # For single folder mode, check if it has enough images
        image_count = count_images(args.folder)
        if image_count < args.min_images:
            logger.error(f"Not enough images in folder (found {image_count}, need at least {args.min_images})")
            return 1
        analyze_face_dataset(
            folder_path=args.folder,
            output_dir=args.output,
            model_name=args.model,
            detector_backend=args.detector,
            outlier_threshold=args.threshold,
            batch_size=args.batch_size,
            max_workers=args.workers,
            variance_retained=args.variance,
            n_clusters=args.clusters,
            visualization=args.visualization,
            should_delete_outliers=args.delete,
            force_delete=args.force
        )
    
    return 0

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Needed for Windows
    sys.exit(main())
