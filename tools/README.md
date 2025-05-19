# DeepFace Suite Tools

This directory contains utility tools for the DeepFace Suite project.

## Face Dataset Outlier Detection Tools

We provide three different tools for identifying potential outliers (mislabeled images) in your face reference database:

### 1. Compare Folder Tool (Recommended)

The `compare_folder.py` script is a simpler and more robust tool that uses DeepFace's built-in verification to compare all images in a folder.

### Features

- Uses DeepFace's verify function for reliable face comparison
- Performs pairwise comparisons between all images
- Identifies outliers based on statistical analysis
- Generates detailed CSV reports of all comparisons
- Creates histogram visualizations of similarity distribution
- Copies outlier images to a review folder for manual inspection

### Usage

```bash
python tools/compare_folder.py --folder /path/to/face/folder [OPTIONS]
```

### Options

- `--folder`: Path to the folder containing face images (required)
- `--output`: Directory to save results and outliers (default: auto-generated based on folder name)
- `--model`: Model to use for face recognition (default: from config, typically "Facenet")
- `--detector`: Face detector to use (default: from config, typically "retinaface")
- `--metric`: Distance metric to use, either "cosine" or "euclidean" (default: "euclidean")
- `--threshold`: Number of standard deviations above mean to consider as outlier (default: 2.0)
- `--no-viz`: Disable visualization generation

### Example

```bash
python tools/compare_folder.py --folder /workspace/data/reference_db/ice_cube --threshold 1.5
```

### 2. Outlier Detection Tool (Alternative)

The `outlier_detection.py` script uses a different approach that works with face embeddings directly.

#### Features

- Generates face embeddings for all images in a specified folder
- Calculates pairwise distances between all embeddings
- Identifies outliers based on statistical analysis
- Creates visualizations of the similarity matrix
- Generates detailed CSV reports of the results
- Copies outlier images to a review folder for manual inspection

#### Usage

```bash
python tools/outlier_detection.py --folder /path/to/face/folder [OPTIONS]
```

#### Options

Same as compare_folder.py, except the default distance metric is "cosine".

## Output

Both tools generate the following outputs in the specified directory:

- `outliers/` - Directory containing copies of the identified outlier images
- `outliers.csv` - CSV file with average distances and outlier status for each image
- `comparisons.csv` - (compare_folder.py only) Detailed list of all pairwise comparisons
- `distance_matrix.png` - (outlier_detection.py only) Heatmap visualization of the pairwise distances
- `distance_histogram.png` - Histogram of average distances with outlier threshold

### 3. Fast Outlier Detection Tool (Optimized)

The `fast_outlier_detection.py` script is an optimized tool that incorporates advanced techniques to analyze large datasets efficiently.

#### Features

- Parallel processing to utilize multiple CPU cores simultaneously
- Dimensionality reduction (PCA) to speed up comparisons
- Batch processing to manage memory usage with large datasets
- Hierarchical clustering for efficient outlier detection
- Advanced visualizations including cluster plots and dendrograms
- Comprehensive statistics and reporting

#### Usage

```bash
python tools/fast_outlier_detection.py --folder /path/to/face/folder [OPTIONS]
```

#### Options

- `--folder`: Path to the folder containing face images (required)
- `--output`: Directory to save results and outliers (default: auto-generated)
- `--model`: Model to use for embeddings (default: from config)
- `--detector`: Face detector to use (default: from config)
- `--threshold`: Number of standard deviations above mean to consider as outlier (default: 2.0)
- `--batch-size`: Number of images to process in each batch (default: 20)
- `--workers`: Maximum number of worker processes (default: use all available cores)
- `--variance`: Proportion of variance to retain in PCA (default: 0.95)
- `--clusters`: Number of clusters to use (default: auto-determine)
- `--no-viz`: Disable visualization generation

#### Example

```bash
python tools/fast_outlier_detection.py --folder /workspace/data/reference_db/ice_cube --batch-size 30 --workers 4
```

### 4. Advanced Outlier Detection Tool (Latest and Recommended)

The `advanced_outlier_detection.py` script is the most comprehensive tool, with all features from the optimized version plus additional capabilities:

#### Features

- All features of fast_outlier_detection.py, plus:
- Recursive processing of entire directory structures
- Safe minimum image requirements (won't process folders with too few images)
- Option to delete outliers after copying to review folder
- Preserves folder structure in output for recursive operations
- Comprehensive activity logging with detailed statistics
- Enhanced reporting with source folder information

#### Usage

```bash
python tools/advanced_outlier_detection.py --folder /path/to/face/folder [OPTIONS]
```

#### Options

All options from fast_outlier_detection.py, plus:
- `--recursive`: Process all subfolders recursively
- `--min-images`: Minimum number of images required in folder (default: 5)
- `--delete`: Delete outliers from source after copying to review folder
- `--force`: Skip confirmation prompt when deleting outliers

#### Example

```bash
# Process a full reference database recursively
python tools/advanced_outlier_detection.py --folder /workspace/data/reference_db --recursive --threshold 1.8

# Process a folder and delete outliers after confirming
python tools/advanced_outlier_detection.py --folder /workspace/data/reference_db/ice_cube --delete
```

## Which Tool Should I Use?

- **advanced_outlier_detection.py** (recommended for all use cases):
  - Most complete tool with all optimizations and additional features
  - Handles both single folders and recursive processing
  - Includes advanced safety features and better reporting
  - Best choice for routine quality control of the reference database

- **fast_outlier_detection.py** (for large single folders):
  - Fast processing for large datasets with parallel processing
  - Provides advanced visualizations and clustering analysis
  - Good choice for datasets with more than 50 images

- **compare_folder.py** (for small datasets): 
  - More reliable as it uses DeepFace's verified comparison function
  - Simple and straightforward for smaller datasets
  - Default distance metric is euclidean (same as used in DeepFace.verify)

- **outlier_detection.py** (legacy):
  - Basic implementation using embeddings directly
  - May be less reliable as it bypasses some of DeepFace's verification logic
  - Provides detailed matrix visualization

## Tips for Both Tools

1. Start with the default threshold (2.0) and adjust as needed
   - Lower values (e.g., 1.5) are more sensitive and will identify more potential outliers
   - Higher values (e.g., 2.5) are more conservative and will only identify the most extreme outliers

2. Use the generated visualizations to guide your threshold selection:
   - The histogram shows the distribution of average distances
   - The red dashed line indicates the current threshold

3. Review the outlier images manually to determine which ones are actually mislabeled
