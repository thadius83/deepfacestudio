# PostgreSQL/pgvector Outlier Detection System

This system provides a database-backed approach to detecting outliers in large image datasets using the PostgreSQL database with pgvector extension for vector similarity operations.

## Advantages

- **Persistence**: Embeddings are stored once and reused, avoiding regeneration
- **Speed**: Fast retrieval of embeddings compared to regenerating from images
- **Scalability**: Can handle millions of embeddings efficiently
- **History**: Maintains a record of which images were flagged as outliers
- **Flexibility**: Can run different clustering parameters without regenerating embeddings

## Setup

### 1. Install PostgreSQL with pgvector

The system requires a PostgreSQL database with the pgvector extension. This is provided as a Docker container in the project:

```bash
# Start the PostgreSQL container
docker-compose up -d db

# Initialize the pgvector extension
docker exec -it postgres_pgvector bash -c "psql -U faceuser -d faces -c 'CREATE EXTENSION IF NOT EXISTS vector;'"

# Apply the database schema
docker cp tools/init_schema.sql postgres_pgvector:/tmp/
docker exec -it postgres_pgvector psql -U faceuser -d faces -f /tmp/init_schema.sql
```

### 2. Install Python Dependencies

```bash
pip install psycopg2-binary pgvector
```

## Usage

### Step 1: Generate and Store Embeddings

Use the `generate_embeddings_db.py` script to generate and store embeddings for your image folders:

```bash
# Process a single folder
python tools/generate_embeddings_db.py --folder /path/to/images --batch-size 32 --workers 6

# Process all subfolders recursively (each subfolder = different person/label)
python tools/generate_embeddings_db.py --folder /data/faces --recursive --min-images 5
```

Options:
- `--folder`: Path to the folder containing images
- `--recursive`: Process all subfolders recursively
- `--batch-size`: Number of images to process in parallel
- `--workers`: Number of worker processes
- `--model`: Model to use for embeddings (default: Facenet512)
- `--detector`: Face detector backend (default: retinaface)
- `--min-images`: Minimum number of images required to process a folder

### Step 2: Detect Outliers

Use the `db_outlier_detection.py` script to detect outliers in stored embeddings:

```bash
# List available folder labels first
python tools/db_outlier_detection.py --list-folders

# Process a single folder
python tools/db_outlier_detection.py --folder-label celebrity_A --output results/celebrity_A

# Process all folders
python tools/db_outlier_detection.py --output results/all_celebrities

# Delete outliers after identification (with confirmation)
python tools/db_outlier_detection.py --folder-label celebrity_A --delete

# Delete without confirmation
python tools/db_outlier_detection.py --folder-label celebrity_A --delete --force-delete
```

Options:
- `--folder-label`: Folder label to analyze (parent directory name)
- `--output`: Directory to save results
- `--threshold`: Standard deviations above mean for outlier detection (default: 2.0)
- `--variance`: Proportion of variance to retain in PCA (default: 0.95)
- `--clusters`: Number of clusters (default: auto-determine)
- `--no-viz`: Disable visualization generation
- `--delete`: Delete outliers after copying
- `--force-delete`: Skip confirmation prompt
- `--recompute`: Recompute outliers even if already marked
- `--list-folders`: List all available folder labels in database

## Files Generated

For each processed folder, the system generates:
- CSV report of all images with their cluster and outlier status
- Visualization of clusters with outliers marked
- A copy of all outlier images for manual review
- Activity log with detailed process information
- Master list of all outlier files

## Examples

### Complete Workflow

```bash
# 1. Start the database
docker-compose up -d db

# 2. Generate embeddings for all people folders 
python tools/generate_embeddings_db.py --folder /data/faces --recursive

# 3. Check available folders
python tools/db_outlier_detection.py --list-folders

# 4. Run outlier detection on specific person
python tools/db_outlier_detection.py --folder-label "brad_pitt" --threshold 2.5

# 5. Review results and delete outliers if needed
python tools/db_outlier_detection.py --folder-label "brad_pitt" --delete
```

### Advanced: Nearest Neighbor Search

You can also use the database for fast nearest neighbor searches. Example in Python:

```python
from backend.app.db import get_conn

def find_similar_images(embedding, limit=10):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT filepath, embedding <-> %s AS distance "
            "FROM image_embeddings ORDER BY embedding <-> %s LIMIT %s;",
            (embedding.tolist(), embedding.tolist(), limit)
        )
        return cur.fetchall()
```

## Notes

- The database stores all embeddings with the folder label (taken from parent directory name)
- You can adjust the vector dimension in the schema if using models other than Facenet512
- For very large datasets, consider tuning PostgreSQL parameters in docker-compose.yml
