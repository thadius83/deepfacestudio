# PGVector Outlier Detection System – Command Reference

## 1. Database Setup

```bash
# Start PostgreSQL container
docker-compose up -d db

# Enable pgvector
docker exec -it postgres_pgvector psql -U faceuser -d faces \
  -c 'CREATE EXTENSION IF NOT EXISTS vector;'

# Apply 128-dimensional schema
docker cp tools/init_schema_128dim.sql postgres_pgvector:/tmp/
docker exec -it postgres_pgvector psql -U faceuser -d faces \
  -f /tmp/init_schema_128dim.sql
```

---

## 2. Embedding Generation  
`tools/generate_embeddings_db.py`

| Option                | Description                                           | Default |
|-----------------------|-------------------------------------------------------|---------|
| `--folder`            | Path to folder with images **(required)**             | –       |
| `--recursive`         | Process sub-folders recursively                       | False   |
| `--min-images`        | Skip folders with fewer images                        | 5       |
| `--model`             | Face model (e.g. Facenet, Facenet512)                 | Facenet |
| `--detector`          | Detector backend (retinaface, mtcnn, …)              | retinaface |
| `--batch-size`        | Images per DeepFace batch                             | 20      |
| `--workers`           | Parallel workers                                      | 1       |
| `--db-batch-size`     | Rows per DB insert                                    | 1000    |

### Examples
```bash
# Single folder
python tools/generate_embeddings_db.py --folder /data/faces/actor_A

# Recursive over all people
python tools/generate_embeddings_db.py \
  --folder /data/faces --recursive --min-images 10 --workers 4
```

---

## 3. Outlier Detection  
`tools/db_outlier_detection.py`

| Option                 | Description                                         | Default |
|------------------------|-----------------------------------------------------|---------|
| `--folder-label`       | Label (folder name) to analyse. Omit = all labels   | –       |
| `--output`             | Output dir (auto-generated if omitted)              | auto    |
| `--threshold`          | Std-devs above mean to flag outliers                | 2.0     |
| `--variance`           | PCA variance to keep (0-1)                          | 0.95    |
| `--clusters`           | Fixed #clusters (auto if omitted)                   | None    |
| `--no-viz`             | Disable PNG visualisations                          | False   |
| `--delete`             | Delete outliers after copying                       | False   |
| `--force-delete`       | Skip confirmation prompt                            | False   |
| `--recompute`          | Ignore existing DB flags and recompute              | False   |
| `--list-folders`       | List all labels with image counts                   | False   |
| `--min-images`         | Skip labels with fewer images                       | 5       |

### Single-label Examples
```bash
# Analyse one person
python tools/db_outlier_detection.py --folder-label harrison_ford

# Stricter threshold & delete outliers
python tools/db_outlier_detection.py \
  --folder-label harrison_ford --threshold 2.5 --delete
```

### All-labels Examples
```bash
# Process everything, create master summary
python tools/db_outlier_detection.py

# No visualisation, custom output dir
python tools/db_outlier_detection.py --no-viz --output results/all_people
```

---

## 4. Database Helper Functions (`backend/app/db.py`)

| Function                         | Purpose                                         |
|----------------------------------|-------------------------------------------------|
| `get_conn()`                     | Open PG connection (pgvector ready)             |
| `bulk_upsert_embeddings(rows)`   | Insert/Update many embeddings                   |
| `fetch_embeddings(label, files)` | Fetch embeddings by label or specific files     |
| `update_outlier_results(rows)`   | Persist cluster/outlier results                 |
| `get_outliers(label)`            | List outlier filepaths for a label              |
| `count_embeddings(label)`        | Count embeddings (per label / total)            |
| `get_folder_labels()`            | List distinct labels                            |
| `run_query(sql, params)`         | Execute custom SQL                              |

---

## 5. Output Artifacts

Per-label directory (e.g. `db_outliers_harrison_ford_<timestamp>/`)
```
outlier_report.csv      – detailed results
clusters.png            – PCA scatter (outliers marked)
dendrogram.png          – cluster tree (≤100 images)
outliers/               – copied outlier files
outliers_to_delete.txt  – list suitable for bulk deletion
activity_log.txt        – human-readable summary
process.log             – internal log
```
Multi-label runs also include `summary_report.csv` and a master list of outliers.

---

## 6. Quick Workflow

```bash
# 1. Embed all images once
python tools/generate_embeddings_db.py --folder /data/faces --recursive

# 2. Explore available labels
python tools/db_outlier_detection.py --list-folders

# 3. Run detection & review
python tools/db_outlier_detection.py --folder-label "actor_A"

# 4. Confident? Delete outliers
python tools/db_outlier_detection.py --folder-label "actor_A" --delete --force-delete
```

This document captures every command-line option and helper function available in the pgvector-enabled outlier detection system.
