import os
import psycopg2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from psycopg2.extras import register_default_jsonb, execute_values
from pgvector.psycopg2 import register_vector

# Set up configuration values with sensible defaults
DB_CONFIG = {
    "host": os.getenv("PG_HOST", "10.1.10.4"),
    "port": int(os.getenv("PG_PORT", "5432")),
    "user": os.getenv("PG_USER", "faceuser"),
    "password": os.getenv("PG_PASSWORD", "changeme"),
    "dbname": os.getenv("PG_DB", "faces")
}

def get_conn():
    """
    Create and return a database connection with pgvector support.
    
    Returns:
        psycopg2 connection object with autocommit enabled
    """
    # Register JSON support
    register_default_jsonb(globally=True, loads=lambda x: x)
    
    # Create connection
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Enable autocommit
    conn.autocommit = True
    
    # Register pgvector support
    register_vector(conn)
    
    return conn

def bulk_upsert_embeddings(rows: List[Tuple[str, str, List[float]]]):
    """
    Bulk upsert embeddings into the database.
    
    Args:
        rows: List of tuples containing (filepath, folder_label, embedding)
             where embedding is a list of floats
    """
    if not rows:
        return
    
    conn = get_conn()
    
    sql = """
        INSERT INTO image_embeddings (filepath, folder_label, embedding)
        VALUES %s
        ON CONFLICT (filepath)
        DO UPDATE SET embedding = EXCLUDED.embedding,
                      folder_label = EXCLUDED.folder_label,
                      last_updated = now();
    """
    
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    
    conn.close()

def fetch_embeddings(folder_label: Optional[str] = None, 
                    filepaths: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Fetch embeddings from the database.
    
    Args:
        folder_label: Optional folder label to filter by
        filepaths: Optional list of filepaths to filter by
        
    Returns:
        Dictionary mapping filepath to embedding as numpy array
    """
    conn = get_conn()
    embeddings = {}
    
    try:
        with conn.cursor() as cur:
            if folder_label:
                cur.execute(
                    "SELECT filepath, embedding FROM image_embeddings WHERE folder_label = %s",
                    (folder_label,)
                )
            elif filepaths:
                # Create a placeholder string with the right number of parameters
                placeholders = ','.join(['%s'] * len(filepaths))
                cur.execute(
                    f"SELECT filepath, embedding FROM image_embeddings WHERE filepath IN ({placeholders})",
                    filepaths
                )
            else:
                cur.execute("SELECT filepath, embedding FROM image_embeddings")
            
            for filepath, embedding in cur.fetchall():
                embeddings[filepath] = np.array(embedding, dtype=np.float32)
    finally:
        conn.close()
    
    return embeddings

def update_outlier_results(results: List[Tuple[str, int, float, bool]]) -> int:
    """
    Update outlier results in the database.
    
    Args:
        results: List of tuples containing (filepath, cluster_id, distance, is_outlier)
    
    Returns:
        Number of rows updated
    """
    if not results:
        return 0
    
    conn = get_conn()
    
    try:
        with conn.cursor() as cur:
            sql = """
                UPDATE image_embeddings
                SET cluster_id = data.cluster_id,
                    distance_centroid = data.distance,
                    is_outlier = data.outlier,
                    last_updated = now()
                FROM (VALUES %s) AS data(filepath, cluster_id, distance, outlier)
                WHERE image_embeddings.filepath = data.filepath;
            """
            
            execute_values(cur, sql, results)
            return cur.rowcount
    finally:
        conn.close()

def get_outliers(folder_label: str) -> List[str]:
    """
    Get list of outlier filepaths for a specific folder.
    
    Args:
        folder_label: Folder label to filter by
        
    Returns:
        List of filepaths marked as outliers
    """
    conn = get_conn()
    outliers = []
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT filepath FROM image_embeddings WHERE folder_label = %s AND is_outlier = TRUE",
                (folder_label,)
            )
            outliers = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
    
    return outliers

def count_embeddings(folder_label: Optional[str] = None) -> int:
    """
    Count embeddings in the database.
    
    Args:
        folder_label: Optional folder label to filter by
        
    Returns:
        Number of embeddings in the database
    """
    conn = get_conn()
    
    try:
        with conn.cursor() as cur:
            if folder_label:
                cur.execute(
                    "SELECT COUNT(*) FROM image_embeddings WHERE folder_label = %s",
                    (folder_label,)
                )
            else:
                cur.execute("SELECT COUNT(*) FROM image_embeddings")
            
            return cur.fetchone()[0]
    finally:
        conn.close()

def get_folder_labels() -> List[str]:
    """
    Get list of unique folder labels in the database.
    
    Returns:
        List of folder labels
    """
    conn = get_conn()
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT folder_label FROM image_embeddings")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

def delete_embeddings(filepaths: List[str]) -> int:
    """
    Permanently remove embedding rows for the given file paths.

    Args:
        filepaths: List of absolute file paths whose embeddings should be deleted.

    Returns:
        Number of rows removed.
    """
    if not filepaths:
        return 0

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(filepaths))
            cur.execute(
                f"DELETE FROM image_embeddings WHERE filepath IN ({placeholders})",
                filepaths,
            )
            return cur.rowcount
    finally:
        conn.close()


def run_query(query: str, params: Optional[Union[Tuple, List, Dict]] = None) -> List[Tuple]:
    """
    Run a custom query against the database.
    
    Args:
        query: SQL query to run
        params: Optional parameters for the query
        
    Returns:
        List of rows returned by the query
    """
    conn = get_conn()
    
    try:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            if cur.description:  # If this is a SELECT query
                return cur.fetchall()
            return []
    finally:
        conn.close()
