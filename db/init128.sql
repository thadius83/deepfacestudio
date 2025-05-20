-- init_schema_128dim.sql
-- PostgreSQL schema initialization for Facenet 128-dimensional embeddings
-- Drop existing table and recreate with correct dimensions

-- First drop the existing table and its dependencies
DROP TABLE IF EXISTS image_embeddings CASCADE;

-- Main table for storing image embeddings and outlier detection results
CREATE TABLE IF NOT EXISTS image_embeddings (
    id              bigserial PRIMARY KEY,
    filepath        text UNIQUE NOT NULL,
    folder_label    text NOT NULL,           -- parent folder name
    embedding       vector(128) NOT NULL,    -- Changed from 512 to 128 for Facenet
    pca_embedding   vector(64),              -- Reduced vector size proportionally
    cluster_id      integer,
    distance_centroid double precision,
    is_outlier      boolean DEFAULT FALSE,
    inserted_at     timestamptz DEFAULT now(),
    last_updated    timestamptz DEFAULT now()
);

-- ivfflat index for fast similarity search with pgvector
-- (requires pgvector ≥0.4.0 and Postgres ≥14)
CREATE INDEX IF NOT EXISTS idx_embedding_ivfflat
    ON image_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- simple B-tree for folder label queries (improves performance for folder-based lookups)
CREATE INDEX IF NOT EXISTS idx_folder_label
    ON image_embeddings (folder_label);

-- index to quickly find outliers within a folder
CREATE INDEX IF NOT EXISTS idx_outliers
    ON image_embeddings (folder_label, is_outlier)
    WHERE is_outlier = TRUE;
