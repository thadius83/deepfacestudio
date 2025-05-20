-- PostgreSQL schema initialization for image embeddings with pgvector
-- Run this after creating the 'vector' extension

-- Main table for storing image embeddings and outlier detection results
CREATE TABLE IF NOT EXISTS image_embeddings (
    id              bigserial PRIMARY KEY,
    filepath        text UNIQUE NOT NULL,
    folder_label    text NOT NULL,           -- parent folder name
    embedding       vector(512) NOT NULL,    -- DeepFace models are 128/512 dims; adjust if needed
    pca_embedding   vector(128),             -- optional reduced vector (can be populated later)
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

-- Comments on usage:
-- 1. For finding similar images across all folders:
--    SELECT filepath, embedding <-> %s AS distance 
--    FROM image_embeddings 
--    ORDER BY embedding <-> %s LIMIT 10;
--
-- 2. For finding similar images within a folder:
--    SELECT filepath, embedding <-> %s AS distance 
--    FROM image_embeddings 
--    WHERE folder_label = %s
--    ORDER BY embedding <-> %s LIMIT 10;
--
-- 3. For listing all outliers:
--    SELECT folder_label, COUNT(*) 
--    FROM image_embeddings 
--    WHERE is_outlier = TRUE 
--    GROUP BY folder_label;
