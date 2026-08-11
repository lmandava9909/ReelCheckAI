CREATE TABLE IF NOT EXISTS reelcheck_lakehouse.silver_reel_semantic_chunks (
  chunk_id STRING,
  context_id STRING,
  submission_id STRING,
  shortcode STRING,
  chunk_type STRING,
  chunk_text STRING,
  source_start_seconds DOUBLE,
  source_end_seconds DOUBLE,
  claim_object_id STRING,
  content_quality_score DOUBLE,
  created_at TIMESTAMP
) USING DELTA;
