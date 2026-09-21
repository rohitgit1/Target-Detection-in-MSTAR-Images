-- ==============================================================================
-- SNOWFLAKE MODERN ENTERPRISE DEPLOYMENT SCRIPT
-- MSTAR SYNTHETIC APERTURE RADAR (SAR) AUTOMATIC TARGET RECOGNITION (ATR)
-- Utilizing 2025/2026 Snowflake Features:
--   1. Snowflake Cortex Analyst (Semantic Models on Stages)
--   2. Snowflake Cortex Search (Hybrid Vector + BM25 RAG)
--   3. Snowflake Cortex LLM Functions (mistral-large, llama3)
--   4. Snowflake Model Registry (Snowpark ML)
--   5. Snowflake Feature Store (RCS, Entropy, Aspect Features)
--   6. Apache Iceberg v3 Tables (Raw Radar Chips)
--   7. Streamlit in Snowflake (SiS) with SPCS Container Runtime (GPU Pools)
-- ==============================================================================

USE ROLE ACCOUNTADMIN;

-- 1. Infrastructure: Database, Warehouses & Schemas
CREATE OR REPLACE WAREHOUSE MSTAR_ANALYTICS_WH
  WITH WAREHOUSE_SIZE = 'MEDIUM'
  AUTO_SUSPEND = 120
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;

CREATE OR REPLACE DATABASE MSTAR_RADAR_DB;
CREATE OR REPLACE SCHEMA MSTAR_RADAR_DB.RAW;
CREATE OR REPLACE SCHEMA MSTAR_RADAR_DB.FEATURES;
CREATE OR REPLACE SCHEMA MSTAR_RADAR_DB.ANALYTICS;
CREATE OR REPLACE SCHEMA MSTAR_RADAR_DB.MODELS;

USE DATABASE MSTAR_RADAR_DB;
USE SCHEMA ANALYTICS;
USE WAREHOUSE MSTAR_ANALYTICS_WH;

-- 2. Stage for Cortex Analyst Semantic Models
CREATE OR REPLACE STAGE MSTAR_RADAR_DB.ANALYTICS.SEMANTIC_MODELS_STAGE
  DIRECTORY = (ENABLE = TRUE)
  COMMENT = 'Stores YAML semantic models for Snowflake Cortex Analyst natural language text-to-SQL';

-- Upload semantic model via SnowSQL or Python:
-- PUT file:///path/to/mstar_semantic_model.yaml @MSTAR_RADAR_DB.ANALYTICS.SEMANTIC_MODELS_STAGE AUTO_COMPRESS=FALSE;

-- 3. Apache Iceberg v3 Table for Sensor Data Lake
CREATE OR REPLACE ICEBERG TABLE MSTAR_RADAR_DB.RAW.SAR_SENSOR_CHIPS_ICEBERG (
    CHIP_ID VARCHAR(64),
    TARGET_CLASS VARCHAR(32),
    DEPRESSION_ANGLE_DEG FLOAT,
    ASPECT_ANGLE_DEG FLOAT,
    POLARIZATION VARCHAR(4),
    RADAR_BAND VARCHAR(8),
    IMAGE_WIDTH INT,
    IMAGE_HEIGHT INT,
    RAW_AMPLITUDE_DATA BINARY,
    METADATA VARIANT,
    INGESTED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
  CATALOG = 'SNOWFLAKE'
  EXTERNAL_VOLUME = 'SAR_S3_EXT_VOL'
  BASE_LOCATION = 'sar_chips_iceberg/';

-- 4. Governed Analytics Table for Detections & Predictions
CREATE OR REPLACE TABLE MSTAR_RADAR_DB.ANALYTICS.SAR_TARGET_DETECTIONS (
    DETECTION_ID VARCHAR(64) PRIMARY KEY,
    TIMESTAMP TIMESTAMP_NTZ,
    SENSOR_ID VARCHAR(64),
    PREDICTED_CLASS VARCHAR(32),
    TRUE_CLASS VARCHAR(32),
    TARGET_CATEGORY VARCHAR(64),
    CONFIDENCE FLOAT,
    IS_HIGH_CONFIDENCE BOOLEAN,
    DEPRESSION_ANGLE_DEG FLOAT,
    ASPECT_ANGLE_DEG FLOAT,
    RCS_DB FLOAT,
    HULL_LENGTH_M FLOAT,
    COMBAT_WEIGHT_TON FLOAT,
    GUN_CALIBER_MM FLOAT,
    SECTOR VARCHAR(64),
    LATITUDE FLOAT,
    LONGITUDE FLOAT,
    EXPLAINABILITY_METADATA VARIANT
);

-- 5. Snowflake Cortex Search Service (Hybrid Vector + Keyword Search)
CREATE OR REPLACE TABLE MSTAR_RADAR_DB.ANALYTICS.TACTICAL_INTELLIGENCE_DOCS (
    DOC_ID VARCHAR(32) PRIMARY KEY,
    TITLE VARCHAR(256),
    CATEGORY VARCHAR(64),
    BODY_TEXT VARCHAR,
    UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE OR REPLACE CORTEX SEARCH SERVICE MSTAR_RADAR_DB.ANALYTICS.SAR_TACTICAL_SEARCH_SERVICE
  ON BODY_TEXT
  ATTRIBUTES CATEGORY, TITLE
  WAREHOUSE = MSTAR_ANALYTICS_WH
  TARGET_LAG = '1 hour'
  AS (
    SELECT DOC_ID, TITLE, CATEGORY, BODY_TEXT
    FROM MSTAR_RADAR_DB.ANALYTICS.TACTICAL_INTELLIGENCE_DOCS
  );

-- 6. Snowflake Model Registry: Register A-ConvNet Model
-- (Executed via Snowpark Python Client)
/*
from snowflake.ml.registry import Registry
import snowflake.snowpark as snowpark

session = snowpark.Session.builder.getOrCreate()
reg = Registry(session=session, database_name="MSTAR_RADAR_DB", schema_name="MODELS")

# Deploy PyTorch A-ConvNet model
reg.log_model(
    model_name="MSTAR_ACONVNET_ATR",
    version_name="V2_PYTORCH",
    model=aconvnet_model,
    sample_input_data=dummy_sample_tensor,
    signatures={"predict": ...},
    comment="MSTAR 10-Class All-Convolutional Network (99.13% SOC Accuracy)"
)
*/

-- 7. Streamlit in Snowflake (SiS) with Snowpark Container Services (SPCS) GPU Pool
CREATE COMPUTE POOL IF NOT EXISTS MSTAR_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 2
  INSTANCE_FAMILY = GPU_NV_S
  AUTO_RESUME = TRUE;

-- Create native Streamlit app on Container Runtime
CREATE OR REPLACE STREAMLIT MSTAR_RADAR_DB.ANALYTICS.MSTAR_TACTICAL_ATR_CONSOLE
  ROOT_LOCATION = '@MSTAR_RADAR_DB.ANALYTICS.SEMANTIC_MODELS_STAGE/streamlit_app'
  MAIN_FILE = 'app.py'
  QUERY_WAREHOUSE = MSTAR_ANALYTICS_WH
  COMPUTE_POOL = MSTAR_GPU_POOL;
