"""
feature_store.py
DuckDB In-Memory Analytical Warehouse & Feature Store Connector.
Manages metadata tag catalogs, dynamic SQL macro layering, and DB housekeeping.
"""

import sqlite3
import duckdb
import pandas as pd
import os
from typing import List, Optional


def init_duckdb_warehouse(duck_conn: duckdb.DuckDBPyConnection) -> None:
    """Initializes PRAGMA settings and metadata catalog table inside DuckDB."""
    duck_conn.execute("PRAGMA memory_limit = '4GB';")
    duck_conn.execute("PRAGMA threads = 4;")

    duck_conn.execute("""
        CREATE TABLE IF NOT EXISTS kba_feature_metadata_catalog (
            feature_code VARCHAR PRIMARY KEY,
            name_tag VARCHAR NOT NULL,             -- 'Loan Panel', 'M-Pesa Core', 'Macroeconomic', etc.
            time_period VARCHAR NOT NULL,          -- '30d', '90d', '12m', 'Lifetime', etc.
            data_type_tag VARCHAR NOT NULL,        -- 'Behavioral', 'Transactional', 'Macroeconomic', etc.
            pii_level VARCHAR NOT NULL,            -- 'Zero-PII / Public', 'Consented PII', etc.
            sql_data_type VARCHAR NOT NULL,        -- 'FLOAT', 'INTEGER', 'BOOLEAN'
            iv_band VARCHAR NOT NULL               -- 'Very Strong', 'Strong', 'Medium', 'Weak'
        );
    """)


def apply_macro_layers(duck_conn: duckdb.DuckDBPyConnection, selected_layers: List[str], data_dir: str) -> pd.DataFrame:
    """
    Executes vectorized SQL joins inside DuckDB across selected macro & alternative databases.
    Returns the transformed Pandas DataFrame.
    """
    init_duckdb_warehouse(duck_conn)

    if not selected_layers or 'macro_layer.db' not in selected_layers:
        return duck_conn.execute("SELECT * FROM ml_features").df()

    db_path = os.path.join(data_dir, "macro_layer.db")
    if not os.path.exists(db_path):
        return duck_conn.execute("SELECT * FROM ml_features").df()

    # 1. Read macro reference layer from SQLite
    ext_conn = sqlite3.connect(db_path)
    macro_df = pd.read_sql("SELECT * FROM macro_gdp", ext_conn)
    ext_conn.close()

    # 2. Register macro table in DuckDB memory
    duck_conn.register('macro_warehouse_temp', macro_df)

    # 3. Vectorized Dynamic SQL Join
    duck_conn.execute("""
        CREATE OR REPLACE TABLE ml_features AS 
        SELECT 
            loan.*,
            macro.* EXCLUDE (country_code, country_name, year, indicator_type)
        FROM ml_features loan
        LEFT JOIN macro_warehouse_temp macro
            ON TRIM(loan.country_code) = TRIM(macro.country_code) 
            AND (
                (macro.frequency = 'Annual' AND CAST(loan.year AS INTEGER) = CAST(macro.year AS INTEGER))
                OR
                (macro.frequency != 'Annual' AND CAST(loan.year AS INTEGER) = CAST(macro.year AS INTEGER))
            );
    """)

    return duck_conn.execute("SELECT * FROM ml_features").df()


def export_parquet_snapshot(duck_conn: duckdb.DuckDBPyConnection, output_filepath: str) -> bool:
    """Exports current DuckDB feature store table to compressed ZSTD Parquet lakehouse storage."""
    try:
        duck_conn.execute(f"COPY ml_features TO '{output_filepath}' (FORMAT PARQUET, COMPRESSION ZSTD);")
        return True
    except Exception:
        return False