import sqlite3
import duckdb
import pandas as pd

def apply_macro_layers(duck_conn, selected_layers, data_dir):
    """
    Executes the macro data layering inside the DuckDB warehouse.
    Returns the transformed Pandas DataFrame.
    """
    # If no valid databases were selected, just return the raw features
    if 'macro_layer.db' not in selected_layers:
        return duck_conn.execute("SELECT * FROM ml_features").df()

    # Using the exact absolute path you provided. 
    # The 'r' before the string ensures Python reads the Windows backslashes correctly.
    db_path = r"C:\Users\franc\OneDrive\KBA_2026_PROJECT\credit-risk-automl\Data\Alternative_Data\macro_layer.db"
    
    # 1. Read macro data from SQLite disk storage
    ext_conn = sqlite3.connect(db_path)
    macro_df = pd.read_sql("SELECT * FROM macro_gdp", ext_conn)
    ext_conn.close()

    # 2. Register the macro data into the active DuckDB memory warehouse
    duck_conn.register('macro_warehouse_temp', macro_df)
    
    # 3. Run the SQL Transformation
    # The join logic dynamically checks if the data is annual and matches on the extracted loan year.
    duck_conn.execute("""
        CREATE OR REPLACE TABLE ml_features AS 
        SELECT 
            loan.*,
            macro.* EXCLUDE (country_code, country_name, year, indicator_type)
        FROM ml_features loan
        LEFT JOIN macro_warehouse_temp macro
            ON TRIM(loan.country_code) = TRIM(macro.country_code) 
            AND (
                -- If macro data is Annual, we join strictly by matching the extracted year. Cast to INT to fix 2010.0 != 2010
                (macro.frequency = 'Annual' AND CAST(loan.year AS INTEGER) = CAST(macro.year AS INTEGER))
                OR
                -- Fallback for non-annual or unknown frequency (e.g. if we add Monthly data later)
                (macro.frequency != 'Annual' AND CAST(loan.year AS INTEGER) = CAST(macro.year AS INTEGER))
            )
    """)
    
    # 4. Fetch and return the newly transformed data
    return duck_conn.execute("SELECT * FROM ml_features").df()