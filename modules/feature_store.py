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
    duck_conn.execute("""
        CREATE OR REPLACE TABLE ml_features AS 
        SELECT 
            loan.*,
            macro.gdp_usd,
            macro.frequency AS macro_frequency
        FROM ml_features loan
        LEFT JOIN macro_warehouse_temp macro
            ON loan.country_code = macro.country_code 
            AND CAST(loan.year AS VARCHAR) = CAST(macro.year AS VARCHAR)
    """)
    
    # 4. Fetch and return the newly transformed data
    return duck_conn.execute("SELECT * FROM ml_features").df()