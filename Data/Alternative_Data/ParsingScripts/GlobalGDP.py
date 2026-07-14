import wbgapi as wb
import pandas as pd
import sqlite3

def fetch_and_store_macro_data():
    indicator = 'NY.GDP.MKTP.CD' # GDP (current US$)
    
    print("Fetching data from World Bank API...")
    # Fetch data. time=range(2010, 2024)
    df_wide = wb.data.DataFrame(indicator, time=range(2010, 2024), labels=True)
    
    # Clean the year columns (e.g., 'YR2010' -> '2010')
    df_wide.columns = [str(col).replace('YR', '') for col in df_wide.columns]
    
    # Filter out regional aggregates to only keep sovereign countries
    economies = wb.economy.DataFrame()
    countries_only = economies[economies['aggregate'] == False].index
    df_wide = df_wide[df_wide.index.isin(countries_only)]
    
    # --- RESHAPE TO LONG FORMAT FOR SQL ---
    # We move the index (Country code) to a normal column first
    df_wide = df_wide.reset_index()
    df_wide.rename(columns={'economy': 'country_code', 'Country': 'country_name'}, inplace=True)
    
    # Melt the dataframe: collapse all the year columns into two columns ('year' and 'gdp_value')
    # This makes it infinitely easier to JOIN with your loan panel data later
    df_long = df_wide.melt(
        id_vars=['country_code', 'country_name'], 
        var_name='year', 
        value_name='gdp_usd'
    )
    
    # --- ADD METADATA ---
    # The World Bank GDP data is annual. We add the column here.
    df_long['frequency'] = 'Annual'
    df_long['indicator_type'] = 'GDP'
    
    # Clean up any missing data (countries that don't report GDP for certain years)
    df_long = df_long.dropna(subset=['gdp_usd'])
    
    print(f"Reshaped into {len(df_long)} rows of long-format panel data.")
    
    # --- SAVE TO SQLITE ---
    db_name = 'macro_layer.db'
    print(f"Connecting to {db_name}...")
    
    # Create a physical SQLite database file that your app can call on later
    conn = sqlite3.connect(db_name)
    
    # Write the dataframe to a table named 'macro_gdp'
    # if_exists='replace' ensures you don't duplicate rows if you run the script twice
    df_long.to_sql('macro_gdp', conn, if_exists='replace', index=False)
    
    # Create an index on country and year to make your future JOIN operations lightning fast
    conn.execute("CREATE INDEX IF NOT EXISTS idx_country_year ON macro_gdp(country_code, year)")
    
    conn.close()
    print("Successfully committed to the SQLite database.")

# --- Execution ---
if __name__ == "__main__":
    fetch_and_store_macro_data()