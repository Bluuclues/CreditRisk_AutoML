import wbgapi as wb
import sqlite3
import os

def build_country_db():
    print("Fetching country list from World Bank...")
    
    # Fetch all economies
    economies = wb.economy.DataFrame()
    
    # Filter to only include actual sovereign countries (aggregate = False)
    countries_only = economies[economies['aggregate'] == False].copy()
    
    # Reset index so the ISO-3 code (currently the index) becomes a normal column
    countries_only = countries_only.reset_index()
    countries_only.rename(columns={'id': 'country_code', 'name': 'country_name'}, inplace=True)
    
    # Keep only the two columns we need
    df_mapping = countries_only[['country_name', 'country_code']]
    
    # Define your exact path
    target_dir = r"C:\Users\franc\OneDrive\KBA_2026_PROJECT\credit-risk-automl\Data\Alternative_Data"
    os.makedirs(target_dir, exist_ok=True)
    
    db_path = os.path.join(target_dir, 'countries.db')
    
    # Save to SQLite
    conn = sqlite3.connect(db_path)
    df_mapping.to_sql('country_mapping', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"Successfully saved {len(df_mapping)} countries to {db_path}")

if __name__ == "__main__":
    build_country_db()