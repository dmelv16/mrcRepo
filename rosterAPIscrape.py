import pandas as pd
import requests
from datetime import datetime
from sqlalchemy import create_engine
import urllib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- CONFIG ---
SQL_SERVER = "DESKTOP-J9IV3OH"
SQL_DATABASE = "StatcastDB"
SOURCE_TABLE = "baseballScrapev2"  # Your games table
ROSTER_TABLE = "roster_data"
MAX_WORKERS = 4  # Adjust based on your machine
START_YEAR = 2017  # Starting from 2017 as requested

# --- SQL Server Connection ---
connection_string = (
    "mssql+pyodbc://DESKTOP-J9IV3OH/StatcastDB"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&trusted_connection=yes"
)
engine = create_engine(connection_string)

# --- Load games from baseballScrapev2 table ---
print(f"📊 Loading games from {SOURCE_TABLE} starting from {START_YEAR}...")
query = f"""
SELECT DISTINCT 
    team_id,
    CAST(game_date AS DATE) as date
FROM {SOURCE_TABLE}
WHERE YEAR(game_date) >= {START_YEAR}
"""

team_dates = pd.read_sql(query, engine)
team_dates['date'] = pd.to_datetime(team_dates['date'])
print(f"📋 Found {len(team_dates)} unique team-date combinations")

# --- Load already scraped team-date combos from roster table ---
print(f"🔍 Checking existing rosters in {ROSTER_TABLE}...")
try:
    existing_query = f"""
    SELECT DISTINCT 
        team_id, 
        CAST(date AS DATE) as date 
    FROM {ROSTER_TABLE}
    """
    existing = pd.read_sql(existing_query, engine)
    existing['date'] = pd.to_datetime(existing['date'])
    print(f"✅ Found {len(existing)} existing team-date combinations")
except Exception as e:
    print(f"⚠️ No existing roster data found or table doesn't exist: {e}")
    existing = pd.DataFrame(columns=['team_id', 'date'])

# --- Remove existing pairs ---
team_dates = team_dates.merge(existing, on=['team_id', 'date'], how='left', indicator=True)
team_dates = team_dates[team_dates['_merge'] == 'left_only'].drop(columns=['_merge'])
print(f"🎯 {len(team_dates)} team-date combinations need to be scraped")

# --- Define threaded roster fetcher ---
def fetch_and_store_roster(row):
    team_id = row['team_id']
    date_str = row['date'].strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?date={date_str}"

    try:
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.1)
        
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            print(f"⚠️ Status {res.status_code} for team {team_id} on {date_str}")
            return 0

        data = res.json()
        players = []
        
        # Check if roster data exists
        if "roster" not in data or not data["roster"]:
            return 0
            
        for p in data.get("roster", []):
            # Ensure all required fields exist
            if "person" in p and "position" in p:
                players.append({
                    "player_id": p["person"]["id"],
                    "full_name": p["person"]["fullName"],
                    "team_id": team_id,
                    "position": p["position"].get("abbreviation", "N/A"),
                    "jersey_number": p.get("jerseyNumber", None),
                    "status": p.get("status", {}).get("description", "Active"),
                    "date": date_str
                })

        if players:
            roster_df = pd.DataFrame(players)
            # Convert date to datetime for SQL Server
            roster_df['date'] = pd.to_datetime(roster_df['date'])
            roster_df.to_sql(ROSTER_TABLE, engine, if_exists='append', index=False, method='multi')
            return len(roster_df)
        return 0

    except requests.exceptions.RequestException as e:
        print(f"🌐 Network error for team {team_id} on {date_str}: {e}")
        return 0
    except Exception as e:
        print(f"❌ Error fetching for team {team_id} on {date_str}: {e}")
        return 0

# --- Create roster table if it doesn't exist ---
create_table_query = f"""
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{ROSTER_TABLE}' AND xtype='U')
CREATE TABLE {ROSTER_TABLE} (
    player_id INT NOT NULL,
    full_name NVARCHAR(255),
    team_id INT NOT NULL,
    position NVARCHAR(10),
    jersey_number INT,
    status NVARCHAR(50),
    date DATE NOT NULL,
    inserted_at DATETIME DEFAULT GETDATE()
)
"""
try:
    with engine.connect() as conn:
        conn.execute(create_table_query)
        conn.commit()
    print(f"✅ Ensured {ROSTER_TABLE} table exists")
except Exception as e:
    print(f"⚠️ Table creation warning: {e}")

# --- Use concurrent workers ---
if len(team_dates) > 0:
    print(f"🚀 Starting roster scraping with {MAX_WORKERS} workers...")
    
    # Sort by date (oldest first) for better progress tracking
    team_dates = team_dates.sort_values('date')
    
    total_inserted = 0
    batch_size = 1000  # Process in batches to avoid memory issues
    
    for i in range(0, len(team_dates), batch_size):
        batch = team_dates.iloc[i:i+batch_size]
        print(f"\n📦 Processing batch {i//batch_size + 1} of {(len(team_dates)-1)//batch_size + 1}")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(fetch_and_store_roster, row) for _, row in batch.iterrows()]
            
            with tqdm(total=len(futures), desc="Fetching rosters") as pbar:
                for future in as_completed(futures):
                    inserted = future.result()
                    total_inserted += inserted
                    pbar.update(1)
                    pbar.set_postfix({'Total Players': total_inserted})
    
    print(f"\n✅ Done! Inserted {total_inserted} total player records into {ROSTER_TABLE}.")
    
else:
    print("✅ All rosters are already up to date!")