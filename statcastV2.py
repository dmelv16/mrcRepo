import pandas as pd
import requests
import time
from pybaseball import statcast
from sqlalchemy import create_engine, inspect
import numpy as np

# === SQL Server Connection ===
import urllib

# Database connection string for SQLAlchemy
# Format: dialect+driver://username:password@host:port/database
params = urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=DESKTOP-J9IV3OH;"
    "DATABASE=StatcastDB;"
    "UID=mlb_user;"
    "PWD=mlbAdmin;"
    "Encrypt=no;"
    "TrustServerCertificate=yes;"
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")  # change to your DB

df = pd.read_sql("""
    SELECT gamePk, CONVERT(date, game_date) AS date, team_id
    FROM baseballScrapev2
    WHERE YEAR(game_date) = 2025
""", engine)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

# === Get All Players and Starting Pitcher from Team ===
def get_team_players(gamePk, team_id):
    url = f"https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore"
    try:
        r = requests.get(url)
        data = r.json()
        for side in ['home', 'away']:
            team = data['teams'][side]
            if team['team']['id'] == team_id:
                pitchers = []
                batters = []
                starter_id = None
                starter_name = None

                for pid, pdata in team['players'].items():
                    if pdata.get('position', {}).get('code') == '1':
                        pitchers.append(pdata['person']['id'])
                        if pdata.get('stats', {}).get('pitching', {}).get('gamesStarted', 0) == 1:
                            starter_id = pdata['person']['id']
                            starter_name = pdata['person']['fullName']
                    if pdata.get('battingOrder') and 1 <= int(pdata['battingOrder']) <= 900:
                        batters.append(pdata['person']['id'])
                return pitchers, batters, starter_id, starter_name
    except Exception as e:
        print(f"⚠️ Error fetching player info: {e}")
    return [], [], None, None

# === Save to SQL ===
def save_to_sql(df, table_name):
    if df.empty:
        print(f"⚠️ No data to save for {table_name}.")
        return

    try:
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Replace NaNs with None
        df = df.where(pd.notnull(df), None)

        # Deep convert every item to its Python-native type
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x.item() if isinstance(x, np.generic) else x)

        df.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"💾 Saved {len(df)} rows to {table_name}")
    except Exception as e:
        print(f"❌ SQL upload error for {table_name}: {e}")

def summarize_all_statcast_fields(df, gamePk, team_id, starter_id, starter_name, date):
    if df.empty:
        return pd.DataFrame()

    summary_rows = []

    for role, id_col in [('pitcher', 'pitcher'), ('batter', 'batter')]:
        role_df = df[df[id_col].notna()]
        grouped = role_df.groupby(id_col)

        for pid, group in grouped:
            row = {
                'player_id': pid,
                'player_role': role,
                'game_date': date,
                'gamePk': gamePk,
                'team_id': team_id,
                'starting_pitcher_id': starter_id,
                'starting_pitcher_name': starter_name
            }

            numeric_cols = group.select_dtypes(include='number').columns

            for col in numeric_cols:
                try:
                    row[f'{col}_sum'] = group[col].sum()
                    row[f'{col}_mean'] = group[col].mean()
                    row[f'{col}_max'] = group[col].max()
                    row[f'{col}_min'] = group[col].min()
                except Exception as e:
                    print(f"⚠️ Could not summarize column: {col} — {e}")

            summary_rows.append(row)

    return pd.DataFrame(summary_rows)
existing_gamepks = pd.read_sql("""
    SELECT DISTINCT game_pk
    FROM statcast_game_logs
    WHERE YEAR(game_date) = 2025
""", engine)

existing_gamepks_set = set(existing_gamepks['game_pk'])

def game_already_uploaded(table_name, gamePk):
    query = f"""
    SELECT COUNT(*) FROM {table_name}
    WHERE gamePk = ?
    """
    conn = engine.raw_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, (gamePk,))
        count = cursor.fetchone()[0]
        cursor.close()
        return count > 0
    finally:
        conn.close()

# === Main Script ===
def main():
    for _, row in df.iterrows():
        gamePk = row['gamePk']
        team_id = row['team_id']
        date = row['date']

        if gamePk in existing_gamepks_set:
            print(f"✅ Already uploaded: gamePk {gamePk} | team_id {team_id}")
            continue

        print(f"\n📌 Game {gamePk} | Team {team_id} | Date {date}")

        # Get players
        pitchers, batters, starter_id, starter_name = get_team_players(gamePk, team_id)
        if not pitchers or not batters or starter_id is None:
            print("⚠️ Skipping due to missing data.")
            continue

        print(f"🎯 Pitchers: {len(pitchers)} | Batters: {len(batters)} | Starter: {starter_name} ({starter_id})")

        try:
            all_logs = statcast(start_dt=date, end_dt=date)
            print(f"📊 Pulled {len(all_logs)} total Statcast rows for {date}")
        except Exception as e:
            print(f"⚠️ Statcast error: {e}")
            continue

        # Filter only your team’s players
        team_logs = all_logs[
            (all_logs['pitcher'].isin(pitchers)) | 
            (all_logs['batter'].isin(batters))
        ]

        if not team_logs.empty:
            save_to_sql(team_logs, 'statcast_game_logs')
            # summary_df = summarize_all_statcast_fields(
            #     team_logs, gamePk, team_id, starter_id, starter_name, date
            # )
            # save_to_sql(summary_df, 'statcast_game_summaries')
        else:
            print("⚠️ No matching player events found in Statcast data.")

        time.sleep(1.5)

    print("\n✅ All games processed.")

if __name__ == "__main__":
    main()

