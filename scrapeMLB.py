import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import re

BASE_URL = "https://statsapi.mlb.com/api/v1"

def get_schedule(start_date, end_date):
    url = f"{BASE_URL}/schedule"
    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date
    }
    response = requests.get(url, params=params)
    data = response.json()
    games = []
    for date in data.get('dates', []):
        for game in date.get('games', []):
            if game.get('gameType') == 'R':  # Only include regular season games
                games.append({
                    'gamePk': game['gamePk'],
                    'gameDate': game['gameDate']
                })
    return pd.DataFrame(games)

def get_weather_and_venue(gamePk):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"
    response = requests.get(url)
    if not response.ok:
        print(f"Failed to get feed/live data for gamePk {gamePk}")
        return {
            'venue': None,
            'game_time': None,
            'dayNight': None,
            'temperature': None,
            'wind_speed': None,
            'wind_dir': None,
            'conditions': None,
            'game_date': None
        }

    data = response.json()
    game_data = data.get('gameData', {})

    # Venue and time
    venue = game_data.get('venue', {}).get('name')
    iso_time = game_data.get('datetime', {}).get('dateTime')
    official_date = game_data.get('datetime', {}).get('officialDate')
    day_night = game_data.get('datetime', {}).get('dayNight')

    # Format game time (HH:MM)
    game_time = None
    if iso_time:
        try:
            dt = datetime.strptime(iso_time, "%Y-%m-%dT%H:%M:%SZ")
            game_time = dt.strftime("%H:%M")
        except:
            pass

    # Weather
    weather = game_data.get('weather', {})
    temp = weather.get('temp')
    wind = weather.get('wind')
    conditions = weather.get('condition')

    wind_speed, wind_dir = None, None
    if wind:
        wind_match = re.match(r"(\d+)\s*mph(?:,\s*(.*))?", wind)
        if wind_match:
            wind_speed = int(wind_match.group(1))
            wind_dir = wind_match.group(2)

    return {
        'venue': venue,
        'game_time': game_time,
        'dayNight': day_night,
        'temperature': int(temp) if temp and temp.isdigit() else None,
        'wind_speed': wind_speed,
        'wind_dir': wind_dir,
        'conditions': conditions,
        'game_date': official_date  # Added here
    }

def get_boxscore_stats(gamePk):
    url = f"{BASE_URL}/game/{gamePk}/boxscore"
    response = requests.get(url)
    if not response.ok:
        return pd.DataFrame()
    data = response.json()
    rows = []
    for side in ['home', 'away']:
        team_data = data.get('teams', {}).get(side, {})
        stats = team_data.get('teamStats', {})
        row = {
            'gamePk': gamePk,
            'team': team_data.get('team', {}).get('name'),
            'team_id': team_data.get('team', {}).get('id'),
            'side': side
        }
        for category in ['batting', 'pitching', 'fielding']:
            for k, v in stats.get(category, {}).items():
                row[f"{category}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    # 1. SQL setup 
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
    # Get existing gamePks from the table
    existing_gamepks = pd.read_sql("SELECT DISTINCT gamePk FROM baseballScrapev2", con=engine)
    existing_gamepks_set = set(existing_gamepks['gamePk'].astype(int).tolist())
    # 2. Loop seasons
    start = datetime(2025, 4, 1)
    today = datetime.today()  # Only date part
    # We'll batch by month to avoid huge single calls
    current = start
    while current < today:
        end = min(current + timedelta(days=30), today - timedelta(days=1))
        print(f"Processing {current.date()} to {end.date()}")
        schedule = get_schedule(current.strftime('%Y-%m-%d'),
                                end.strftime('%Y-%m-%d'))
        for _, g in schedule.iterrows():
            gp = int(g.gamePk)
            # Skip if gamePk already in database
            if gp in existing_gamepks_set:
                continue
            # Weather & venue
            info = get_weather_and_venue(gp)

            # Batting stats
            stats_df = get_boxscore_stats(gp)
            # Calculate OBP, SLG, OPS
            if not stats_df.empty:
                # Fill NaNs with zeros for safety
                stats_df = stats_df.fillna(0)

                # Rename for brevity
                ab = stats_df['batting_atBats']
                h = stats_df['batting_hits']
                bb = stats_df['batting_baseOnBalls']
                hbp = stats_df.get('batting_hitByPitch', 0)
                sf = stats_df.get('batting_sacFlies', 0)
                hr = stats_df['batting_homeRuns']
                doubles = stats_df['batting_doubles']
                triples = stats_df['batting_triples']
                singles = h - doubles - triples - hr

                # OBP
                obp_denom = ab + bb + hbp + sf
                stats_df['OBP'] = (h + bb + hbp) / obp_denom.replace(0, pd.NA)

                # SLG
                tb = (singles + 2*doubles + 3*triples + 4*hr)
                stats_df['SLG'] = tb / ab.replace(0, pd.NA)

                # OPS
                stats_df['OPS'] = stats_df['OBP'] + stats_df['SLG']
                stats_df[['OBP', 'SLG', 'OPS']] = stats_df[['OBP', 'SLG', 'OPS']].round(3)
            # Merge metadata
            for k,v in info.items():
                stats_df[k] = v
            # Write to SQL
            stats_df.to_sql('baseballScrapev2', con=engine,
                            if_exists='append', index=False)
        current = end + timedelta(days=1)

if __name__ == '__main__':
    main()
