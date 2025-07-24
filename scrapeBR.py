import cloudscraper  # new import
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random
from io import StringIO

teams = {
    'ari': 'ARI',
    'atl': 'ATL',
    'bal': 'BAL',
    'bos': 'BOS',
    'chc': 'CHC',
    'chw': 'CHW',
    'cin': 'CIN',
    'cle': 'CLE',
    'col': 'COL',
    'det': 'DET',
    'hou': 'HOU',
    'kcr': 'KCR',
    'laa': 'LAA',
    'lad': 'LAD',
    'mia': 'MIA',
    'mil': 'MIL',
    'min': 'MIN',
    'nym': 'NYM',
    'nyy': 'NYY',
    # 'oak': 'OAK',     # ✅ keep 'oak' as normal
    'ath': 'OAK',     # ✅ map 'ath' to 'OAK'
    'phi': 'PHI',
    'pit': 'PIT',
    'sdp': 'SDP',
    'sfg': 'SFG',
    'sea': 'SEA',
    'stl': 'STL',
    'tbr': 'TBR',
    'tex': 'TEX',
    'tor': 'TOR',
    'wsn': 'WSN',
}

# Create a CloudScraper session
scraper = cloudscraper.create_scraper()

def fetch_game_logs(team, log_type, year, retries=5):
    url = f'https://www.baseball-reference.com/teams/tgl.cgi?team={team}&t={log_type}&year={year}'
    for attempt in range(retries):
        try:
            response = scraper.get(url)
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Error 429: Too Many Requests. Retrying in {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            div_id = {
                'b': 'div_players_standard_batting',
                'p': 'div_players_standard_pitching'
            }.get(log_type)

            if not div_id:
                print(f"Invalid log_type: {log_type}")
                return None

            div = soup.find('div', id=div_id)
            if not div:
                print(f"Div {div_id} not found for {team} {year}")
                continue

            table = div.find('table')
            if not table:
                print(f"No table found inside div {div_id} for {team} {year}")
                continue

            data = pd.read_html(StringIO(str(table)))[0]

            # Flatten multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() for col in data.columns]
            else:
                data.columns = [col.strip() for col in data.columns]

            # Rename columns based on source type
            rename_map = {}
            for col in data.columns:
                if "Batting Stats" in col or "Pitching Stats" in col:
                    rename_map[col] = col.split("_")[-1]
                elif col.startswith("Score_"):
                    continue  # handled later
                elif "Opp Starter_" in col:
                    if "Player" in col:
                        rename_map[col] = "Opp Starter (Player)"
                    elif "T" in col:
                        rename_map[col] = "Opp Starter (T)"
                    elif "GmSc" in col:
                        rename_map[col] = "Opp Starter (GmSc)"
                elif "Unnamed" in col and "Umpire" in col:
                    rename_map[col] = "Umpire"
                elif "Unnamed" in col and "Pitchers Used" in col:
                    rename_map[col] = "Pitchers Used (Rest-GameScore-Dec)"
                elif col.startswith("Unnamed:"):
                    continue
                else:
                    rename_map[col] = col

            data.rename(columns=rename_map, inplace=True)

            # Fuzzy remapping of core columns
            fuzzy_rename = {
                'Date': None,
                'Rk': None,
                'Gtm': None,
                'Opp': None,
                'Rslt': None
            }
            for col in data.columns:
                for target in fuzzy_rename:
                    if target in col and fuzzy_rename[target] is None:
                        fuzzy_rename[target] = col
            for clean_name, original_col in fuzzy_rename.items():
                if original_col:
                    data[clean_name] = data[original_col]
                    if original_col != clean_name:
                        data.drop(columns=[original_col], inplace=True)

            # Parse Date column for any HTML
            if 'Date' in data.columns:
                data['Date'] = data['Date'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text() if isinstance(x, str) else x)
            else:
                print(f"'Date' column still missing in data for {team} {year}. Columns are: {data.columns.tolist()}")
                return None

            # Normalize specific columns
            data.rename(columns={"GIDP": "GDP"}, inplace=True)

            if log_type == 'b':
                # Opp Starter name formatting
                if {'Opp Starter (Player)', 'Opp Starter (T)', 'Opp Starter (GmSc)'}.issubset(data.columns):
                    def format_starter(row):
                        name_parts = str(row['Opp Starter (Player)']).split()
                        if len(name_parts) >= 2:
                            first_initial = name_parts[0][0]
                            last_name = name_parts[-1]
                            formatted_name = f"{first_initial}.{last_name}"
                        else:
                            formatted_name = row['Opp Starter (Player)']
                        return f"{formatted_name}({row['Opp Starter (GmSc)']})"

                    data['Opp. Starter (GmeSc)'] = data.apply(format_starter, axis=1)
                    data['Thr'] = data['Opp Starter (T)']
                    data.drop(columns=['Opp Starter (Player)', 'Opp Starter (T)', 'Opp Starter (GmSc)'], inplace=True)

                expected_columns = {
                    'Rk', 'Gtm', 'Date', 'Unnamed: 3', 'Opp', 'Rslt', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR',
                    'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SH', 'SF', 'ROE', 'GDP', 'SB', 'CS', 'BA', 'OBP',
                    'SLG', 'OPS', 'LOB', '#', 'Thr', 'Opp. Starter (GmeSc)', 'Year'
                }
            else:  # log_type == 'p'
                expected_columns = {
                    'Rk', 'Gtm', 'Date', 'Unnamed: 3', 'Opp', 'Rslt', 'IP', 'H', 'R', 'ER', 'UER', 'BB', 'SO',
                    'HR', 'HBP', 'ERA', 'BF', 'Pit', 'Str', 'IR', 'IS', 'SB', 'CS', 'AB', '2B', '3B', 'IBB', 'SH',
                    'SF', 'ROE', 'GDP', '#', 'Umpire', 'Pitchers Used (Rest-GameScore-Dec)', 'Year'
                }
                # If UER is missing from modern BR pages, skip it gracefully
                expected_columns = {col for col in expected_columns if col in data.columns}

            # Final filtering
            data = data[[col for col in data.columns if col in expected_columns]]
            data['Year'] = year
            return data

        except Exception as e:
            print(f"Error fetching data for {team} ({year}): {e}")
            if attempt < retries - 1:
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Retrying in {retry_after} seconds...")
                time.sleep(retry_after)
    return None


def update_csv(team, directory, log_type, new_data):
    file_path = os.path.join(directory, f'{team}_{log_type}.csv')

    key_cols = ['Date', 'Opp', 'Year', 'Gtm']

    expected_columns_b = {
        'Rk', 'Gtm', 'Date', 'Unnamed: 3', 'Opp', 'Rslt', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR',
        'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SH', 'SF', 'ROE', 'GDP', 'SB', 'CS', 'BA', 'OBP',
        'SLG', 'OPS', 'LOB', '#', 'Thr', 'Opp. Starter (GmeSc)', 'Year'
    }

    expected_columns_p = {
        'Rk', 'Gtm', 'Date', 'Unnamed: 3', 'Opp', 'Rslt', 'IP', 'H', 'R', 'ER', 'UER', 'BB', 'SO',
        'HR', 'HBP', 'ERA', 'BF', 'Pit', 'Str', 'IR', 'IS', 'SB', 'CS', 'AB', '2B', '3B', 'IBB', 'SH',
        'SF', 'ROE', 'GDP', '#', 'Umpire', 'Pitchers Used (Rest-GameScore-Dec)', 'Year'
    }

    expected_columns = expected_columns_b if log_type == 'b' else expected_columns_p
    import re
    def parse_date(df):
        if 'Date' in df.columns:
            def convert(val):
                val = str(val).strip()
                # Match 'YYYY-MM-DD' or 'YYYY-MM-DD (1/2)'
                match = re.match(r'^(\d{4}-\d{2}-\d{2})(?: \((\d)\))?$', val)
                if match:
                    base_date = pd.to_datetime(match.group(1)).strftime('%b %d')
                    if match.group(2):
                        return f"{base_date} ({match.group(2)})"
                    else:
                        return base_date
                return val  # Leave existing formats untouched

            df['Date'] = df['Date'].apply(convert)
        return df

    def filter_columns(df):
        return df[[col for col in df.columns if col in expected_columns]]

    def normalize_keys(df):
        # Only strip (do not lowercase) to preserve original formatting
        for col in key_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        return df

    # Clean new data
    new_data = parse_date(new_data)
    new_data = normalize_keys(new_data)
    new_data = filter_columns(new_data)

    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        existing_data = parse_date(existing_data)
        existing_data = normalize_keys(existing_data)
        existing_data = filter_columns(existing_data)

        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        combined_data = combined_data.drop_duplicates(subset=key_cols).reset_index(drop=True)
    else:
        combined_data = new_data.drop_duplicates(subset=key_cols).reset_index(drop=True)

    combined_data.to_csv(file_path, index=False)
    print(f"CSV for {team} ({log_type}) updated. Shape: {combined_data.shape}")

def scrape_and_update(directory):
    for team, team_abbr in teams.items():
        for year in [2025]:  # Only fetch 2024 data
            for log_type in ['b', 'p']:
                print(f"Fetching data for {team.upper()} {log_type} logs ({year})...")
                new_data = fetch_game_logs(team.upper(), log_type, year)  # use actual URL abbreviation (e.g., ATH)
                if new_data is not None:
                    update_csv(team_abbr, directory, log_type, new_data)
                else:
                    print(f"No data fetched for {team.upper()} {log_type} logs ({year}).")
                # Add random delay between 30 and 60 seconds
                delay = random.uniform(15, 60)
                print(f"Waiting for {delay:.2f} seconds before next request...")
                time.sleep(delay)

# Define the data directory
data_directory = "C:/Users/DMelv/Documents/bettingModelBaseball/baseball_logs"

# Start the scraping and updating process
scrape_and_update(data_directory)








