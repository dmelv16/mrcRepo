from math import comb
import select
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
import numpy as np
import requests
from bs4 import BeautifulSoup
from twilio.rest import Client
import datetime
import cloudscraper


def load_and_merge_team_data(team_name, directory):
    pitching_path = os.path.join(directory, f'{team_name}_p.csv')
    batting_path = os.path.join(directory, f'{team_name}_b.csv')
    if not os.path.exists(pitching_path) or not os.path.exists(batting_path):
        raise FileNotFoundError(f"Data files for {team_name} not found in directory {directory}")
    pitching_df = pd.read_csv(pitching_path)
    batting_df = pd.read_csv(batting_path)
    merged_df = pd.merge(left=pitching_df, right=batting_df, how='inner', on=['Date', 'Opp', 'Rslt', 'Year'], suffixes=['_p', '_b'])
    merged_df['Team'] = team_name  # Add the 'Team' column
    return merged_df

def get_current_streak(df):
    last_win_streak = df['win_streak'].iloc[-1]
    last_lose_streak = df['lose_streak'].iloc[-1]
    last_result = df['W/L'].iloc[-1]

    if last_result == 1:
        return last_win_streak + 1, 0
    else:
        return 0, last_lose_streak + 1

def process_team_data_initial(df):
    df['W/L'] = df['Rslt'].apply(lambda x: 1 if 'W' in x else 0)
    df['WHIP'] = (df['H_p'] + df['BB_p']) / df['IP']
    df['Opp_WHIP'] = (df['H_b'] + df['BB_b']) / df['IP']
    df['Opp_S_GameScr'] = df['Opp. Starter (GmeSc)'].str.extract(r'\((\d+)\)').astype(float)
    
    

    win_streak = [0]
    lose_streak = [0]
    current_win_streak = 0
    current_lose_streak = 0

    for result in df['W/L']:
        if result == 1:
            current_win_streak += 1
            win_streak.append(current_win_streak)
            current_lose_streak = 0
            lose_streak.append(0)
        else:
            current_lose_streak += 1
            lose_streak.append(current_lose_streak)
            current_win_streak = 0
            win_streak.append(0)

    df['win_streak'] = win_streak[:len(df)]
    df['lose_streak'] = lose_streak[:len(df)]
    return df

def calculate_xFIP(df, constant=3.10):
    df.loc[:, 'xFIP'] = (13 * df['HR_p'] + 3 * (df['BB_p'] + df['HBP_p']) - 2 * df['SO_p']) / df['IP'] + constant
    return df['xFIP']

def preprocess_data(df):
    df = process_team_data_initial(df)
    df['xFIP'] = calculate_xFIP(df)
    return df

def get_pitcher_xFIP(pitcher_name, df, constant=3.10):
    relevant_rows = df[df['Pitchers Used (Rest-GameScore-Dec)'].str.contains(pitcher_name)].tail(5)
    if relevant_rows.empty:
        return None

    HR = relevant_rows['HR_p'].sum()
    BB = relevant_rows['BB_p'].sum()
    HBP = relevant_rows['HBP_p'].sum()
    K = relevant_rows['SO_p'].sum()
    IP = relevant_rows['IP'].sum()

    if IP == 0:
        return None

    xFIP = (13 * HR + 3 * (BB + HBP) - 2 * K) / IP + constant
    return xFIP

def get_pitcher_stats(pitcher_name, df, team_data=False):
    if pitcher_name == "Not Listed" or team_data:
        relevant_rows = df.tail(15)
    else:
        relevant_rows = df[df['Pitchers Used (Rest-GameScore-Dec)'].str.contains(pitcher_name)].tail(5)
        
    if relevant_rows.empty:
        return None, None, None, None, None

    game_scores = relevant_rows['Pitchers Used (Rest-GameScore-Dec)'].apply(lambda x: extract_pitcher_game_score(x, pitcher_name))
    game_scores = game_scores.dropna()

    pitcher_whip = relevant_rows['WHIP'].mean()
    ER = relevant_rows['ER'].mean()
    H = relevant_rows['H_p'].mean()
    SO = relevant_rows['SO_p'].mean()

    return pitcher_whip, game_scores.mean(), ER, H, SO

def extract_pitcher_game_score(row, pitcher_name):
    for entry in row.split(', '):
        if pitcher_name in entry:
            parts = entry.split('(')
            if len(parts) > 1:
                details = parts[1].split(')')
                if len(details) > 0:
                    game_score_part = details[0]
                    if game_score_part.isdigit():
                        return int(game_score_part)
    return None

def combine_date_year(date, year):
    if isinstance(date, (int, float)):
        date = str(date)
    if isinstance(year, (int, float)):
        year = str(int(year))
    return pd.to_datetime(f"{year}-{date}", format="%Y-%b %d", errors='coerce')

def extract_pitcher_name(opp_starter):
    if pd.isna(opp_starter):
        return None
    return opp_starter.split('(')[0].strip()

def prepare_features(df, pitcher_name, opposing_team_data, opposing_pitcher_name):

    #print(f"Data being used for feature preparation: \n{df}"
    # Shift the DataFrame by 1 to exclude the current game
    shifted_df = df[['WHIP', 'SLG', 'BA', 'OPS', 'xFIP', 'Opp_WHIP', 'Opp_S_GameScr']].shift(1)
    # Calculate the mean over the last 5 rows (previous 5 games)
    features = shifted_df.rolling(window=5).mean().iloc[-1].to_dict()
    current_win_streak, current_lose_streak = get_current_streak(df)
    features['win_streak'] = current_win_streak  # Use the updated win streak
    features['lose_streak'] = current_lose_streak  # Use the updated lose streak
    features['xFIP'] = get_pitcher_xFIP(pitcher_name, df) if pitcher_name != "Not Listed" else calculate_xFIP(df).mean()

    opp_pitcher_whip, opp_pitcher_game_score, _, _, _ = get_pitcher_stats(opposing_pitcher_name, opposing_team_data, team_data=(opposing_pitcher_name == "Not Listed"))

    opp_team_whip = opposing_team_data.tail(15)['WHIP'].mean()
    features['Opp_WHIP'] = (opp_pitcher_whip + opp_team_whip) / 2 if opp_pitcher_whip is not None else opp_team_whip

    features['Opp_S_GameScr'] = opp_pitcher_game_score

    pitcher_whip, pitcher_game_score, pitcher_ER, pitcher_H, pitcher_SO = get_pitcher_stats(pitcher_name, df, team_data=(pitcher_name == "Not Listed"))
    team_whip = df.tail(15)['WHIP'].mean()
    features['WHIP'] = (pitcher_whip + team_whip) / 2 if pitcher_whip is not None else team_whip
    features['pitcher_ER'] = pitcher_ER
    features['pitcher_H'] = pitcher_H
    features['pitcher_SO'] = pitcher_SO

    return features

def list_pitchers(df):
    pitchers = set()
    for pitchers_used in df['Pitchers Used (Rest-GameScore-Dec)']:
        for pitcher in pitchers_used.split(', '):
            pitcher_name = pitcher.split('(')[0].strip()
            pitchers.add(pitcher_name)
    pitchers.add("Not Listed")  # Add "Not Listed" option
    return sorted(list(pitchers), key=lambda x: x.split('.')[-1])

def select_pitcher(pitchers, pitcher_name):
    """
    Automatically select the pitcher if available, otherwise default to "Not Listed".
    """
    if pitcher_name in pitchers:
        return pitcher_name
    return "Not Listed"

def process_team_data(df, combined_df, er_model, runs_model):
    team_name = df['Team'].iloc[0]
    print(f"Processing data for team: {team_name}")
    
    combined_df['Team'] = combined_df['Team'].str.lower()

    combined_df['Combined_Date'] = combined_df.apply(lambda row: combine_date_year(row['Date'], row['Year']), axis=1)
    df['Combined_Date'] = df.apply(lambda row: combine_date_year(row['Date'], row['Year']), axis=1)
    
    if 'Projected_ER' not in df.columns:
        df['Projected_ER'] = np.nan
    if 'Projected_Runs' not in df.columns:
        df['Projected_Runs'] = np.nan

    df = df.sort_values(by='Combined_Date')  # Ensure data is sorted by date
    df = df.dropna(subset = ['Combined_Date'])

    for i in range(len(df)):
        if i < 10:
            team_last_games = None
        else:
            team_last_games = df.iloc[max(0, i-10):i]

        opp_starter = df.iloc[i]['Opp. Starter (GmeSc)']
        current_pitcher = extract_pitcher_name(opp_starter)
        
        if not current_pitcher:
            df.loc[i, 'Projected_ER'] = np.nan
            df.loc[i, 'Projected_Runs'] = np.nan
            continue

        opp_team = df.iloc[i]['Opp'].lower()
        filtered_rows = combined_df[
            (combined_df['Team'] == opp_team) & 
            (combined_df['Pitchers Used (Rest-GameScore-Dec)'].str.contains(current_pitcher)) &
            (combined_df['Combined_Date'] < df.iloc[i]['Combined_Date'])
        ].tail(5)

        if filtered_rows.empty:
            df.loc[i, 'Projected_ER'] = np.nan
            df.loc[i, 'Projected_Runs'] = np.nan
            continue
        
        # Ensure all feature columns are numeric and handle missing values
        numeric_columns = ['WHIP', 'SLG', 'BA', 'OPS', 'win_streak', 'lose_streak', 'xFIP', 'Opp_WHIP', 'Opp_S_GameScr']
        feature_vector = df.iloc[max(0, i-10):i][numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(df[numeric_columns].mean())
        
        # Handle the case when the pitcher is "Not Listed"
        if current_pitcher == "Not Listed" or feature_vector.isnull().values.any():
            feature_vector = df.iloc[max(0, i-15):i][numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(df[numeric_columns].mean())

        if feature_vector.isnull().values.any():
            df.loc[i, 'Projected_ER'] = np.nan
            df.loc[i, 'Projected_Runs'] = np.nan
            continue
        
        feature_vector = feature_vector.mean().values.reshape(1, -1)
        feature_vector_df = pd.DataFrame(feature_vector, columns=numeric_columns)
        
        projected_er = er_model.predict(feature_vector_df)[0]
        projected_runs = runs_model.predict(feature_vector_df)[0]

        df.loc[i, 'Projected_ER'] = projected_er
        df.loc[i, 'Projected_Runs'] = projected_runs
    
    return df

def get_team_names(directory):
    files = os.listdir(directory)
    team_names = set()
    for file in files:
        if file.endswith('_p.csv') or file.endswith('_b.csv'):
            team_name = file.split('_')[0].lower()
            team_names.add(team_name)
    return sorted(list(team_names))

def preprocess_and_merge_team_data(team_name, directory):
    team_data = preprocess_data(load_and_merge_team_data(team_name, directory))
    return team_data

def train_and_validate_model(directory):
    team_names = get_team_names(directory)
    
    all_teams_data = Parallel(n_jobs=-1)(delayed(preprocess_and_merge_team_data)(team_name, directory) for team_name in team_names)

    combined_df = pd.concat(all_teams_data, ignore_index=True)

    er_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    runs_model = RandomForestRegressor(random_state=42, n_jobs=-1)

    feature_cols = ['WHIP', 'SLG', 'BA', 'OPS', 'win_streak', 'lose_streak', 'xFIP', 'Opp_WHIP', 'Opp_S_GameScr']

    X_features = []
    y_er = []
    y_runs = []

    for team_data in all_teams_data:
        team_name = team_data['Team'].iloc[0]
        for i in range(len(team_data)):
            if i < 10:
                continue  # Skip the first few rows for each team as there isn't enough data

            team_last_games = team_data.iloc[max(0, i-10):i]
            opp_team = team_data.iloc[i]['Opp'].lower()
            opp_starter = team_data.iloc[i]['Opp. Starter (GmeSc)']
            current_pitcher = extract_pitcher_name(opp_starter)

            opposing_team_data = combined_df[combined_df['Team'] == opp_team]

            features = prepare_features(team_last_games, current_pitcher, opposing_team_data, opp_starter)
            X_features.append([features[col] for col in feature_cols])
            y_er.append(team_data.iloc[i]['ER'])
            y_runs.append(team_data.iloc[i]['R_b'])

    X = pd.DataFrame(X_features, columns=feature_cols)
    y_er = np.array(y_er)
    y_runs = np.array(y_runs)

    X_train_er, X_test_er, y_train_er, y_test_er = train_test_split(X, y_er, test_size=0.2, random_state=42)
    X_train_runs, X_test_runs, y_train_runs, y_test_runs = train_test_split(X, y_runs, test_size=0.2, random_state=42)

    er_model.fit(X_train_er, y_train_er)
    runs_model.fit(X_train_runs, y_train_runs)

    all_teams_data = Parallel(n_jobs=-1)(delayed(process_team_data)(team_data, combined_df, er_model, runs_model) for team_data in all_teams_data)
    
    combined_df_with_projections = pd.concat(all_teams_data, ignore_index=True)
    
    return er_model, runs_model, combined_df_with_projections

def format_name(name):
    parts = name.split()
    if len(parts) == 1:
        return name  # Single-part name stays as is
    elif len(parts) == 2:
        return f"{parts[0][0]}.{parts[1]}"  # Standard two-part name
    elif len(parts) > 2:
        return f"{parts[0][0]}.{parts[-2]} {parts[-1]}"  # Keep the last two parts for names with more than two parts

# Dictionary to map full team names to their abbreviations
TEAM_ABBREVIATIONS = {
    "ROYALS": "KCR",
    "REDS": "CIN",
    "YANKEES": "NYY",
    "TIGERS": "DET",
    "NATIONALS": "WSN",
    "PHILLIES": "PHI",
    "MARINERS": "SEA",
    "PIRATES": "PIT",
    "D'BACKS": "ARI",
    "RAYS": "TBR",
    "RED SOX": "BOS",  # Assuming "RED" refers to the Red Sox
    "ORIOLES": "BAL",
    "MARLINS": "MIA",
    "METS": "NYM",
    "TWINS": "MIN",
    "RANGERS": "TEX",
    "WHITE SOX": "CHW",  # Assuming "WHITE" refers to the White Sox
    "ASTROS": "HOU",
    "GUARDIANS": "CLE",
    "BREWERS": "MIL",
    "DODGERS": "LAD",
    "CARDINALS": "STL",
    "PADRES": "SDP",
    "ROCKIES": "COL",
    "BRAVES": "ATL",
    "ANGELS": "LAA",
    "CUBS": "CHC",
    "BLUE JAYS": "TOR",  # Assuming "BLUE" refers to the Blue Jays
    "GIANTS": "SFG",
    "ATHLETICS": "OAK"
}

def scrape_pitchers():
    url = "https://www.baseball-reference.com/previews/"
    scraper = cloudscraper.create_scraper()
    response = scraper.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    matchups = []

    # Find all game summaries
    game_summaries = soup.find_all("div", class_="game_summary nohover")
    print(f"Found {len(game_summaries)} game summaries.")  # Debugging: print number of game summaries found

    for game in game_summaries:
        # Extract the team names using the first table row
        team_rows = game.find_all("tr")
        if len(team_rows) < 2:
            continue  # Skip if we don't have two teams listed

        # Extract and map team names to abbreviations
        team1_full = team_rows[0].find("a").text.strip().upper()
        team2_full = team_rows[1].find("a").text.strip().upper()

        team1 = TEAM_ABBREVIATIONS.get(team1_full, team1_full)
        team2 = TEAM_ABBREVIATIONS.get(team2_full, team2_full)

        # Extract pitcher information from the second table within each game_summary
        pitcher_info = game.find_all("table")[1].find_all("a") if len(game.find_all("table")) > 1 else []

        if len(pitcher_info) >= 2:
            pitcher1 = format_name(pitcher_info[0].text.strip())
            pitcher2 = format_name(pitcher_info[1].text.strip())
        elif len(pitcher_info) == 1:
            pitcher1 = format_name(pitcher_info[0].text.strip())
            pitcher2 = "Not Listed"
        else:
            pitcher1 = "Not Listed"
            pitcher2 = "Not Listed"



        # Append the matchup
        matchups.append((team1, pitcher1, team2, pitcher2))

    print("Matchups scraped:", matchups)  # Debugging: print the matchups found
    return matchups


def predict_projected_scores(team1_name, team2_name, pitcher1_name, pitcher2_name, combined_df_with_projections, runs_model):
    team1_data = combined_df_with_projections[combined_df_with_projections['Team'] == team1_name]
    team2_data = combined_df_with_projections[combined_df_with_projections['Team'] == team2_name]

    pitchers_team1 = list_pitchers(team1_data)
    pitchers_team2 = list_pitchers(team2_data)

    if pitcher1_name not in pitchers_team1:
        pitcher1_name = "Not Listed"
    if pitcher2_name not in pitchers_team2:
        pitcher2_name = "Not Listed"

    feature_cols = ['WHIP', 'SLG', 'BA', 'OPS', 'win_streak', 'lose_streak', 'xFIP', 'Opp_WHIP', 'Opp_S_GameScr']

    team1_features = prepare_features(team1_data, pitcher1_name, team2_data, pitcher2_name)
    team1_feature_values = [team1_features[col] for col in feature_cols]

    team2_features = prepare_features(team2_data, pitcher2_name, team1_data, pitcher1_name)
    team2_feature_values = [team2_features[col] for col in feature_cols]

    team1_projected_runs = runs_model.predict([team1_feature_values])[0]
    team2_projected_runs = runs_model.predict([team2_feature_values])[0]

    return team1_projected_runs, team2_projected_runs

def evaluate_model_accuracy(combined_df_with_projections):
    correct_predictions = 0
    total_predictions = 0

    for index, row in combined_df_with_projections.iterrows():
        if pd.notna(row['Projected_Runs']) and pd.notna(row['Projected_ER']):
            total_predictions += 1
            actual_winner = row['W/L']
            projected_winner = 1 if row['Projected_Runs'] > row['Projected_ER'] else 0
            if actual_winner == projected_winner:
                correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

def evalutate_matchup_accuracy(team1_name, team2_name, combined_df_with_projections, threshold=8.5):
    selected_games = combined_df_with_projections[
        (combined_df_with_projections['Team'] == team1_name) |
        (combined_df_with_projections['Team'] == team2_name)
    ]
    correct_predictions = 0
    total_predictions = 0

    for index, row in selected_games.iterrows():
        if pd.notna(row['Projected_Runs']) and pd.notna(row['Projected_ER']):
            total_predictions += 1
            actual_winner = row['W/L']
            projected_winner = 1 if row['Projected_Runs'] > row['Projected_ER'] else 0
            if actual_winner == projected_winner:
                correct_predictions += 1

    accuracy_MATCH = correct_predictions / total_predictions if total_predictions > 0 else 0

    correct_predictions = 0
    total_predictions = 0

    for index, row in selected_games.iterrows():
        if pd.notna(row['Projected_Runs']) and pd.notna(row['Projected_ER']):
            total_predictions += 1
            projected_total_runs = row['Projected_Runs'] + row['Projected_ER']
            actual_total_runs = row['R_b'] + row['ER']
            projected_over = projected_total_runs > threshold
            actual_over = actual_total_runs > threshold

            if projected_over == actual_over:
                correct_predictions += 1

    accuracy_OU = correct_predictions / total_predictions if total_predictions > 0 else 0


    return accuracy_MATCH, accuracy_OU

def upload_file(predictions):
    today = datetime.datetime.now().strftime("%m%d")
    message_body = "\n".join(predictions)
    print(message_body)
    with open(f'C:\\Users\\DMelv\\Documents\\code\\baseballPredictions\\scorepredictions\\scorepredictor_{today}.txt', 'w') as file:
        file.write(message_body)

def main():
    directory = r"C:\Users\DMelv\Documents\bettingModelBaseball\baseball_logs"
    er_model, runs_model, combined_df_with_projections = train_and_validate_model(directory)
    
    matchups = scrape_pitchers()
    predictions = []

    for team1, pitcher1, team2, pitcher2 in matchups:
        team1_projected_runs, team2_projected_runs = predict_projected_scores(
            team1.lower(), team2.lower(), pitcher1, pitcher2, combined_df_with_projections, runs_model
        )
        predictions.append(f"{team1} {team1_projected_runs:.2f} v {team2_projected_runs:.2f} {team2}")
    
    # # Twilio credentials and configuration
    # account_sid = 'AC0bc41cb0a132bbf2b8fec246d8cedd27'
    # auth_token = '63a4deeed7a03be3138e33c5b22d6ea2'
    # twilio_number = '+18449523391'  # Your Twilio phone number
    # to_number = '+13157264010'  # Your phone number in E.164 format (+1 for US)

    upload_file(predictions)


if __name__ == "__main__":
    main()