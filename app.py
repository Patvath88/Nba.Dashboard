# This code can be saved as a Python file (e.g., nba_dashboard.py) and run with streamlit run nba_dashboard.py

import streamlit as st
from nba_api.stats.static import players, teams # Import teams module
from nba_api.stats.endpoints import playercareerstats, playergamelogs, teamgamelogs
import pandas as pd
import re # Import regex for parsing matchup string
import numpy as np # Import numpy for handling NaN values and calculations
import time # Import time to add delays for API calls

# Define core stats columns globally so they are accessible by all functions and the main app
stats_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']


# Define the get_player_stats function (MODIFIED to fetch all career data)
@st.cache_data # Cache this function to avoid re-fetching data for the same player
def get_player_stats(player_id):
    """
    Fetches comprehensive player career data (summary and game logs) and calculates
    various averages.

    Args:
        player_id (int): The ID of the NBA player.

    Returns:
        dict: A dictionary containing career_df, career game logs DataFrame,
              historical career averages, overall career averages, last season averages,
              and recent game averages, or None if data fetching fails.
    """
    career_df = None
    career_game_logs = None
    historical_career_averages = None
    overall_career_averages = None
    last_season_averages = None
    last_5_games_avg = None
    last_10_games_avg = None
    last_20_games_avg = None
    last_5_games_individual = None


    try:
        # Fetch career summary stats
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        career_df = career_stats.get_data_frames()[0]

        if career_df.empty:
            st.info("No career data found for this player.")
            return None

        # Fetch career game logs across all available seasons
        all_game_logs_list = []
        seasons_played = career_df['SEASON_ID'].tolist()

        # Fetch game logs for each season played by the player
        for season in seasons_played:
            try:
                # st.write(f"Fetching game logs for {season}...") # Optional: show progress
                season_logs = playergamelogs.PlayerGameLogs(player_id_nullable=player_id, season_nullable=season).get_data_frames()[0]
                all_game_logs_list.append(season_logs)
                # Add a small delay between API calls to be polite and avoid rate limits
                if len(seasons_played) > 1: # Only add delay if fetching multiple seasons
                    time.sleep(0.2) # Reduced delay slightly


            except Exception as e:
                st.warning(f"Could not fetch game logs for season {season}. Skipping this season. Error: {e}")
                # Continue to the next season if one season fails

        if all_game_logs_list:
            career_game_logs = pd.concat(all_game_logs_list, ignore_index=True)
            career_game_logs['GAME_DATE'] = pd.to_datetime(career_game_logs['GAME_DATE'])
            career_game_logs = career_game_logs.sort_values(by='GAME_DATE').reset_index(drop=True) # Sort by date ascending


        # Calculate averages from the comprehensive career_df and career_game_logs

        # Calculate overall career averages
        overall_total_games = career_df['GP'].sum()
        if overall_total_games > 0:
            overall_career_totals = career_df[stats_columns].sum()
            valid_stats_columns_overall = [col for col in stats_columns if col in overall_career_totals.index]
            overall_career_averages = overall_career_totals[valid_stats_columns_overall] / overall_total_games
            overall_career_averages = overall_career_averages.to_frame(name='Overall Career Avg').T # Convert to DataFrame for display


        # Calculate historical career averages (excluding the most recent season in career_df)
        if len(career_df) > 1:
            historical_career_df = career_df.iloc[:-1]
            historical_total_games = historical_career_df['GP'].sum()
            if historical_total_games > 0:
                valid_stats_columns_hist = [col for col in stats_columns if col in historical_career_df.columns]
                historical_career_totals = historical_career_df[valid_stats_columns_hist].sum()
                historical_career_averages = historical_career_totals / historical_total_games
                historical_career_averages = historical_career_averages.to_frame(name='Historical Career Avg').T # Convert to DataFrame for display


        # Get last season's averages (the second to last season in career_df, if available)
        if len(career_df) >= 2:
            last_season_df = career_df.iloc[-2] # Second to last row is the previous season
            last_season_id = last_season_df['SEASON_ID']
            last_season_games = last_season_df['GP']
            if last_season_games > 0:
                 valid_stats_columns_last = [col for col in stats_columns if col in last_season_df.index]
                 last_season_totals = last_season_df[valid_stats_columns_last]
                 last_season_averages = last_season_totals / last_season_games
                 last_season_averages = last_season_averages.to_frame(name=f"Last Season ({last_season_id}) Avg").T # Convert to DataFrame for display
            else:
                 st.info(f"Player played 0 games in their second to last season ({last_season_id}). Last season averages not available.")


        # Calculate recent game averages from the comprehensive career game logs
        if career_game_logs is not None and not career_game_logs.empty:
             # Sort game logs by date descending for recent games (already sorted ascending, reverse it)
             recent_game_logs = career_game_logs.sort_values(by='GAME_DATE', ascending=False).reset_index(drop=True)

             valid_game_stats_columns = [col for col in stats_columns if col in recent_game_logs.columns]

             if len(recent_game_logs) >= 5:
                 last_5_games_avg = recent_game_logs.head(5)[valid_game_stats_columns].mean()
                 last_5_games_avg = last_5_games_avg.to_frame(name='Last 5 Games Avg').T # Convert to DataFrame
                 last_5_games_individual = recent_game_logs.head(5)[['GAME_DATE', 'MATCHUP', 'SEASON_YEAR'] + valid_game_stats_columns] # Get individual game stats, include season year
             if len(recent_game_logs) >= 10:
                 last_10_games_avg = recent_game_logs.head(10)[valid_game_stats_columns].mean()
                 last_10_games_avg = last_10_games_avg.to_frame(name='Last 10 Games Avg').T # Convert to DataFrame
             if len(recent_game_logs) >= 20:
                 last_20_games_avg = recent_game_logs.head(20)[valid_game_stats_columns].mean()
                 last_20_games_avg = last_20_games_avg.to_frame(name='Last 20 Games Avg').T # Convert to DataFrame
        else:
            st.warning("No career game logs available to calculate recent game averages.")


        return {
            'career_df': career_df, # Comprehensive career summary
            'career_game_logs': career_game_logs, # Comprehensive career game logs
            'historical_career_averages': historical_career_averages,
            'overall_career_averages': overall_career_averages,
            'last_season_averages': last_season_averages,
            'last_5_games_avg': last_5_games_avg,
            'last_10_games_avg': last_10_games_avg,
            'last_20_games_avg': last_20_games_avg,
            'last_5_games_individual': last_5_games_individual # Return individual last 5 games
        }

    except Exception as e:
        st.error(f"An error occurred while fetching comprehensive player data: {e}")
        return None


# Define a function to calculate player vs team career stats (reusing and slightly modifying)
@st.cache_data # Cache this data
def get_player_vs_team_career_stats(player_id, team_id, career_game_logs):
    """
    Calculates a player's career statistics against a specific team from career game logs.

    Args:
        player_id (int): The ID of the player.
        team_id (int): The ID of the opponent team.
        career_game_logs (DataFrame): DataFrame containing the player's career game logs.

    Returns:
        DataFrame: A pandas DataFrame containing the player's career averages against
                   the specified team, or None if data is insufficient or they
                   didn't play against that team in their career.
    """
    if career_game_logs is None or career_game_logs.empty:
        # st.info("No career game logs available to calculate player vs team stats.")
        return None

    # stats_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] # Moved to global scope

    try:
        # Get the abbreviation of the opponent team ID
        opponent_team_abbr = None
        team_info = teams.get_teams() # Use teams.get_teams()
        for team in team_info:
             if team['id'] == team_id:
                 opponent_team_abbr = team['abbreviation']
                 break

        if opponent_team_abbr is None:
             st.error(f"Could not find abbreviation for team ID: {team_id}")
             return None

        # Filter player logs for games against the specific opponent team across career
        def played_against_team(row, opponent_abbr_to_find):
            # Check if the opponent abbreviation appears after '@' or 'vs.' in the matchup string
            # Also ensure the opponent team abbreviation matches the selected opponent team's abbreviation
            matchup = row.get('MATCHUP', '')
            player_team_abbr = row.get('TEAM_ABBREVIATION', '') # Player's team abbreviation in this game

            match = re.search(rf'@\s*([A-Z]{3})|vs.\s*([A-Z]{3})|VS.\s*{opponent_abbr_to_find}', matchup, re.IGNORECASE) # Added IGNORECASE for robustness
            if match:
                extracted_abbr = match.group(1) or match.group(2) or match.group(3) # Capture from any group
                if extracted_abbr and extracted_abbr.upper() == opponent_abbr_to_find.upper():
                     # Additional check: Ensure the player's team is the opponent of the extracted team in this game
                     # This is hard to verify accurately just from the matchup string without opponent team logs.
                     # For now, we trust the matchup string and opponent_abbr match.
                     return True # Return True if a match is found and it matches the selected opponent

            return False # Return False if no match or if it doesn't match the selected opponent

        player_logs_vs_team_career = career_game_logs[
            career_game_logs.apply(lambda row: played_against_team(row, opponent_team_abbr), axis=1)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning


        if player_logs_vs_team_career.empty:
            # st.info(f"Player {player_id} did not play against team {team_id} in their career.")
            return None

        # Calculate player's career averages in games against the specific team
        valid_stats_columns_vs_team = [col for col in stats_columns if col in player_logs_vs_team_career.columns]
        player_vs_team_career_avg = player_logs_vs_team_career[valid_stats_columns_vs_team].mean()
        player_vs_team_career_avg = player_vs_team_career_avg.to_frame(name=f"Career Avg vs {opponent_team_abbr}").T # Convert to DataFrame

        return player_vs_team_career_avg

    except Exception as e:
        st.error(f"An error occurred while calculating player vs team career stats: {e}")
        return None


# Define the fetch_and_combine_game_data function for prediction features (MODIFIED)
# This function will now take career_game_logs as input
def fetch_and_combine_game_data_for_prediction(player_id, career_game_logs):
    """
    Combines player career game logs with corresponding opponent team game logs
    for feature engineering for prediction.

    Args:
        player_id (int): The ID of the NBA player.
        career_game_logs (DataFrame): DataFrame containing the player's career game logs.

    Returns:
        DataFrame: A DataFrame containing combined player and opponent team stats
                   for each game the player played in their career, or None if
                   data is insufficient or team logs cannot be fetched.
    """
    if career_game_logs is None or career_game_logs.empty:
        st.warning("No career game logs available for prediction feature engineering.")
        return None

    combined_data = None

    try:
        # Fetch all team game logs for all seasons covered by player's career game logs
        seasons_in_career_logs = career_game_logs['SEASON_YEAR'].unique().tolist()
        all_team_logs_list = []
        for season in seasons_in_career_logs:
            try:
                # st.write(f"Fetching team logs for {season}...") # Optional: show progress
                season_team_logs = teamgamelogs.TeamGameLogs(season_nullable=season).get_data_frames()[0]
                all_team_logs_list.append(season_team_logs)
                # Add a small delay between API calls
                if len(seasons_in_career_logs) > 1:
                    time.sleep(0.2) # Reduced delay slightly

            except Exception as e:
                 st.warning(f"Could not fetch team game logs for season {season}. Skipping this season's team data. Error: {e}")


        all_team_logs_combined = pd.concat(all_team_logs_list, ignore_index=True) if all_team_logs_list else pd.DataFrame()


        if not all_team_logs_combined.empty:
            # Merge player logs with all team logs to get player's team's stats for each game
            player_team_combined = pd.merge(
                career_game_logs,
                all_team_logs_combined,
                on=['GAME_ID', 'GAME_DATE', 'TEAM_ID'],
                suffixes=('_player', '_team'),
                how='left' # Use left join to keep all player's games
                )


            # Extract opponent team abbreviation from the 'MATCHUP_player' column
            def extract_opponent_team_abbr(matchup, player_team_abbr):
                # Corrected regex to avoid referencing opponent_abbr in the pattern
                match = re.search(r'@\s*([A-Z]{3})|vs.\s*([A-Z]{3})|VS.\s*([A-Z]{3})', matchup, re.IGNORECASE) # Capture any 3 uppercase letters after @ or vs.
                if match:
                    extracted_abbr = match.group(1) or match.group(2) or match.group(3) # Capture from any group
                    # Basic check to ensure the extracted opponent is not the player's own team in that game
                    if extracted_abbr and player_team_abbr and extracted_abbr.upper() != player_team_abbr.upper():
                        return extracted_abbr.upper() # Return uppercase for consistent matching
                    elif extracted_abbr: # Handle cases where the team might be the opponent but the abbreviation matches player's team (e.g., both are LAL in different games)
                         pass # For now, rely on the basic check if team abbrs are the same
                return None # Return None if no match or if it doesn't match player's team abbr


            # Apply the function to extract opponent team abbreviation
            player_team_combined['OPPONENT_TEAM_ABBREVIATION'] = player_team_combined.apply(
                lambda row: extract_opponent_team_abbr(row.get('MATCHUP', ''), row.get('TEAM_ABBREVIATION', '')), axis=1
            )

            # Get a mapping of team abbreviation to team ID from the combined team logs
            team_abbr_to_id = all_team_logs_combined[['TEAM_ABBREVIATION', 'TEAM_ID']].drop_duplicates().set_index('TEAM_ABBREVIATION')['TEAM_ID'].to_dict()

            # Map opponent abbreviation to opponent team ID
            player_team_combined['OPPONENT_TEAM_ID'] = player_team_combined['OPPONENT_TEAM_ABBREVIATION'].map(team_abbr_to_id)

            # Merge with all_team_logs_combined again to get opponent stats, using GAME_ID and OPPONENT_TEAM_ID
            # Need to rename columns in all_team_logs_combined before merging to avoid conflicts
            # Exclude columns that might already be in player_team_combined from the opponent logs
            cols_to_rename_opponent = [col for col in all_team_logs_combined.columns if col not in player_team_combined.columns or col in ['GAME_ID', 'GAME_DATE', 'TEAM_ID']]
            opponent_cols_mapping = {col: f'{col}_opponent' for col in cols_to_rename_opponent}
            all_team_logs_renamed = all_team_logs_combined.rename(columns=opponent_cols_mapping)

            # Perform the merge for opponent stats
            combined_data = pd.merge(
                player_team_combined,
                all_team_logs_renamed,
                left_on=['GAME_ID', 'OPPONENT_TEAM_ID'],
                right_on=[opponent_cols_mapping.get('GAME_ID', 'GAME_ID_opponent'), opponent_cols_mapping.get('TEAM_ID', 'TEAM_ID_opponent')],
                how='left' # Use left join to keep all player's games
            )

            # Drop redundant columns after merge - adjusted based on potential renamed columns
            cols_to_drop_after_merge = [opponent_cols_mapping.get('GAME_ID', 'GAME_ID_opponent'), opponent_cols_mapping.get('TEAM_ID', 'TEAM_ID_opponent')]
            # Add other potential redundant columns if they weren't used in the merge key
            cols_to_drop_after_merge.extend([col for col in opponent_cols_mapping.values() if col not in combined_data.columns and col not in cols_to_drop_after_merge])
            # Also drop original columns that were only needed for merging or extracting opponent info if they have a _player or _team suffix
            cols_to_drop_after_merge.extend(['OPPONENT_TEAM_ABBREVIATION']) # This was a helper column

            combined_data = combined_data.drop(columns=cols_to_drop_after_merge, errors='ignore')

            # Ensure data is sorted by date for feature engineering
            combined_data['GAME_DATE'] = pd.to_datetime(combined_data['GAME_DATE'])
            combined_data = combined_data.sort_values(by='GAME_DATE').reset_index(drop=True)

        else:
             st.warning("Could not fetch any team logs. Combined data with opponent stats will not be available for prediction features.")
             combined_data = career_game_logs.copy() # Return player logs only if team logs failed


        return combined_data

    except Exception as e:
        st.error(f"An error occurred while fetching and combining data for prediction: {e}")
        return None


# Define the feature engineering function (MODIFIED)
# This function will take the combined data across seasons
def engineer_features_for_prediction(combined_data):
    """
    Engineers features and defines targets for prediction from combined career game data.
    Calculates rolling averages and rest days based on combined data across seasons.


    Args:
        combined_data (DataFrame): DataFrame containing combined player and opponent game data across career.

    Returns:
        tuple: A tuple containing the feature DataFrame (X), target DataFrame (y),
               and the cleaned DataFrame with features and targets.
               Returns (None, None, None) if data is insufficient.
    """
    if combined_data is None or combined_data.empty:
        st.warning("No data available for feature engineering for prediction.")
        return None, None, None

    # stats_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] # Moved to global scope
    prediction_targets = ['PTS', 'AST', 'REB', 'FG3M'] # Specific targets for prediction

    # Ensure data is sorted by date before calculating rolling averages and rest days
    combined_data['GAME_DATE'] = pd.to_datetime(combined_data['GAME_DATE'])
    combined_data = combined_data.sort_values(by='GAME_DATE').reset_index(drop=True)


    # Calculate rolling averages for player stats across combined seasons
    for col in stats_columns:
        # Use a helper function to handle potential NaN values at the start
        # and ensure rolling is done by player across the combined dataset
        # Use the player's stats columns with '_player' suffix if available, otherwise the original column
        player_col = f'{col}_player' if f'{col}_player' in combined_data.columns else col
        if player_col in combined_data.columns:
            combined_data[f'{col}_rolling_5'] = combined_data.groupby('PLAYER_ID')[player_col].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
            combined_data[f'{col}_rolling_10'] = combined_data.groupby('PLAYER_ID')[player_col].transform(lambda x: x.rolling(window=10, min_periods=1).mean().shift(1))
            combined_data[f'{col}_rolling_20'] = combined_data.groupby('PLAYER_ID')[player_col].transform(lambda x: x.rolling(window=20, min_periods=1).mean().shift(1))
        else:
             st.warning(f"Player stat column '{player_col}' not found for rolling average calculation.")


    # Create a binary feature indicating home or away game
    def is_home_game(matchup):
        # Check if the matchup string contains "vs." (typically indicates a home game)
        return 1 if 'vs.' in matchup else 0

    # Use the player's matchup column ('MATCHUP_player' or 'MATCHUP')
    matchup_col = 'MATCHUP_player' if 'MATCHUP_player' in combined_data.columns else 'MATCHUP'
    if matchup_col in combined_data.columns:
        combined_data['IS_HOME_player'] = combined_data[matchup_col].apply(is_home_game)
    else:
         st.warning("Matchup column not found for IS_HOME_player feature.")
         combined_data['IS_HOME_player'] = 0 # Default to 0 if matchup column is missing


    # Calculate rest days since the last game across combined seasons
    if 'GAME_DATE' in combined_data.columns:
        combined_data['PREV_GAME_DATE'] = combined_data.groupby('PLAYER_ID')['GAME_DATE'].shift(1)
        combined_data['REST_DAYS_player'] = (combined_data['GAME_DATE'] - combined_data['PREV_GAME_DATE']).dt.days.fillna(0) # Fill NaN for the first game
    else:
        st.warning("GAME_DATE column not found for REST_DAYS_player feature.")
        combined_data['REST_DAYS_player'] = 0 # Default to 0 if GAME_DATE is missing


    # Select feature columns - adjusted to account for combined data
    # Include rolling averages (check if columns were successfully created)
    rolling_cols = [col for col in combined_data.columns if '_rolling_' in col]
    feature_columns = rolling_cols

    # Include opponent statistics if available (check if any opponent columns exist)
    opponent_cols = [col for col in combined_data.columns if '_opponent' in col and col not in ['SEASON_YEAR_opponent']]
    if opponent_cols:
         feature_columns.extend(opponent_cols)
    else:
         st.warning("Opponent statistics are not available for feature engineering.")

    # Include home/away and rest days features (check if columns were successfully created)
    if 'IS_HOME_player' in combined_data.columns:
        feature_columns.append('IS_HOME_player')
    if 'REST_DAYS_player' in combined_data.columns:
        feature_columns.append('REST_DAYS_player')


    # Define target variables (the actual stats from the current game)
    # Use the player's stats columns with '_player' suffix if available, otherwise the original column
    target_columns = [f'{col}_player' if f'{col}_player' in combined_data.columns else col for col in prediction_targets]
    # Ensure target columns exist
    valid_target_columns = [col for col in target_columns if col in combined_data.columns]
    if len(valid_target_columns) != len(prediction_targets):
         missing_targets = [target for target in prediction_targets if f'{target}_player' not in combined_data.columns and target not in combined_data.columns]
         st.warning(f"Missing target columns for prediction: {missing_targets}. Cannot proceed with prediction.")
         return None, None, None
    target_columns = valid_target_columns


    # Create X and y, dropping rows with NaNs created by rolling averages or missing opponent data
    # We also need to shift the target variables up by one row to predict the *next* game's stats
    # Ensure all required columns exist before dropping NaNs
    required_cols = feature_columns + target_columns
    # Filter combined_data to only include required columns before dropping NaNs
    combined_data_subset = combined_data[required_cols].copy()

    combined_data_cleaned = combined_data_subset.dropna(subset=feature_columns).copy() # Only drop NaNs from features for now
    combined_data_cleaned[target_columns] = combined_data_cleaned[target_columns].shift(-1)

    # Drop the last row after shifting targets, as its target will be NaN
    combined_data_cleaned = combined_data_cleaned.iloc[:-1].copy()

    # Select the features (X) and targets (y) from the cleaned data
    # Re-select feature columns from the cleaned data in case some were dropped due to NaNs
    X = combined_data_cleaned[feature_columns]
    y = combined_data_cleaned[target_columns]


    if X.empty or y.empty:
        st.warning("Insufficient data after feature engineering and cleaning to generate predictions.")
        return None, None, None

    return X, y, combined_data_cleaned


# Define a simple prediction function (MODIFIED)
# This function will now take the combined_data with engineered features
def predict_next_game_stats(player_name, combined_data_for_prediction_features):
    """
    Generates a simplified prediction for the next game's stats using engineered features.
    This uses a weighted average of recent game averages and season average derived from
    the engineered features.

    Args:
        player_name (str): The name of the player.
        combined_data_for_prediction_features (DataFrame): DataFrame with engineered features (but not shifted targets).

    Returns:
        dict: A dictionary containing predicted stats, or None if prediction is not possible.
    """
    if combined_data_for_prediction_features is None or combined_data_for_prediction_features.empty:
        st.info("No combined data with engineered features available to generate prediction.")
        return None

    # Get the most recent game's engineered features to predict the next game
    # The 'combined_data_for_prediction_features' DataFrame should be sorted by date
    last_game_engineered_features = combined_data_for_prediction_features.iloc[-1]

    # Use a simplified prediction based on weighted averages
    stats_for_prediction = ['PTS', 'AST', 'REB', 'FG3M']
    predicted_stats = {}

    # Calculate season average from the entire combined data for the player
    # Use the player's stats columns with '_player' suffix if available, otherwise the original column
    player_stats_cols = [f'{col}_player' if f'{col}_player' in combined_data_for_prediction_features.columns else col for col in stats_columns]
    # Ensure columns exist before calculating mean
    valid_player_stats_cols = [col for col in player_stats_cols if col in combined_data_for_prediction_features.columns]

    player_overall_avg = combined_data_for_prediction_features[valid_player_stats_cols].mean()


    # Apply a simple weighted average. Example weights: 40% last 5, 30% last 10, 20% last 20, 10% overall career avg
    for stat in stats_for_prediction:
        stat_col = stat
        stat_col_rolling_5 = f'{stat}_rolling_5'
        stat_col_rolling_10 = f'{stat}_rolling_10'
        stat_col_rolling_20 = f'{stat}_rolling_20'
        stat_col_player = f'{stat}_player' if f'{stat}_player' in combined_data_for_prediction_features.columns else stat # Use _player suffix if available

        # Ensure rolling average feature columns exist in the engineered data
        if stat_col_rolling_5 in last_game_engineered_features.index and \
           stat_col_rolling_10 in last_game_engineered_features.index and \
           stat_col_rolling_20 in last_game_engineered_features.index and \
           stat_col_player in player_overall_avg.index: # Check if overall average is available

            recent_5_val = last_game_engineered_features[stat_col_rolling_5] if pd.notna(last_game_engineered_features[stat_col_rolling_5]) else player_overall_avg[stat_col_player]
            recent_10_val = last_game_engineered_features[stat_col_rolling_10] if pd.notna(last_game_engineered_features[stat_col_rolling_10]) else player_overall_avg[stat_col_player]
            recent_20_val = last_game_engineered_features[stat_col_rolling_20] if pd.notna(last_game_engineered_features[stat_col_rolling_20]) else player_overall_avg[stat_col_player]
            overall_avg_val = player_overall_avg[stat_col_player] if pd.notna(player_overall_avg[stat_col_player]) else 0 # Handle potential NaN overall avg


            predicted_value = (0.4 * recent_5_val +
                               0.3 * recent_10_val +
                               0.2 * recent_20_val +
                               0.1 * overall_avg_val) # Include overall average
            predicted_stats[stat] = round(predicted_value, 2) # Round predictions

        else:
             predicted_stats[stat] = "N/A" # Cannot predict if required data is missing

    if predicted_stats:
        return predicted_stats
    else:
        st.info("Could not generate prediction based on available data.")
        return None


# --- Streamlit App Layout and Logic ---

st.title("NBA Player Stats Dashboard")

# Get list of active players for dropdowns
@st.cache_data # Cache this data to avoid refetching every time
def get_active_nba_players():
    try:
        return players.get_active_players()
    except Exception as e:
        st.error(f"Error fetching active players: {e}")
        return []

active_players = get_active_nba_players()

# Get list of all teams for dropdowns
@st.cache_data # Cache this data
def get_all_nba_teams():
    try:
        return teams.get_teams() # Use teams.get_teams()
    except Exception as e:
        st.error(f"Error fetching teams: {e}")
        return []

all_teams = get_all_nba_teams()
# Create a dictionary for team names to IDs and abbreviations
team_name_to_id = {team['full_name']: team['id'] for team in all_teams}
team_abbr_to_id = {team['abbreviation']: team['id'] for team in all_teams}
team_names = sorted(list(team_name_to_id.keys()))
team_abbreviations = sorted(list(team_abbr_to_id.keys()))

# Create a dictionary for player names to IDs
player_name_to_id = {player['full_name']: player['id'] for player in active_players}
player_names = sorted(list(player_name_to_id.keys())) # Sort names alphabetically


# Add dropdown for selecting player
player1_name_select = st.selectbox("Select Player", player_names)

# Fetch comprehensive player data when a player is selected
player1_id = None # Initialize player_id
player_comprehensive_data = None # Initialize comprehensive data dictionary

if player1_name_select:
    player1_id = player_name_to_id.get(player1_name_select)
    if player1_id is not None:
        # Call the updated get_player_stats function to fetch all data
        player_comprehensive_data = get_player_stats(player1_id)

        if player_comprehensive_data:
            st.header(f"{player1_name_select} Career Statistics")

            # Display all season averages in a single table
            career_df_all_seasons = player_comprehensive_data.get('career_df')
            if career_df_all_seasons is not None and not career_df_all_seasons.empty:
                st.subheader("Season Averages")
                # Select relevant columns for display and calculate averages per game
                stats_columns_for_display = ['SEASON_ID', 'TEAM_ABBREVIATION', 'GP', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
                # Ensure columns exist before selecting
                valid_cols_for_display = [col for col in stats_columns_for_display if col in career_df_all_seasons.columns]
                display_career_df = career_df_all_seasons[valid_cols_for_display].copy()

                # Calculate averages per game for relevant columns
                for col in ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']:
                     if col in display_career_df.columns and 'GP' in display_career_df.columns:
                         # Avoid division by zero if GP is 0
                         display_career_df[col] = display_career_df.apply(lambda row: round(row[col] / row['GP'], 2) if row['GP'] > 0 else 0, axis=1) # Round averages here


                st.dataframe(display_career_df) # Display the career dataframe


            else:
                 st.info(f"No career season data available for {player1_name_select}.")

            # Remove the separate displays for historical and overall career averages
            # st.subheader("Career Averages Summary")
            # if player_comprehensive_data.get('historical_career_averages') is not None and not player_comprehensive_data['historical_career_averages'].empty:
            #      st.write("Historical Career (excluding most recent season):")
            #      st.dataframe(player_comprehensive_data['historical_career_averages'].round(2))
            # else:
            #      st.write("Historical Career Averages Not Available.")

            # if player_comprehensive_data.get('overall_career_averages') is not None and not player_comprehensive_data['overall_career_averages'].empty:
            #      st.write("Overall Career (including all seasons):")
            #      st.dataframe(player_comprehensive_data['overall_career_averages'].round(2))
            # else:
            #      st.write("Overall Career Averages Not Available.")

            # Display last season averages
            if player_comprehensive_data.get('last_season_averages') is not None and not player_comprehensive_data['last_season_averages'].empty:
                 last_season_name = player_comprehensive_data['last_season_averages'].index[0].replace('Last Season (','').replace(') Avg','')
                 st.subheader(f"Last Season Averages ({last_season_name})")
                 st.dataframe(player_comprehensive_data['last_season_averages'].round(2))
            else:
                 st.subheader("Last Season Averages Not Available.")

            # Display recent game averages in expanders
            st.subheader("Recent Game Averages (Across Career)")
            if player_comprehensive_data.get('last_5_games_avg') is not None and not player_comprehensive_data['last_5_games_avg'].empty:
                with st.expander("Last 5 Games Average"):
                    st.dataframe(player_comprehensive_data['last_5_games_avg'].round(2))
            else:
                 st.write("Last 5 Games Averages Not Available (not enough recent games in career).")


            if player_comprehensive_data.get('last_10_games_avg') is not None and not player_comprehensive_data['last_10_games_avg'].empty:
                with st.expander("Last 10 Games Average"):
                    st.dataframe(player_comprehensive_data['last_10_games_avg'].round(2))
            else:
                 st.write("Last 10 Games Averages Not Available (not enough recent games in career).")

            if player_comprehensive_data.get('last_20_games_avg') is not None and not player_comprehensive_data['last_20_games_avg'].empty:
                with st.expander("Last 20 Games Average"):
                    st.dataframe(player_comprehensive_data['last_20_games_avg'].round(2))
            else:
                 st.write("Last 20 Games Averages Not Available (not enough recent games in career).")

            # Display individual last 5 games stats
            st.subheader("Last 5 Games (Individual Performance - Most Recent in Career)")
            if player_comprehensive_data.get('last_5_games_individual') is not None and not player_comprehensive_data['last_5_games_individual'].empty:
                 # Select relevant columns and format date
                 cols_to_display = ['GAME_DATE', 'MATCHUP', 'SEASON_YEAR'] + [col for col in stats_columns if col in player_comprehensive_data['last_5_games_individual'].columns]
                 display_df = player_comprehensive_data['last_5_games_individual'][cols_to_display].copy()
                 display_df['GAME_DATE'] = display_df['GAME_DATE'].dt.strftime('%Y-%m-%d') # Format date for display
                 st.dataframe(display_df)
            else:
                 st.write("Individual stats for the last 5 most recent games are not available.")

        else:
            st.error(f"Could not fetch comprehensive data for {player1_name_select}.")


# Add dropdown for selecting opponent team for H2H (using career stats)
opponent_team_name_select = st.selectbox("Select Opponent Team (for Career Stats Against)", [''] + team_names) # Add empty option

# Display player vs team career stats when an opponent team is selected and comprehensive data is available
if player1_name_select and opponent_team_name_select: # Check if a player is selected first
    if player_comprehensive_data is not None and player_comprehensive_data.get('career_game_logs') is not None:
        opponent_team_id = team_name_to_id.get(opponent_team_name_select)
        if opponent_team_id:
            st.subheader(f"{player1_name_select} Career Statistics Against {opponent_team_name_select}")
            # Use the career game logs already fetched
            player_vs_team_career_avg = get_player_vs_team_career_stats(player1_id, opponent_team_id, player_comprehensive_data['career_game_logs'])
            if player_vs_team_career_avg is not None and not player_vs_team_career_avg.empty:
                 st.dataframe(player_vs_team_career_avg.round(2)) # Round for display
            else:
                 st.info(f"{player1_name_select} did not play against {opponent_team_name_select} in their career.")
        else:
             st.error(f"Could not find ID for team: {opponent_team_name_select}")
    # else:
         # st.info("Comprehensive player data not available for Head-to-Head comparison. Please select a player.")


# --- Prediction Section ---
st.header(f"{player1_name_select} Next Game Prediction")

# Determine the latest season from the career data if available, otherwise use current year logic
latest_season_for_prediction_context = None
if player_comprehensive_data is not None and player_comprehensive_data.get('career_df') is not None and not player_comprehensive_data['career_df'].empty:
     latest_season_for_prediction_context = player_comprehensive_data['career_df']['SEASON_ID'].iloc[-1]

if latest_season_for_prediction_context is None:
    # Fallback to current year logic if career data is not available
    current_year = pd.Timestamp.now().year
    latest_season_for_prediction_context = f"{current_year}-{str(current_year+1)[-2:]}"


if st.button(f"Predict Next Game Stats for {player1_name_select}"): # Removed season context from button text
    if not player1_name_select:
        st.warning("Please select a Player first.")
    elif player_comprehensive_data is None or player_comprehensive_data.get('career_game_logs') is None:
         st.info("Comprehensive player data (game logs) not available to generate prediction. Please select a player.")
    else:
        # Fetch and engineer data for prediction using career game logs
        st.write("Preparing data for prediction...")
        combined_data_for_prediction_features = fetch_and_combine_game_data_for_prediction(player1_id, player_comprehensive_data['career_game_logs'])

        if combined_data_for_prediction_features is not None and not combined_data_for_prediction_features.empty:
            # Use the simple prediction function with the combined data for features
            predicted_stats = predict_next_game_stats(player1_name_select, combined_data_for_prediction_features)


            if predicted_stats:
                 st.subheader("Predicted Stats for Next Game:")
                 predicted_df = pd.DataFrame([predicted_stats]) # Convert dict to DataFrame for display
                 st.dataframe(predicted_df)
                 st.caption("Prediction based on a weighted average of recent career game averages.")

                 # Add button to save prediction (using session state)
                 if st.button(f"Save Prediction for {player1_name_select}"):
                      if 'saved_predictions' not in st.session_state:
                          st.session_state.saved_predictions = {}

                      # Store prediction with player name and season as key
                      prediction_key = f"{player1_name_select}" # Store just player name
                      st.session_state.saved_predictions[prediction_key] = {
                           'Predicted Points': predicted_stats.get('PTS', 'N/A'),
                           'Predicted Assists': predicted_stats.get('AST', 'N/A'),
                           'Predicted Rebounds': predicted_stats.get('REB', 'N/A'),
                           'Predicted 3-Pointers': predicted_stats.get('FG3M', 'N/A'),
                           # 'Season Context': latest_season_for_prediction_context # Can add season context if needed
                           # Add other relevant prediction details here
                      }
                      st.success(f"Prediction for {player1_name_select} saved!")


            else:
                st.info("Could not generate prediction for the next game based on available data.")

        else:
            st.info("Could not fetch or combine sufficient game data for prediction.")


# --- Display Saved Predictions (Optional - for multi-page app, this would be on a separate page) ---
# If this were a multi-page app, the code below would be in pages/Saved_Predictions.py
# Here, we'll just display them on the main page if they exist in session state

if 'saved_predictions' in st.session_state and st.session_state.saved_predictions:
    st.subheader("Saved Predictions:")
    saved_predictions_list = []
    for prediction_key, prediction_data in st.session_state.saved_predictions.items():
        saved_predictions_list.append({
            'Player': prediction_key,
            'Predicted Points': prediction_data.get('Predicted Points', 'N/A'),
            'Predicted Assists': prediction_data.get('Predicted Assists', 'N/A'),
            'Predicted Rebounds': prediction_data.get('Predicted Rebounds', 'N/A'),
            'Predicted 3-Pointers': prediction_data.get('Predicted 3-Pointers', 'N/A'),
            # 'Season Context': prediction_data.get('Season Context', 'N/A')
        })
    saved_predictions_df = pd.DataFrame(saved_predictions_list)
    st.dataframe(saved_predictions_df)

    if st.button("Clear All Saved Predictions"):
        st.session_state.saved_predictions = {}
        st.experimental_rerun() # Rerun to update the display
