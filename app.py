import streamlit as st
from nba_api.stats.static import players, teams # Import teams module
from nba_api.stats.endpoints import playercareerstats, playergamelogs, teamgamelogs
import pandas as pd
import re # Import regex for parsing matchup string
import numpy as np # Import numpy for handling NaN values and calculations

# Define core stats columns globally so they are accessible by all functions and the main app
stats_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']


# Define the get_player_stats function (reusing the refined version)
def get_player_stats(player_id, season='2023-24'):
    """
    Fetches player stats and calculates various averages using player ID.
    Includes recent game averages across season boundaries.

    Args:
        player_id (int): The ID of the NBA player.
        season (str): The primary season to fetch data for (e.g., '2023-24').

    Returns:
        dict: A dictionary containing career, last season, and recent game averages,
              or None if data fetching fails.
    """
    try:
        # Fetch career stats
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        career_df = career_stats.get_data_frames()[0]

        # Calculate career averages
        # stats_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] # Moved to global scope

        # Filter out the row for the current season for historical career averages
        historical_career_averages = None
        if len(career_df) > 1:
            historical_career_df = career_df.iloc[:-1]
            total_games = historical_career_df['GP'].sum()
            # Ensure stats_columns only contains columns present in historical_career_df
            valid_stats_columns_hist = [col for col in stats_columns if col in historical_career_df.columns]
            career_totals = historical_career_df[valid_stats_columns_hist].sum()
            historical_career_averages = career_totals / total_games
            historical_career_averages = historical_career_averages.to_frame(name='Historical Career Avg').T # Convert to DataFrame for display


        # Calculate overall career averages
        overall_total_games = career_df['GP'].sum()
        overall_career_totals = career_df[stats_columns].sum()
        # Ensure stats_columns only contains columns present in overall_career_totals
        valid_stats_columns_overall = [col for col in stats_columns if col in overall_career_totals.index]
        overall_career_averages = overall_career_totals[valid_stats_columns_overall] / overall_total_games
        overall_career_averages = overall_career_averages.to_frame(name='Overall Career Avg').T # Convert to DataFrame for display


        # Get last season's averages (Corrected to calculate averages per game)
        last_season_averages = None
        last_season_id = None
        # Find the row for the selected season
        if season in career_df['SEASON_ID'].values:
            season_index = career_df[career_df['SEASON_ID'] == season].index[0]
            if season_index > 0:
                 last_season_df = career_df.iloc[season_index - 1]
                 last_season_id = last_season_df['SEASON_ID'] # Get last season's ID
                 last_season_games = last_season_df['GP']
                 if last_season_games > 0:
                     # Ensure stats_columns only contains columns present in last_season_df
                     valid_stats_columns_last = [col for col in stats_columns if col in last_season_df.index]
                     last_season_totals = last_season_df[valid_stats_columns_last]
                     last_season_averages = last_season_totals / last_season_games
                     last_season_averages = last_season_averages.to_frame(name=f"Last Season ({last_season_id}) Avg").T # Convert to DataFrame for display
                 else:
                      st.info(f"Player played 0 games in the season before {season}. Last season averages not available.")

            # If season_index is 0 and len(career_df) > 1, the selected season is the earliest, no last season before it
            # If season_index is 0 and len(career_df) == 1, only one season available, no last season
        # else:
             # st.warning(f"Selected season {season} not found in career history. Cannot determine last season's averages relative to this season.")


        # Fetch game logs for the specified season and the previous season
        game_logs_df = None
        last_5_games_avg = None
        last_10_games_avg = None
        last_20_games_avg = None
        last_5_games_individual = None

        all_game_logs_list = []

        try:
            # Fetch game logs for the current season
            current_season_game_logs = playergamelogs.PlayerGameLogs(player_id_nullable=player_id, season_nullable=season).get_data_frames()[0]
            all_game_logs_list.append(current_season_game_logs)

            # Fetch game logs for the last season if available
            if last_season_id:
                 last_season_game_logs = playergamelogs.PlayerGameLogs(player_id_nullable=player_id, season_nullable=last_season_id).get_data_frames()[0]
                 all_game_logs_list.append(last_season_game_logs)


            # Combine game logs from all fetched seasons
            if all_game_logs_list:
                 game_logs_df = pd.concat(all_game_logs_list, ignore_index=True)


            if game_logs_df is not None and not game_logs_df.empty:
                # Ensure game logs are sorted by date in descending order (most recent first)
                game_logs_df['GAME_DATE'] = pd.to_datetime(game_logs_df['GAME_DATE'])
                game_logs_df = game_logs_df.sort_values(by='GAME_DATE', ascending=False)

                # Calculate recent game averages from the most recent games
                # Ensure stats_columns only contains columns present in game_logs_df
                valid_game_stats_columns = [col for col in stats_columns if col in game_logs_df.columns]

                if len(game_logs_df) >= 5:
                    last_5_games_avg = game_logs_df.head(5)[valid_game_stats_columns].mean()
                    last_5_games_avg = last_5_games_avg.to_frame(name='Last 5 Games Avg').T # Convert to DataFrame
                    last_5_games_individual = game_logs_df.head(5)[['GAME_DATE', 'MATCHUP', 'SEASON_YEAR'] + valid_game_stats_columns] # Get individual game stats, include season year
                if len(game_logs_df) >= 10:
                    last_10_games_avg = game_logs_df.head(10)[valid_game_stats_columns].mean()
                    last_10_games_avg = last_10_games_avg.to_frame(name='Last 10 Games Avg').T # Convert to DataFrame
                if len(game_logs_df) >= 20:
                    last_20_games_avg = game_logs_df.head(20)[valid_game_stats_columns].mean()
                    last_20_games_avg = last_20_games_avg.to_frame(name='Last 20 Games Avg').T # Convert to DataFrame


        except Exception as e:
            st.warning(f"Could not fetch game logs for recent game calculations across seasons. Error: {e}")


        return {
            'career_df': career_df, # Return career_df to display all season averages
            'historical_career_averages': historical_career_averages,
            'overall_career_averages': overall_career_averages,
            'last_season_averages': last_season_averages,
            'last_5_games_avg': last_5_games_avg,
            'last_10_games_avg': last_10_games_avg,
            'last_20_games_avg': last_20_games_avg,
            'game_logs_df': game_logs_df, # Return game logs for prediction feature engineering
            'last_5_games_individual': last_5_games_individual # Return individual last 5 games
        }

    except Exception as e:
        st.error(f"An error occurred while fetching career stats: {e}")
        return None

# Define the get_player_vs_team_stats function (MODIFIED for player vs team)
def get_player_vs_team_stats(player_id, team_id, season='2023-24'):
    """
    Fetches and calculates a player's statistics against a specific team.

    Args:
        player_id (int): The ID of the player.
        team_id (int): The ID of the opponent team.
        season (str): The season to fetch game logs for (e.g., '2023-24').

    Returns:
        DataFrame: A pandas DataFrame containing the player's averages against
                   the specified team, or None if data fetching fails or they
                   didn't play against that team in the specified season.
    """
    # stats_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] # Moved to global scope

    try:
        # Fetch game logs for the player for the specified season
        player_logs = playergamelogs.PlayerGameLogs(player_id_nullable=player_id, season_nullable=season).get_data_frames()[0]
        player_logs['GAME_DATE'] = pd.to_datetime(player_logs['GAME_DATE'])

        # Fetch game logs for the opponent team for the specified season
        # We need this to get the opponent team's ID from the matchup string
        # all_team_logs = teamgamelogs.TeamGameLogs(season_nullable=season).get_data_frames()[0] # Not directly needed here

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

        # Filter player logs for games against the specific opponent team
        # We can identify opponent by checking if the opponent team abbreviation is in the MATCHUP string
        # This requires parsing the MATCHUP string
        def played_against_team(matchup, opponent_abbr_to_find): # Renamed parameter to avoid conflict
            # Check if the opponent abbreviation appears after '@' or 'vs.' in the matchup string
            match = re.search(rf'@\s*{opponent_abbr_to_find}|vs.\s*{opponent_abbr_to_find}|VS.\s*{opponent_abbr_to_find}', matchup, re.IGNORECASE) # Use the parameter
            # Refined check: also ensure the found abbreviation is not the player's team abbreviation in this game
            if match:
                 found_abbr = match.group(1) or match.group(2) or match.group(3)
                 # Need player's team abbreviation for the current game to avoid self-matches
                 # This is complex to get accurately within this function without game logs.
                 # For simplicity, we'll rely on the strict matching for now.
                 return bool(found_abbr) # Return True if a match is found

            return False # Return False if no match

        player_logs_vs_team = player_logs[
            player_logs['MATCHUP'].apply(lambda x: played_against_team(x, opponent_team_abbr)) # Pass the found opponent_team_abbr
        ].copy() # Use .copy() to avoid SettingWithCopyWarning


        if player_logs_vs_team.empty:
            # st.info(f"Player {player_id} did not play against team {team_id} in the {season} season.")
            return None

        # Calculate player's averages in games against the specific team
        # Ensure stats_columns only contains columns present in player_logs_vs_team
        valid_stats_columns_vs_team = [col for col in stats_columns if col in player_logs_vs_team.columns]
        player_vs_team_avg = player_logs_vs_team[valid_stats_columns_vs_team].mean()
        player_vs_team_avg = player_vs_team_avg.to_frame(name=f"Avg vs {opponent_team_abbr} ({season})").T # Convert to DataFrame

        return player_vs_team_avg

    except Exception as e:
        st.error(f"An error occurred while fetching player vs team stats: {e}")
        return None

# Define the fetch_and_combine_game_data function for prediction features
def fetch_and_combine_game_data(player_id, season):
    """
    Fetches player and team game logs for a given season and combines them
    for feature engineering for prediction.
    Includes game logs from the previous season for potentially more data.


    Args:
        player_id (int): The ID of the NBA player.
        season (str): The primary season to fetch data for (e.g., '2023-24').

    Returns:
        DataFrame: A DataFrame containing combined player and opponent team stats
                   for each game the player played in that season, or None if
                   data fetching fails.
    """
    all_game_logs_list = []
    combined_data = None

    try:
        # Fetch player game logs for the current season
        player_logs_current = playergamelogs.PlayerGameLogs(player_id_nullable=player_id, season_nullable=season).get_data_frames()[0]
        player_logs_current['GAME_DATE'] = pd.to_datetime(player_logs_current['GAME_DATE'])
        all_game_logs_list.append(player_logs_current)

        # Determine the previous season
        current_season_year = int(season.split('-')[0])
        previous_season_year = current_season_year - 1
        previous_season = f"{previous_season_year}-{str(previous_season_year + 1)[-2:]}"


        # Fetch player game logs for the previous season
        try:
            player_logs_previous = playergamelogs.PlayerGameLogs(player_id_nullable=player_id, season_nullable=previous_season).get_data_frames()[0]
            player_logs_previous['GAME_DATE'] = pd.to_datetime(player_logs_previous['GAME_DATE'])
            all_game_logs_list.append(player_logs_previous)
        except Exception as e:
             st.warning(f"Could not fetch game logs for the previous season ({previous_season}). Skipping previous season data for combined logs. Error: {e}")


        # Fetch all team game logs for the current and previous seasons to easily access opponent team data
        all_team_logs_list = []
        try:
            all_team_logs_current = teamgamelogs.TeamGameLogs(season_nullable=season).get_data_frames()[0]
            all_team_logs_current['GAME_DATE'] = pd.to_datetime(all_team_logs_current['GAME_DATE'])
            all_team_logs_list.append(all_team_logs_current)

            # Determine the last season ID fetched in get_player_stats for team logs as well
            last_season_id_fetched = None
            # Re-fetch career stats to get the last season ID reliably
            try:
                career_df_check = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]
                if season in career_df_check['SEASON_ID'].values:
                    season_index_in_career = career_df_check[career_df_check['SEASON_ID'] == season].index[0]
                    if season_index_in_career > 0:
                        last_season_id_fetched = career_df_check.iloc[season_index_in_career - 1]['SEASON_ID']
            except Exception as e:
                 st.warning(f"Could not fetch career stats to determine last season ID for team logs. Error: {e}")


            if last_season_id_fetched: # Check if last_season_id was successfully determined
                 all_team_logs_previous = teamgamelogs.TeamGameLogs(season_nullable=last_season_id_fetched).get_data_frames()[0]
                 all_team_logs_previous['GAME_DATE'] = pd.to_datetime(all_team_logs_previous['GAME_DATE'])
                 all_team_logs_list.append(all_team_logs_previous)

        except Exception as e:
             st.warning(f"Could not fetch team game logs for combined data across seasons. Error: {e}")

        all_team_logs_combined = pd.concat(all_team_logs_list, ignore_index=True) if all_team_logs_list else pd.DataFrame()


        if all_game_logs_list and not all_team_logs_combined.empty:
            player_logs_combined = pd.concat(all_game_logs_list, ignore_index=True)

            # Merge player logs with team logs to get player's team's stats for each game
            # Need to handle potential duplicate columns carefully or merge in stages
            # Let's merge player logs with all team logs to get player's team stats for that game
            player_team_combined = pd.merge(
                player_logs_combined,
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
                    opponent_abbr = match.group(1) or match.group(2) or match.group(3) # Capture from any group
                    # Basic check to ensure the extracted opponent is not the player's team
                    if opponent_abbr and player_team_abbr and opponent_abbr.upper() != player_team_abbr.upper():
                        return opponent_abbr.upper() # Return uppercase for consistent matching
                    elif opponent_abbr: # Handle cases where the team might be the opponent but the abbreviation matches player's team (e.g., both are LAL in different games)
                         # Need a more robust check here, maybe compare team IDs from game logs if available
                         pass # For now, rely on the basic check
                return None # Return None if no match or if it matches player's team abbr

            # Apply the function to extract opponent team abbreviation
            player_team_combined['OPPONENT_TEAM_ABBREVIATION'] = player_team_combined.apply(
                lambda row: extract_opponent_team_abbr(row.get('MATCHUP_player', ''), row.get('TEAM_ABBREVIATION_player', '')), axis=1
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

        elif all_game_logs_list:
             st.warning("Could not fetch team logs. Combined data with opponent stats will not be available.")
             combined_data = pd.concat(all_game_logs_list, ignore_index=True) # Still return player logs if team logs failed


        return combined_data

    except Exception as e:
        st.error(f"An error occurred while fetching and combining data for prediction: {e}")
        return None

# Define the feature engineering function
def engineer_features(combined_data):
    """
    Engineers features and defines targets for prediction from combined game data.
    Calculates rolling averages and rest days based on combined data across seasons.


    Args:
        combined_data (DataFrame): DataFrame containing combined player and opponent game data.

    Returns:
        tuple: A tuple containing the feature DataFrame (X), target DataFrame (y),
               and the cleaned DataFrame with features and targets.
               Returns (None, None, None) if data is insufficient.
    """
    if combined_data is None or combined_data.empty:
        st.warning("No data available for feature engineering.")
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
        combined_data[f'{col}_rolling_5'] = combined_data.groupby('PLAYER_ID')[f'{col}_player'].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
        combined_data[f'{col}_rolling_10'] = combined_data.groupby('PLAYER_ID')[f'{col}_player'].transform(lambda x: x.rolling(window=10, min_periods=1).mean().shift(1))
        combined_data[f'{col}_rolling_20'] = combined_data.groupby('PLAYER_ID')[f'{col}_player'].transform(lambda x: x.rolling(window=20, min_periods=1).mean().shift(1))

    # Create a binary feature indicating home or away game
    def is_home_game(matchup):
        # Check if the matchup string contains "vs." (typically indicates a home game)
        return 1 if 'vs.' in matchup else 0

    combined_data['IS_HOME_player'] = combined_data['MATCHUP_player'].apply(is_home_game)

    # Calculate rest days since the last game across combined seasons
    combined_data['PREV_GAME_DATE'] = combined_data.groupby('PLAYER_ID')['GAME_DATE'].shift(1)
    combined_data['REST_DAYS_player'] = (combined_data['GAME_DATE'] - combined_data['PREV_GAME_DATE']).dt.days.fillna(0) # Fill NaN for the first game


    # Select feature columns - adjusted to account for combined data
    # Include rolling averages
    feature_columns = [col for col in combined_data.columns if '_rolling_' in col]

    # Include opponent statistics if available (check if any opponent columns exist)
    opponent_cols = [col for col in combined_data.columns if '_opponent' in col and col not in ['SEASON_YEAR_opponent']]
    if opponent_cols:
         feature_columns.extend(opponent_cols)
    else:
         st.warning("Opponent statistics are not available for feature engineering.")

    # Include home/away and rest days features
    feature_columns.append('IS_HOME_player')
    feature_columns.append('REST_DAYS_player')

    # Define target variables (the actual stats from the current game)
    target_columns = [f'{col}_player' for col in prediction_targets]

    # Create X and y, dropping rows with NaNs created by rolling averages or missing opponent data
    # We also need to shift the target variables up by one row to predict the *next* game's stats
    # Ensure all required columns exist before dropping NaNs
    required_cols = feature_columns + target_columns
    combined_data_cleaned = combined_data.dropna(subset=required_cols).copy()
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

# Define a simple prediction function (simplified approach)
def predict_next_game_stats(player_id, season, player_name, combined_data_for_prediction):
    """
    Generates a simplified prediction for the next game's stats using combined data.
    This uses a weighted average of recent game averages and season average from the
    combined data across seasons.

    Args:
        player_id (int): The ID of the NBA player.
        season (str): The primary season for which data was fetched.
        player_name (str): The name of the player.
        combined_data_for_prediction (DataFrame): DataFrame containing combined player
                                                  and opponent game data across seasons.

    Returns:
        dict: A dictionary containing predicted stats, or None if prediction is not possible.
    """
    if combined_data_for_prediction is None or combined_data_for_prediction.empty:
        st.info("No combined data available to generate prediction.")
        return None

    # Ensure data is sorted by date before calculating recent averages
    combined_data_for_prediction['GAME_DATE'] = pd.to_datetime(combined_data_for_prediction['GAME_DATE'])
    combined_data_for_prediction = combined_data_for_prediction.sort_values(by='GAME_DATE', ascending=True).reset_index(drop=True)


    # Calculate rolling averages from the combined player's game logs (across seasons)
    # stats_columns_pred = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] # Moved to global scope
    valid_stats_columns_pred = [col for col in stats_columns if col in combined_data_for_prediction.columns] # Use the global stats_columns


    player_features_for_pred = {}
    if not combined_data_for_prediction.empty:
        for col in valid_stats_columns_pred:
            # Calculate rolling averages and get the last value (most recent game)
            # Handle potential errors if not enough games for rolling window
            try:
                 rolling_5 = combined_data_for_prediction[col].rolling(window=5, min_periods=1).mean().iloc[-1]
            except IndexError:
                 rolling_5 = np.nan # Not enough games for rolling 5
            try:
                 rolling_10 = combined_data_for_prediction[col].rolling(window=10, min_periods=1).mean().iloc[-1]
            except IndexError:
                 rolling_10 = np.nan # Not enough games for rolling 10
            try:
                 rolling_20 = combined_data_for_prediction[col].rolling(window=20, min_periods=1).mean().iloc[-1]
            except IndexError:
                 rolling_20 = np.nan # Not enough games for rolling 20


            player_features_for_pred[f'{col}_rolling_5'] = rolling_5 # Store with original stat name prefix
            player_features_for_pred[f'{col}_rolling_10'] = rolling_10
            player_features_for_pred[f'{col}_rolling_20'] = rolling_20

        # Calculate season average for the player from the game logs (using the primary selected season)
        # Filter game logs for the primary selected season
        current_season_data = combined_data_for_prediction[combined_data_for_prediction['SEASON_YEAR_player'] == season]
        player_season_avg_pred = current_season_data[valid_stats_columns_pred].mean() if not current_season_data.empty else combined_data_for_prediction[valid_stats_columns_pred].mean() # Fallback to overall average if current season data is empty


        # Generate prediction using the simplified approach with calculated features
        predicted_stats = {}
        stats_for_prediction = ['PTS', 'AST', 'REB', 'FG3M']

        for stat in stats_for_prediction:
            stat_col = stat
            stat_col_player = f'{stat}_player'
            stat_col_rolling_5 = f'{stat}_rolling_5'
            stat_col_rolling_10 = f'{stat}_rolling_10'
            stat_col_rolling_20 = f'{stat}_rolling_20'

            # Ensure rolling average features exist in player_features_for_pred before accessing
            if stat_col_rolling_5 in player_features_for_pred and \
               stat_col_rolling_10 in player_features_for_pred and \
               stat_col_rolling_20 in player_features_for_pred:


                recent_5_val = player_features_for_pred[stat_col_rolling_5] if pd.notna(player_features_for_pred[stat_col_rolling_5]) else (player_season_avg_pred[stat_col] if stat_col in player_season_avg_pred.index and pd.notna(player_season_avg_pred[stat_col]) else 0)
                recent_10_val = player_features_for_pred[stat_col_rolling_10] if pd.notna(player_features_for_pred[stat_col_rolling_10]) else (player_season_avg_pred[stat_col] if stat_col in player_season_avg_pred.index and pd.notna(player_season_avg_pred[stat_col]) else 0)
                recent_20_val = player_features_for_pred[stat_col_rolling_20] if pd.notna(player_features_for_pred[stat_col_rolling_20]) else (player_season_avg_pred[stat_col] if stat_col in player_season_avg_pred.index and pd.notna(player_season_avg_pred[stat_col]) else 0)
                season_avg_val = player_season_avg_pred[stat_col] if stat_col in player_season_avg_pred.index and pd.notna(player_season_avg_pred[stat_col]) else 0 # Handle potential NaN season avg if no games played


                predicted_value = (0.4 * recent_5_val +
                                   0.3 * recent_10_val +
                                   0.2 * recent_20_val +
                                   0.1 * season_avg_val)
                predicted_stats[stat] = round(predicted_value, 2) # Round predictions

            else:
                 predicted_stats[stat] = "N/A" # Cannot predict if data is missing


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

# Fetch player stats including career_df when a player is selected
player1_stats = None
career_df_all_seasons = None # Initialize to None
if player1_name_select:
    player1_id = player_name_to_id.get(player1_name_select)
    if player1_id is not None:
        # Fetch career_df to display all season averages
        try:
            career_stats_all_seasons = playercareerstats.PlayerCareerStats(player_id=player1_id)
            career_df_all_seasons = career_stats_all_seasons.get_data_frames()[0]

            # Display all season averages in a table
            if career_df_all_seasons is not None and not career_df_all_seasons.empty:
                st.subheader(f"{player1_name_select} Season Averages")
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

        except Exception as e:
            st.error(f"An error occurred while fetching career stats for display: {e}")


# Add dropdown for selecting opponent team for H2H
opponent_team_name_select = st.selectbox("Select Opponent Team (for Stats Against)", [''] + team_names) # Add empty option


# Add a button to trigger detailed stats and predictions for the latest season
# Determine the latest season from the career data if available, otherwise use current year logic
# Generate a list of seasons from 2000-01 to current/next season (Moved outside the button block)
current_year = pd.Timestamp.now().year
seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2000, current_year + 1)]

latest_season_for_detailed_view = seasons[-1] # Default to the latest season in the generated list
if career_df_all_seasons is not None and not career_df_all_seasons.empty:
     latest_season_for_detailed_view = career_df_all_seasons['SEASON_ID'].iloc[-1]


if st.button(f"Get Detailed Stats and Predictions for {latest_season_for_detailed_view}"):
    if not player1_name_select:
        st.warning("Please select a Player.")
    else:
        player1_id = player_name_to_id.get(player1_name_select)
        if player1_id is None:
             st.error(f"Could not find ID for player: {player1_name_select}")
        else:
            # Call the get_player_stats function for the selected player and latest season
            player1_stats = get_player_stats(player1_id, season=latest_season_for_detailed_view)

            if player1_stats:
                st.header(f"{player1_name_select} Detailed Statistics ({latest_season_for_detailed_view} Season)")

                # Display career averages (already displayed above, can remove or keep for redundancy)
                # st.subheader("Career Averages")
                # if player1_stats.get('historical_career_averages') is not None and not player1_stats['historical_career_averages'].empty:
                #     st.write("Historical Career (excluding current season):")
                #     st.dataframe(player1_stats['historical_career_averages'].round(2)) # Round for display
                # else:
                #      st.write("Historical Career Averages Not Available.")


                # if player1_stats.get('overall_career_averages') is not None and not player1_stats['overall_career_averages'].empty:
                #     st.write("Overall Career (including current season):")
                #     st.dataframe(player1_stats['overall_career_averages'].round(2)) # Round for display
                # else:
                #      st.write("Overall Career Averages Not Available.")


                if player1_stats.get('last_season_averages') is not None and not player1_stats['last_season_averages'].empty:
                     st.subheader(f"Last Season Averages ({player1_stats['last_season_averages'].index[0].replace('Last Season (','').replace(') Avg','')})") # Extract season year from index
                     st.dataframe(player1_stats['last_season_averages'].round(2)) # Round for display
                else:
                     st.subheader("Last Season Averages Not Available.")


                # Display recent game averages in expanders
                st.subheader("Recent Game Averages (Across Seasons if Needed)")
                if player1_stats.get('last_5_games_avg') is not None and not player1_stats['last_5_games_avg'].empty:
                    with st.expander("Last 5 Games Average"):
                        st.dataframe(player1_stats['last_5_games_avg'].round(2)) # Round for display
                else:
                     st.write("Last 5 Games Averages Not Available (not enough recent games).")


                if player1_stats.get('last_10_games_avg') is not None and not player1_stats['last_10_games_avg'].empty:
                    with st.expander("Last 10 Games Average"):
                        st.dataframe(player1_stats['last_10_games_avg'].round(2)) # Round for display
                else:
                     st.write("Last 10 Games Averages Not Available (not enough recent games).")

                if player1_stats.get('last_20_games_avg') is not None and not player1_stats['last_20_games_avg'].empty:
                    with st.expander("Last 20 Games Average"):
                        st.dataframe(player1_stats['last_20_games_avg'].round(2)) # Round for display
                else:
                     st.write("Last 20 Games Averages Not Available (not enough recent games).")

                # Display individual last 5 games stats
                st.subheader("Last 5 Games (Individual Performance - Most Recent)")
                if player1_stats.get('last_5_games_individual') is not None and not player1_stats['last_5_games_individual'].empty:
                     # Select relevant columns and format date
                     cols_to_display = ['GAME_DATE', 'MATCHUP', 'SEASON_YEAR'] + [col for col in stats_columns if col in player1_stats['last_5_games_individual'].columns] # Use the global stats_columns
                     display_df = player1_stats['last_5_games_individual'][cols_to_display].copy()
                     display_df['GAME_DATE'] = display_df['GAME_DATE'].dt.strftime('%Y-%m-%d') # Format date for display
                     st.dataframe(display_df)
                else:
                     st.write("Individual stats for the last 5 most recent games are not available.")


                # --- Player vs Team Stats ---
                # Moved this section here to be triggered by the "Get Detailed Stats" button
                if opponent_team_name_select:
                     opponent_team_id = team_name_to_id.get(opponent_team_name_select)
                     if opponent_team_id:
                         st.header(f"{player1_name_select} Statistics Against {opponent_team_name_select} ({latest_season_for_detailed_view} Season)")
                         player_vs_team_avg = get_player_vs_team_stats(player1_id, opponent_team_id, season=latest_season_for_detailed_view)
                         if player_vs_team_avg is not None and not player_vs_team_avg.empty:
                              st.dataframe(player_vs_team_avg.round(2)) # Round for display
                         else:
                              st.info(f"{player1_name_select} did not play against {opponent_team_name_select} in the {latest_season_for_detailed_view} season.")
                     else:
                          st.error(f"Could not find ID for team: {opponent_team_name_select}")
                else:
                     st.info("Select an Opponent Team to see Player vs Team statistics.")


                # --- Prediction Logic ---
                st.header(f"{player1_name_select} Next Game Prediction ({latest_season_for_detailed_view} Season)")
                # Fetch and engineer data for prediction
                combined_data = fetch_and_combine_game_data(player1_id, latest_season_for_detailed_view)

                if combined_data is not None:
                    # engineer_features also handles dropping NaNs and shifting targets for model training,
                    # but we need the last row *before* the shift for prediction features.
                    # Let's engineer features without shifting targets for prediction input.
                    # We'll create a separate function or modify engineer_features to handle this.

                    # Simpler approach: directly calculate features needed for prediction
                    # from the fetched game logs, rather than re-using the training feature engineering.
                    # This avoids the complexity of handling the "next game" target shift during feature engineering for prediction.

                    # Use the combined game logs fetched in get_player_stats for prediction features
                    game_logs_for_pred = player1_stats.get('game_logs_df')


                    if game_logs_for_pred is not None and not game_logs_for_pred.empty:
                        # Calculate rolling averages from the player's game logs (across seasons)
                        # stats_columns_pred = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] # Moved to global scope
                        valid_stats_columns_pred = [col for col in stats_columns if col in game_logs_for_pred.columns] # Use the global stats_columns


                        # Sort game logs by date ascending for rolling window calculation
                        game_logs_for_pred_sorted = game_logs_for_pred.sort_values(by='GAME_DATE', ascending=True).copy()

                        player_features_for_pred = {}
                        if not game_logs_for_pred_sorted.empty:
                             for col in valid_stats_columns_pred:
                                # Calculate rolling averages and get the last value (most recent game)
                                # Handle potential errors if not enough games for rolling window
                                try:
                                     rolling_5 = game_logs_for_pred_sorted[col].rolling(window=5, min_periods=1).mean().iloc[-1]
                                except IndexError:
                                     rolling_5 = np.nan # Not enough games for rolling 5
                                try:
                                     rolling_10 = game_logs_for_pred_sorted[col].rolling(window=10, min_periods=1).mean().iloc[-1]
                                except IndexError:
                                     rolling_10 = np.nan # Not enough games for rolling 10
                                try:
                                     rolling_20 = game_logs_for_pred_sorted[col].rolling(window=20, min_periods=1).mean().iloc[-1]
                                except IndexError:
                                     rolling_20 = np.nan # Not enough games for rolling 20


                                player_features_for_pred[f'{col}_rolling_5'] = rolling_5 # Store with original stat name prefix
                                player_features_for_pred[f'{col}_rolling_10'] = rolling_10
                                player_features_for_pred[f'{col}_rolling_20'] = rolling_20

                             # Calculate season average for the player from the game logs (using the primary selected season)
                             # Filter game logs for the primary selected season
                             current_season_game_logs_for_avg = game_logs_for_pred_sorted[game_logs_for_pred_sorted['SEASON_YEAR'] == latest_season_for_detailed_view]
                             player_season_avg_pred = current_season_game_logs_for_avg[valid_stats_columns_pred].mean() if not current_season_game_logs_for_avg.empty else game_logs_for_pred_sorted[valid_stats_columns_pred].mean() # Fallback to overall average if current season data is empty


                             # Generate prediction using the simplified approach with calculated features
                             predicted_stats = {}
                             stats_for_prediction = ['PTS', 'AST', 'REB', 'FG3M']

                             for stat in stats_for_prediction:
                                 stat_col = stat
                                 stat_col_player = f'{stat}_player'
                                 stat_col_rolling_5 = f'{stat}_rolling_5'
                                 stat_col_rolling_10 = f'{stat}_rolling_10'
                                 stat_col_rolling_20 = f'{stat}_rolling_20'

                                 # Ensure rolling average features exist in player_features_for_pred before accessing
                                 if stat_col_rolling_5 in player_features_for_pred and \
                                    stat_col_rolling_10 in player_features_for_pred and \
                                    stat_col_rolling_20 in player_features_for_pred:


                                     recent_5_val = player_features_for_pred[stat_col_rolling_5] if pd.notna(player_features_for_pred[stat_col_rolling_5]) else (player_season_avg_pred[stat_col] if stat_col in player_season_avg_pred.index and pd.notna(player_season_avg_pred[stat_col]) else 0)
                                     recent_10_val = player_features_for_pred[stat_col_rolling_10] if pd.notna(player_features_for_pred[stat_col_rolling_10]) else (player_season_avg_pred[stat_col] if stat_col in player_season_avg_pred.index and pd.notna(player_season_avg_pred[stat_col]) else 0)
                                     recent_20_val = player_features_for_pred[stat_col_rolling_20] if pd.notna(player_features_for_pred[stat_col_rolling_20]) else (player_season_avg_pred[stat_col] if stat_col in player_season_avg_pred.index and pd.notna(player_season_avg_pred[stat_col]) else 0)
                                     season_avg_val = player_season_avg_pred[stat_col] if stat_col in player_season_avg_pred.index and pd.notna(player_season_avg_pred[stat_col]) else 0 # Handle potential NaN season avg if no games played


                                     predicted_value = (0.4 * recent_5_val +
                                                        0.3 * recent_10_val +
                                                        0.2 * recent_20_val +
                                                        0.1 * season_avg_val)
                                     predicted_stats[stat] = round(predicted_value, 2) # Round predictions

                                 else:
                                      predicted_stats[stat] = "N/A" # Cannot predict if data is missing


                             if predicted_stats:
                                 st.subheader("Predicted Stats for Next Game:")
                                 predicted_df = pd.DataFrame([predicted_stats]) # Convert dict to DataFrame for display
                                 st.dataframe(predicted_df)
                                 st.caption("Prediction based on a weighted average of recent game averages and season average.")

                                 # Add button to save prediction (using session state)
                                 if st.button(f"Save Prediction for {player1_name_select}"):
                                      if 'saved_predictions' not in st.session_state:
                                          st.session_state.saved_predictions = {}

                                      # Store prediction with player name and season as key
                                      prediction_key = f"{player1_name_select} ({latest_season_for_detailed_view})"
                                      st.session_state.saved_predictions[prediction_key] = {
                                           'PTS': predicted_stats.get('PTS', 'N/A'),
                                           'AST': predicted_stats.get('AST', 'N/A'),
                                           'REB': predicted_stats.get('REB', 'N/A'),
                                           'FG3M': predicted_stats.get('FG3M', 'N/A'),
                                           'Season': latest_season_for_detailed_view # Store season as well
                                           # Add other relevant prediction details here
                                      }
                                      st.success(f"Prediction for {player1_name_select} saved!")


                             else:
                                 st.info("Could not generate prediction for the next game based on available data.")
                        else:
                             st.info("Insufficient game logs available to generate prediction.")


                else:
                     st.info("Could not fetch combined game data for prediction.")


            else:
                st.error(f"Could not fetch stats for {player1_name_select}.")
