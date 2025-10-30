import streamlit as st
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats, playergamelogs, teamgamelogs
import pandas as pd
import re # Import regex for parsing matchup string
import numpy as np # Import numpy for handling NaN values and calculations


# Define the get_player_stats function (reusing the refined version)
def get_player_stats(player_id, season='2023-24'):
    """
    Fetches player stats and calculates various averages using player ID.

    Args:
        player_id (int): The ID of the NBA player.
        season (str): The season to fetch game logs for (e.g., '2023-24').

    Returns:
        dict: A dictionary containing career, last season, and recent game averages,
              or None if data fetching fails.
    """
    try:
        # Fetch career stats
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        career_df = career_stats.get_data_frames()[0]

        # Calculate career averages
        stats_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

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


        # Get last season's averages
        last_season_averages = None
        # Find the row for the selected season or the season before it if selected season is the current
        # Assuming career_df is sorted by season
        # Need to handle cases where the selected season might not be in the career_df
        if season in career_df['SEASON_ID'].values:
            season_index = career_df[career_df['SEASON_ID'] == season].index[0]
            if season_index > 0:
                 last_season_df = career_df.iloc[season_index - 1]
                 # Ensure stats_columns only contains columns present in last_season_df
                 valid_stats_columns_last = [col for col in stats_columns if col in last_season_df.index]
                 last_season_averages = last_season_df[valid_stats_columns_last]
                 last_season_averages = last_season_averages.to_frame(name=f"Last Season ({last_season_df['SEASON_ID']}) Avg").T # Convert to DataFrame for display
            # If season_index is 0 and len(career_df) > 1, the selected season is the earliest, no last season before it
            # If season_index is 0 and len(career_df) == 1, only one season available, no last season
        # else:
             # st.warning(f"Selected season {season} not found in career history. Cannot determine last season's averages relative to this season.")


        # Fetch game logs for the specified season
        game_logs_df = None
        last_5_games_avg = None
        last_10_games_avg = None
        last_20_games_avg = None

        try:
            game_logs = playergamelogs.PlayerGameLogs(player_id_nullable=player_id, season_nullable=season)
            game_logs_df = game_logs.get_data_frames()[0]

            # Ensure game logs are sorted by date
            game_logs_df['GAME_DATE'] = pd.to_datetime(game_logs_df['GAME_DATE'])
            game_logs_df = game_logs_df.sort_values(by='GAME_DATE', ascending=False)

            # Calculate recent game averages
            # Ensure stats_columns only contains columns present in game_logs_df
            valid_game_stats_columns = [col for col in stats_columns if col in game_logs_df.columns]

            if len(game_logs_df) >= 5:
                last_5_games_avg = game_logs_df.head(5)[valid_game_stats_columns].mean()
                last_5_games_avg = last_5_games_avg.to_frame(name='Last 5 Games Avg').T # Convert to DataFrame
            if len(game_logs_df) >= 10:
                last_10_games_avg = game_logs_df.head(10)[valid_game_stats_columns].mean()
                last_10_games_avg = last_10_games_avg.to_frame(name='Last 10 Games Avg').T # Convert to DataFrame
            if len(game_logs_df) >= 20:
                last_20_games_avg = game_logs_df.head(20)[valid_game_stats_columns].mean()
                last_20_games_avg = last_20_games_avg.to_frame(name='Last 20 Games Avg').T # Convert to DataFrame


        except Exception as e:
            st.warning(f"Could not fetch game logs for the selected season ({season}). Recent game averages will not be available. Error: {e}")


        return {
            'historical_career_averages': historical_career_averages,
            'overall_career_averages': overall_career_averages,
            'last_season_averages': last_season_averages,
            'last_5_games_avg': last_5_games_avg,
            'last_10_games_avg': last_10_games_avg,
            'last_20_games_avg': last_20_games_avg,
            'game_logs_df': game_logs_df # Return game logs for prediction feature engineering
        }

    except Exception as e:
        st.error(f"An error occurred while fetching career stats: {e}")
        return None

# Define the get_head_to_head_stats function (reusing the refined version)
def get_head_to_head_stats(player1_id, player2_id, season='2023-24'):
    """
    Fetches and calculates head-to-head statistics between two players.

    Args:
        player1_id (int): The ID of the first player.
        player2_id (int): The ID of the second player.
        season (str): The season to fetch game logs for (e.g., '2023-24').

    Returns:
        tuple: A tuple containing two pandas DataFrames, the head-to-head averages
               for player 1 and player 2, or (None, None) if data fetching fails
               or they didn't play against each other in the specified season.
    """
    stats_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

    try:
        # Fetch game logs for player 1
        player1_logs = playergamelogs.PlayerGameLogs(player_id_nullable=player1_id, season_nullable=season).get_data_frames()[0]
        player1_logs['GAME_DATE'] = pd.to_datetime(player1_logs['GAME_DATE'])

        # Fetch game logs for player 2
        player2_logs = playergamelogs.PlayerGameLogs(player_id_nullable=player2_id, season_nullable=season).get_data_frames()[0]
        player2_logs['GAME_DATE'] = pd.to_datetime(player2_logs['GAME_DATE'])

        # Identify games where they played against each other
        h2h_games = pd.merge(player1_logs, player2_logs, on='GAME_ID', suffixes=('_player1', '_player2'))

        # Extract opponent team abbreviation from the 'MATCHUP' column
        def extract_opponent_team(matchup):
            match = re.search(r'@\s*([A-Z]{3})|vs.\s*([A-Z]{3})', matchup)
            if match:
                return match.group(1) or match.group(2)
            return None

        h2h_games['OPP_TEAM_ABBREVIATION_player1_extracted'] = h2h_games['MATCHUP_player1'].apply(extract_opponent_team)
        h2h_games['OPP_TEAM_ABBREVIATION_player2_extracted'] = h2h_games['MATCHUP_player2'].apply(extract_opponent_team)


        # Filter to ensure the matchups were against each other's teams
        h2h_games_filtered = h2h_games[
            (h2h_games['TEAM_ABBREVIATION_player1'] == h2h_games['OPP_TEAM_ABBREVIATION_player2_extracted']) &
            (h2h_games['TEAM_ABBREVIATION_player2'] == h2h_games['OPP_TEAM_ABBREVIATION_player1_extracted'])
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        if h2h_games_filtered.empty:
            return None, None

        # Calculate head-to-head averages for each player
        # Ensure stats_columns only contains columns present in h2h_games_filtered
        valid_h2h_stats_columns_player1 = [f'{col}_player1' for col in stats_columns if f'{col}_player1' in h2h_games_filtered.columns]
        player1_h2h_avg = h2h_games_filtered[valid_h2h_stats_columns_player1].mean()
        player1_h2h_avg.index = [col.replace('_player1', '') for col in valid_h2h_stats_columns_player1] # Rename index to match original stats columns
        player1_h2h_avg = player1_h2h_avg.to_frame(name='Head-to-Head Avg').T # Convert to DataFrame

        valid_h2h_stats_columns_player2 = [f'{col}_player2' for col in stats_columns if f'{col}_player2' in h2h_games_filtered.columns]
        player2_h2h_avg = h2h_games_filtered[valid_h2h_stats_columns_player2].mean()
        player2_h2h_avg.index = [col.replace('_player2', '') for col in valid_h2h_stats_columns_player2] # Rename index to match original stats columns
        player2_h2h_avg = player2_h2h_avg.to_frame(name='Head-to-Head Avg').T # Convert to DataFrame


        return player1_h2h_avg, player2_h2h_avg

    except Exception as e:
        st.error(f"An error occurred while fetching head-to-head stats: {e}")
        return None, None

# Define the fetch_and_combine_game_data function for prediction features
def fetch_and_combine_game_data(player_id, season):
    """
    Fetches player and team game logs for a given season and combines them
    for feature engineering for prediction.

    Args:
        player_id (int): The ID of the NBA player.
        season (str): The season to fetch data for (e.g., '2023-24').

    Returns:
        DataFrame: A DataFrame containing combined player and opponent team stats
                   for each game the player played in that season, or None if
                   data fetching fails.
    """
    try:
        # Fetch player game logs
        player_logs = playergamelogs.PlayerGameLogs(player_id_nullable=player_id, season_nullable=season).get_data_frames()[0]
        player_logs['GAME_DATE'] = pd.to_datetime(player_logs['GAME_DATE'])

        # Fetch all team game logs for the season to easily access opponent team data
        all_team_logs = teamgamelogs.TeamGameLogs(season_nullable=season).get_data_frames()[0]
        all_team_logs['GAME_DATE'] = pd.to_datetime(all_team_logs['GAME_DATE'])

        # Merge player logs with team logs to get player's team's stats for each game
        player_team_combined = pd.merge(player_logs, all_team_logs, on=['GAME_ID', 'GAME_DATE', 'TEAM_ID'], suffixes=('_player', '_team'))

        # Extract opponent team abbreviation from the 'MATCHUP_player' column
        def extract_opponent_team_abbr(matchup, player_team_abbr):
            match = re.search(r'@\s*([A-Z]{3})|vs.\s*([A-Z]{3})', matchup)
            if match:
                opponent_abbr = match.group(1) or match.group(2)
                # Basic check to ensure the extracted opponent is not the player's team
                if opponent_abbr != player_team_abbr:
                    return opponent_abbr
            return None

        # Apply the function to extract opponent team abbreviation
        player_team_combined['OPPONENT_TEAM_ABBREVIATION'] = player_team_combined.apply(
            lambda row: extract_opponent_team_abbr(row['MATCHUP_player'], row['TEAM_ABBREVIATION_player']), axis=1
        )

        # Get a mapping of team abbreviation to team ID for the season
        team_abbr_to_id = all_team_logs[['TEAM_ABBREVIATION', 'TEAM_ID']].drop_duplicates().set_index('TEAM_ABBREVIATION')['TEAM_ID'].to_dict()

        # Map opponent abbreviation to opponent team ID
        player_team_combined['OPPONENT_TEAM_ID'] = player_team_combined['OPPONENT_TEAM_ABBREVIATION'].map(team_abbr_to_id)

        # Merge with all_team_logs again to get opponent stats, using GAME_ID and OPPONENT_TEAM_ID
        # Need to rename columns in all_team_logs before merging to avoid conflicts
        opponent_cols_mapping = {col: f'{col}_opponent' for col in all_team_logs.columns}
        all_team_logs_renamed = all_team_logs.rename(columns=opponent_cols_mapping)

        # Perform the merge for opponent stats
        combined_data = pd.merge(
            player_team_combined,
            all_team_logs_renamed,
            left_on=['GAME_ID', 'OPPONENT_TEAM_ID'],
            right_on=['GAME_ID_opponent', 'TEAM_ID_opponent'],
            how='left' # Use left join to keep all player's games
        )

        # Drop redundant columns after merge
        combined_data = combined_data.drop(columns=['GAME_ID_opponent', 'TEAM_ID_opponent', 'TEAM_ABBREVIATION_opponent', 'TEAM_NAME_opponent', 'GAME_DATE_opponent', 'MATCHUP_opponent', 'WL_opponent'], errors='ignore')

        # Ensure data is sorted by date for feature engineering
        combined_data['GAME_DATE'] = pd.to_datetime(combined_data['GAME_DATE'])
        combined_data = combined_data.sort_values(by='GAME_DATE').reset_index(drop=True)


        return combined_data

    except Exception as e:
        st.error(f"An error occurred while fetching and combining data for prediction: {e}")
        return None

# Define the feature engineering function
def engineer_features(combined_data):
    """
    Engineers features and defines targets for prediction from combined game data.

    Args:
        combined_data (DataFrame): DataFrame containing combined player and opponent game data.

    Returns:
        tuple: A tuple containing the feature DataFrame (X), target DataFrame (y),
               and the cleaned DataFrame with features and targets.
               Returns (None, None, None) if data is insufficient.
    """
    if combined_data is None or combined_data.empty:
        return None, None, None

    stats_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    prediction_targets = ['PTS', 'AST', 'REB', 'FG3M'] # Specific targets for prediction

    # Calculate rolling averages for player stats
    for col in stats_columns:
        # Use a helper function to handle potential NaN values at the start of the season
        # and ensure rolling is done by player
        combined_data[f'{col}_rolling_5'] = combined_data.groupby('PLAYER_ID')[f'{col}_player'].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
        combined_data[f'{col}_rolling_10'] = combined_data.groupby('PLAYER_ID')[f'{col}_player'].transform(lambda x: x.rolling(window=10, min_periods=1).mean().shift(1))
        combined_data[f'{col}_rolling_20'] = combined_data.groupby('PLAYER_ID')[f'{col}_player'].transform(lambda x: x.rolling(window=20, min_periods=1).mean().shift(1))

    # Create a binary feature indicating home or away game
    def is_home_game(matchup):
        # Check if the matchup string contains "vs." (typically indicates a home game)
        return 1 if 'vs.' in matchup else 0

    combined_data['IS_HOME_player'] = combined_data['MATCHUP_player'].apply(is_home_game)

    # Calculate rest days since the last game
    combined_data['PREV_GAME_DATE'] = combined_data.groupby('PLAYER_ID')['GAME_DATE'].shift(1)
    combined_data['REST_DAYS_player'] = (combined_data['GAME_DATE'] - combined_data['PREV_GAME_DATE']).dt.days.fillna(0) # Fill NaN for the first game


    # Select feature columns
    feature_columns = [col for col in combined_data.columns if '_rolling_' in col or '_opponent' in col or col in ['IS_HOME_player', 'REST_DAYS_player']]

    # Define target variables (the actual stats from the current game)
    target_columns = [f'{col}_player' for col in prediction_targets]

    # Create X and y, dropping rows with NaNs created by rolling averages or missing opponent data
    # We also need to shift the target variables up by one row to predict the *next* game's stats
    combined_data_cleaned = combined_data.dropna(subset=feature_columns + target_columns).copy()
    combined_data_cleaned[target_columns] = combined_data_cleaned[target_columns].shift(-1)

    # Drop the last row after shifting targets, as its target will be NaN
    combined_data_cleaned = combined_data_cleaned.iloc[:-1].copy()

    # Select the features (X) and targets (y) from the cleaned data
    X = combined_data_cleaned[feature_columns]
    y = combined_data_cleaned[target_columns]


    if X.empty or y.empty:
        st.warning("Insufficient data after feature engineering to generate predictions.")
        return None, None, None

    return X, y, combined_data_cleaned

# Define a simple prediction function (simplified approach)
def predict_next_game_stats(player_id, season, player_name, combined_data_cleaned):
    """
    Generates a simplified prediction for the next game's stats.
    This uses a weighted average of recent game averages and season average.

    Args:
        player_id (int): The ID of the NBA player.
        season (str): The season for which data was fetched.
        player_name (str): The name of the player.
        combined_data_cleaned (DataFrame): DataFrame with engineered features and targets.

    Returns:
        dict: A dictionary containing predicted stats, or None if prediction is not possible.
    """
    if combined_data_cleaned is None or combined_data_cleaned.empty:
        return None

    # Get the most recent game's data to determine the features for the next game
    # The 'combined_data_cleaned' DataFrame is sorted by date
    last_game_features_row = combined_data_cleaned.iloc[-1]

    # Use a simplified prediction based on weighted averages
    # Weights can be adjusted based on desired emphasis (e.g., more weight to recent games)
    # This is a basic heuristic, not a trained model.
    stats_for_prediction = ['PTS', 'AST', 'REB', 'FG3M']
    predicted_stats = {}

    # Calculate simple season average from the cleaned data for the player
    player_season_avg = combined_data_cleaned[[f'{col}_player' for col in stats_for_prediction]].mean()

    # Calculate recent averages from the raw game logs (if available) - need to refetch or pass
    # Let's use the rolling averages from the last row of combined_data_cleaned as proxies for recent form
    recent_5_avg = last_game_features_row[[f'{col}_rolling_5' for col in stats_for_prediction]]
    recent_10_avg = last_game_features_row[[f'{col}_rolling_10' for col in stats_for_prediction]]
    recent_20_avg = last_game_features_row[[f'{col}_rolling_20' for col in stats_for_prediction]]


    # Apply a simple weighted average. Example weights: 40% last 5, 30% last 10, 20% last 20, 10% season avg
    # Need to handle potential NaN values in rolling averages if the player hasn't played enough games
    # Fill NaN rolling averages with the season average
    recent_5_avg = recent_5_avg.fillna(player_season_avg)
    recent_10_avg = recent_10_avg.fillna(player_season_avg)
    recent_20_avg = recent_20_avg.fillna(player_season_avg)


    for stat in stats_for_prediction:
        stat_col_player = f'{stat}_player'
        stat_col_rolling_5 = f'{stat}_rolling_5'
        stat_col_rolling_10 = f'{stat}_rolling_10'
        stat_col_rolling_20 = f'{stat}_rolling_20'

        # Ensure the rolling average columns exist before accessing
        if stat_col_rolling_5 in recent_5_avg.index and \
           stat_col_rolling_10 in recent_10_avg.index and \
           stat_col_rolling_20 in recent_20_avg.index and \
           stat_col_player in player_season_avg.index:

            predicted_value = (0.4 * recent_5_avg[stat_col_rolling_5] +
                               0.3 * recent_10_avg[stat_col_rolling_10] +
                               0.2 * recent_20_avg[stat_col_rolling_20] +
                               0.1 * player_season_avg[stat_col_player])
            predicted_stats[stat] = round(predicted_value, 2) # Round predictions

        else:
             predicted_stats[stat] = "N/A" # Cannot predict if data is missing


    return predicted_stats


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

# Create a dictionary for player names to IDs
player_name_to_id = {player['full_name']: player['id'] for player in active_players}
player_names = sorted(list(player_name_to_id.keys())) # Sort names alphabetically


# Add dropdowns for selecting players
player1_name_select = st.selectbox("Select Player 1", player_names)
player2_name_select = st.selectbox("Select Player 2 (for Head-to-Head)", [''] + player_names) # Add empty option for no h2h


# Add selectbox for season
# Generate a list of seasons from 2000-01 to current/next season
current_year = pd.Timestamp.now().year
seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2000, current_year + 1)]
selected_season = st.selectbox("Select Season", seasons, index=len(seasons)-1) # Default to the latest season


# Add a button to trigger data fetching
if st.button("Get Player Stats"):
    if not player1_name_select:
        st.warning("Please select Player 1.")
    else:
        player1_id = player_name_to_id.get(player1_name_select)
        if player1_id is None:
             st.error(f"Could not find ID for player: {player1_name_select}")
        else:
            # Call the get_player_stats function for player1
            player1_stats = get_player_stats(player1_id, season=selected_season)

            if player1_stats:
                st.header(f"{player1_name_select} Statistics ({selected_season} Season)")

                # Display career averages
                st.subheader("Career Averages")
                if player1_stats.get('historical_career_averages') is not None and not player1_stats['historical_career_averages'].empty:
                    st.write("Historical Career (excluding current season):")
                    st.dataframe(player1_stats['historical_career_averages'])
                else:
                     st.write("Historical Career Averages Not Available.")


                if player1_stats.get('overall_career_averages') is not None and not player1_stats['overall_career_averages'].empty:
                    st.write("Overall Career (including current season):")
                    st.dataframe(player1_stats['overall_career_averages'])
                else:
                     st.write("Overall Career Averages Not Available.")


                if player1_stats.get('last_season_averages') is not None and not player1_stats['last_season_averages'].empty:
                     st.subheader(f"Last Season Averages")
                     st.dataframe(player1_stats['last_season_averages'])
                else:
                     st.subheader("Last Season Averages Not Available.")


                # Display recent game averages in expanders
                st.subheader("Recent Game Averages")
                if player1_stats.get('last_5_games_avg') is not None and not player1_stats['last_5_games_avg'].empty:
                    with st.expander("Last 5 Games Average"):
                        st.dataframe(player1_stats['last_5_games_avg'])
                else:
                     st.write("Last 5 Games Averages Not Available for the selected season or not enough games played.")


                if player1_stats.get('last_10_games_avg') is not None and not player1_stats['last_10_games_avg'].empty:
                    with st.expander("Last 10 Games Average"):
                        st.dataframe(player1_stats['last_10_games_avg'])
                else:
                     st.write("Last 10 Games Averages Not Available for the selected season or not enough games played.")

                if player1_stats.get('last_20_games_avg') is not None and not player1_stats['last_20_games_avg'].empty:
                    with st.expander("Last 20 Games Average"):
                        st.dataframe(player1_stats['last_20_games_avg'])
                else:
                     st.write("Last 20 Games Averages Not Available for the selected season or not enough games played.")

                # --- Prediction Logic ---
                st.header(f"{player1_name_select} Next Game Prediction ({selected_season} Season)")
                # Fetch and engineer data for prediction
                combined_data = fetch_and_combine_game_data(player1_id, selected_season)
                X, y, combined_data_cleaned = engineer_features(combined_data)

                if combined_data_cleaned is not None and not combined_data_cleaned.empty:
                    # Generate prediction using the simplified approach
                    predicted_stats = predict_next_game_stats(player1_id, selected_season, player1_name_select, combined_data_cleaned)

                    if predicted_stats:
                        st.subheader("Predicted Stats for Next Game:")
                        predicted_df = pd.DataFrame([predicted_stats]) # Convert dict to DataFrame for display
                        st.dataframe(predicted_df)
                    else:
                        st.info("Could not generate prediction for the next game based on available data.")
                else:
                     st.info("Insufficient data to generate prediction for the next game.")


            else:
                st.error(f"Could not fetch stats for {player1_name_select}.")

        # Handle Head-to-Head stats if player2 is selected
        if player2_name_select and player1_name_select != player2_name_select:
            player2_id = player_name_to_id.get(player2_name_select)
            if player2_id is None:
                st.error(f"Could not find ID for player: {player2_name_select}")
            else:
                st.header(f"Head-to-Head Statistics ({player1_name_select} vs {player2_name_select})")

                # Call the get_head_to_head_stats function
                player1_h2h, player2_h2h = get_head_to_head_stats(player1_id, player2_id, season=selected_season)

                if player1_h2h is not None and player2_h2h is not None:
                    st.subheader(f"{player1_name_select} Head-to-Head Averages vs {player2_name_select}")
                    st.dataframe(player1_h2h)

                    st.subheader(f"{player2_name_select} Head-to-Head Averages vs {player1_name_select}")
                    st.dataframe(player2_h2h)
                else:
                    st.info(f"No head-to-head games found between {player1_name_select} and {player2_name_select} in the {selected_season} season.")
        elif player2_name_select and player1_name_select == player2_name_select:
             st.warning("Please select two different players for Head-to-Head comparison.")
        elif player1_name_select and not player2_name_select:
            st.info("Select a Player 2 for Head-to-Head statistics.")
