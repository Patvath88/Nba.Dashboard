import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playercareerstats, playergamelogs, teamgamelogs
import pandas as pd
import re
import numpy as np
import time # Import time for delays
import plotly.graph_objects as go # Import Plotly for visualizations
import requests # Import requests for fetching player images
from PIL import Image # Import Pillow for image handling
from io import BytesIO # Import BytesIO for handling image bytes
import altair as alt # Import Altair for declarative visualizations


# Define core stats columns globally
stats_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

# Define prediction targets globally
prediction_targets = ['PTS', 'AST', 'REB', 'FG3M']

# Function to get player image (attempting to fetch from a common source)
@st.cache_data # Cache this data
def get_player_image_url(player_id):
    """
    Attempts to get a player's image URL from a common source.
    Note: This is not an official NBA API endpoint and may not work for all players or be stable.
    A more robust solution might involve a dedicated image hosting service or a more reliable API.
    """
    # Example URL pattern (this might need adjustment based on actual sources)
    # Common sources: stats.nba.com, basketball-reference.com, etc.
    # Let's try a common pattern for stats.nba.com
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"

# Fetch data for a player (Career Stats and Career Game Logs)
@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_player_data(player_id):
    """
    Fetches a player's career stats and career game logs across all available seasons.

    Args:
        player_id (int): The ID of the NBA player.

    Returns:
        tuple: A tuple containing the career stats DataFrame and career game logs DataFrame,
               or (None, None) if data fetching fails.
    """
    career_df = None
    career_game_logs = None

    try:
        # Fetch career stats summary
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        career_df = career_stats.get_data_frames()[0]

        if career_df.empty:
            st.info("No career data found for this player.")
            return career_df, career_game_logs # Return what we have

        seasons_played = career_df['SEASON_ID'].tolist()
        all_game_logs_list = []

        # Fetch game logs for each season
        for season in seasons_played:
            try:
                # st.write(f"Fetching game logs for {season}...") # Can uncomment for debugging progress
                season_logs = playergamelogs.PlayerGameLogs(player_id_nullable=player_id, season_nullable=season).get_data_frames()[0]
                all_game_logs_list.append(season_logs)
                time.sleep(0.2) # Add a small delay between API calls
            except Exception as e:
                st.warning(f"Could not fetch game logs for season {season}. Error: {e}")
                # Continue to the next season if one season fails

        if all_game_logs_list:
            # Combine game logs from all fetched seasons
            career_game_logs = pd.concat(all_game_logs_list, ignore_index=True)
            career_game_logs['GAME_DATE'] = pd.to_datetime(career_game_logs['GAME_DATE'])
            career_game_logs = career_game_logs.sort_values(by='GAME_DATE').reset_index(drop=True)
        else:
             st.info("No game logs found for this player's career.")


        return career_df, career_game_logs

    except Exception as e:
        st.error(f"An error occurred while fetching player data: {e}")
        return None, None


# Define a function to calculate player vs team career stats for ALL opponents
def get_player_vs_all_teams_career_stats(career_game_logs_df):
    """
    Calculates a player's career averages against all teams they have played against.

    Args:
        career_game_logs_df (DataFrame): DataFrame containing the player's career game logs.

    Returns:
        DataFrame: A pandas DataFrame containing the player's career averages against
                   each opponent team, or None if data is insufficient.
    """
    if career_game_logs_df is None or career_game_logs_df.empty:
        return None

    stats_columns_to_average = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

    try:
        # Extract opponent team abbreviation from the 'MATCHUP' column
        def extract_opponent_team_abbr(matchup):
            # Check if the opponent abbreviation appears after '@' or 'vs.' in the matchup string
            match = re.search(r'@\s*([A-Z]{3})|vs.\s*([A-Z]{3})|VS.\s*([A-Z]{3})', matchup, re.IGNORECASE)
            if match:
                 return (match.group(1) or match.group(2) or match.group(3)).upper()
            return None

        career_game_logs_df['OPPONENT_TEAM_ABBREVIATION'] = career_game_logs_df['MATCHUP'].apply(extract_opponent_team_abbr)

        # Filter out rows where opponent team could not be extracted
        career_game_logs_df_filtered = career_game_logs_df.dropna(subset=['OPPONENT_TEAM_ABBREVIATION']).copy()

        if career_game_logs_df_filtered.empty:
             st.info("Could not determine opponent teams from game logs.")
             return None

        # Group by opponent team and calculate averages
        player_vs_all_teams_career_avg = career_game_logs_df_filtered.groupby('OPPONENT_TEAM_ABBREVIATION')[stats_columns_to_average].mean()

        # Optionally, count games played against each team
        games_played_vs_team = career_game_logs_df_filtered.groupby('OPPONENT_TEAM_ABBREVIATION').size().to_frame(name='GP')
        player_vs_all_teams_career_avg = player_vs_all_teams_career_avg.join(games_played_vs_team)

        # Reorder columns to have GP first
        cols = ['GP'] + [col for col in player_vs_all_teams_career_avg.columns if col != 'GP']
        player_vs_all_teams_career_avg = player_vs_all_teams_career_avg[cols]


        return player_vs_all_teams_career_avg

    except Exception as e:
        st.error(f"An error occurred while calculating player vs all teams career stats: {e}")
        return None


# Define a function to calculate recent game averages
def calculate_recent_game_averages(career_game_logs_df):
    """
    Calculates recent game averages from career game logs.

    Args:
        career_game_logs_df (DataFrame): DataFrame containing the player's career game logs.

    Returns:
        dict: A dictionary containing DataFrames for last 5, 10, and 20 game averages and individual last 5 games.
    """
    recent_averages = {}
    if career_game_logs_df is None or career_game_logs_df.empty:
        return recent_averages

    # Ensure game logs are sorted by date in descending order (most recent first)
    career_game_logs_df = career_game_logs_df.sort_values(by='GAME_DATE', ascending=False)

    valid_game_stats_columns = [col for col in stats_columns if col in career_game_logs_df.columns]

    if len(career_game_logs_df) >= 5:
        recent_averages['last_5_games_avg'] = career_game_logs_df.head(5)[valid_game_stats_columns].mean().to_frame(name='Last 5 Games Avg').T
    if len(career_game_logs_df) >= 10:
        recent_averages['last_10_games_avg'] = career_game_logs_df.head(10)[valid_game_stats_columns].mean().to_frame(name='Last 10 Games Avg').T
    if len(career_game_logs_df) >= 20:
        recent_averages['last_20_games_avg'] = career_game_logs_df.head(20)[valid_game_stats_columns].mean().to_frame(name='Last 20 Games Avg').T

    # Get individual last 5 games
    if len(career_game_logs_df) >= 5:
         cols_to_display_individual = ['GAME_DATE', 'MATCHUP', 'SEASON_YEAR'] + [col for col in stats_columns if col in career_game_logs_df.columns]
         recent_averages['last_5_games_individual'] = career_game_logs_df.head(5)[cols_to_display_individual].copy()
         recent_averages['last_5_games_individual']['GAME_DATE'] = recent_averages['last_5_games_individual']['GAME_DATE'].dt.strftime('%Y-%m-%d')


    return recent_averages

# Define a simple prediction function (simplified approach)
def predict_next_game_stats(career_game_logs_df, latest_season):
    """
    Generates a simplified prediction for the next game's stats using combined data.
    This uses a weighted average of recent game averages and season average from the
    combined data across seasons.

    Args:
        career_game_logs_df (DataFrame): DataFrame containing combined player
                                                  and opponent game data across seasons.
        latest_season (str): The latest season played by the player.

    Returns:
        dict: A dictionary containing predicted stats, or None if prediction is not possible.
    """
    if career_game_logs_df is None or career_game_logs_df.empty:
        st.info("No career game logs available to generate prediction.")
        return None

    # Ensure data is sorted by date before calculating recent averages
    combined_data_for_prediction = career_game_logs_df.sort_values(by='GAME_DATE', ascending=True).reset_index(drop=True)


    # Calculate rolling averages from the combined player's game logs (across seasons)
    valid_stats_columns_pred = [col for col in stats_columns if col in combined_data_for_prediction.columns]

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


            player_features_for_pred[f'{col}_rolling_5'] = rolling_5
            player_features_for_pred[f'{col}_rolling_10'] = rolling_10
            player_features_for_pred[f'{col}_rolling_20'] = rolling_20

        # Calculate season average for the player from the game logs (using the primary selected season)
        # Filter game logs for the primary selected season (latest season for simplicity)
        current_season_game_logs_for_avg = combined_data_for_prediction[combined_data_for_prediction['SEASON_YEAR'] == latest_season]
        player_season_avg_pred = current_season_game_logs_for_avg[valid_stats_columns_pred].mean() if not current_season_game_logs_for_avg.empty else combined_data_for_prediction[valid_stats_columns_pred].mean() # Fallback to overall average if current season data is empty


        # Generate prediction using the simplified approach with calculated features
        predicted_stats = {}
        stats_for_prediction = ['PTS', 'AST', 'REB', 'FG3M'] # Use global prediction targets

        for stat in stats_for_prediction:
            stat_col = stat
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



    if predicted_stats:
        return predicted_stats
    else:
        st.info("Could not generate prediction based on available data.")
        return None


# Function to create a stat bar
def create_stat_bar(label, value, max_value):
    """Creates a simple horizontal bar for visualizing a stat."""
    # Ensure value and max_value are not None or NaN and max_value is positive
    if value is None or pd.isna(value) or max_value is None or pd.isna(max_value) or max_value <= 0:
        display_value = value if value is not None and pd.notna(value) else "N/A"
        st.write(f"**{label}:** {display_value}")
        st.progress(0.0) # Display empty progress bar if data is invalid or zero max
        return

    percentage = (value / max_value) * 100
    # Clamp percentage between 0 and 100
    percentage = max(0, min(100, percentage))

    st.write(f"**{label}:** {value:.2f}")
    st.progress(percentage / 100.0) # Streamlit progress bar expects value between 0.0 and 1.0


# --- Streamlit App Layout and Logic ---

st.set_page_config(layout="wide", page_title="NBA Player Stats Dashboard", initial_sidebar_state="expanded") # Set wide layout, page title, and expanded sidebar

# Add some basic styling using markdown
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 48px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e6e6e6;
        border-radius: 4px 4px 0px 0px;
        gap: 4px;
        padding-top: 10px;
        padding-bottom: 10px;
        margin: 0;
        cursor: pointer;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }
    .stMarkdown h1, h2, h3 {
        color: #333333;
    }
    .stDataFrame {
        font-size: 0.9em;
    }
    /* Style for stat bars */
    .stProgress > div > div > div > div {
        background-color: #007bff; /* Blue color for bars */
    }
</style>
""", unsafe_allow_html=True)


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

player1_id = None # Initialize player_id
career_df_all_seasons = None # Initialize to None
career_game_logs_df = None # Initialize career game logs

if player1_name_select:
    player1_id = player_name_to_id.get(player1_name_select)
    if player1_id is not None:
        # Fetch all data for the selected player
        career_df_all_seasons, career_game_logs_df = fetch_player_data(player1_id)

        if career_df_all_seasons is not None and not career_df_all_seasons.empty:
            st.header(f"{player1_name_select} Stats Dashboard")

            # Layout with columns for image and key stats
            col1, col2 = st.columns([1, 2]) # Adjust column widths as needed

            with col1:
                 # --- Player Image ---
                 player_image_url = get_player_image_url(player1_id)
                 try:
                     response = requests.get(player_image_url)
                     if response.status_code == 200:
                         image = Image.open(BytesIO(response.content))
                         st.image(image, caption=player1_name_select, use_column_width=True) # Use column width
                     else:
                         st.info("Could not fetch player image.")
                 except Exception as e:
                     st.info(f"Error fetching player image: {e}")

            with col2:
                 st.subheader("Career Summary")
                 # Calculate overall career averages from career_df_all_seasons
                 overall_total_games = career_df_all_seasons['GP'].sum()
                 overall_career_totals = career_df_all_seasons[stats_columns].sum()
                 # Ensure stats_columns only contains columns present in overall_career_totals
                 valid_stats_columns_overall = [col for col in stats_columns if col in overall_career_totals.index]

                 # Calculate overall career averages per game
                 overall_career_averages = None
                 if overall_total_games > 0:
                      overall_career_averages = overall_career_totals[valid_stats_columns_overall] / overall_total_games
                      overall_career_averages_df = overall_career_averages.to_frame(name='Overall Career Avg').T # Convert to DataFrame for display


                      # Define max values for stat bars (can be based on league averages, player's best season, etc.)
                      # Let's use a simple heuristic: max of player's career high for key stats + a buffer
                      # Ensure career_game_logs_df is not None and has the column before calculating max
                      max_pts = career_game_logs_df['PTS'].max() if career_game_logs_df is not None and 'PTS' in career_game_logs_df.columns and not career_game_logs_df['PTS'].empty else 30.0
                      max_reb = career_game_logs_df['REB'].max() if career_game_logs_df is not None and 'REB' in career_game_logs_df.columns and not career_game_logs_df['REB'].empty else 15.0
                      max_ast = career_game_logs_df['AST'].max() if career_game_logs_df is not None and 'AST' in career_game_logs_df.columns and not career_game_logs_df['AST'].empty else 10.0
                      max_min = career_game_logs_df['MIN'].max() if career_game_logs_df is not None and 'MIN' in career_game_logs_df.columns and not career_game_logs_df['MIN'].empty else 40.0

                      # Ensure max values are reasonable if career highs are very low
                      max_pts = max(max_pts, 20.0) # Minimum threshold
                      max_reb = max(max_reb, 10.0)
                      max_ast = max(max_ast, 7.0)
                      max_min = max(max_min, 30.0)


                      # Display stat bars for key career averages
                      st.write("Overall Career Averages:")
                      create_stat_bar("Points", overall_career_avg.get('PTS'), max_pts)
                      create_stat_bar("Rebounds", overall_career_avg.get('REB'), max_reb)
                      create_stat_bar("Assists", overall_career_avg.get('AST'), max_ast)
                      create_stat_bar("Minutes", overall_career_avg.get('MIN'), max_min)

                 else:
                      st.info("Overall Career Averages not available.")


            # Using Tabs for better organization
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Season Averages", "Vs. Opponents", "Recent Games", "Predictions", "Saved Predictions"])

            with tab1:
                st.subheader("Season Averages")
                # Select relevant columns for display and calculate averages per game
                stats_columns_for_display = ['SEASON_ID', 'TEAM_ABBREVIATION', 'GP', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
                valid_cols_for_display = [col for col in stats_columns_for_display if col in career_df_all_seasons.columns]
                display_career_df = career_df_all_seasons[valid_cols_for_display].copy()

                # Calculate averages per game for relevant columns
                for col in ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']:
                     if col in display_career_df.columns and 'GP' in display_career_df.columns:
                         # Avoid division by zero if GP is 0
                         display_career_df[col] = display_career_df.apply(lambda row: round(row[col] / row['GP'], 2) if row['GP'] > 0 else 0, axis=1) # Round averages here


                st.dataframe(display_career_df.set_index('SEASON_ID')) # Display the career dataframe, using Season ID as index for better readability


            with tab2:
                 st.subheader("Career Statistics Against All Opponent Teams")
                 if career_game_logs_df is not None and not career_game_logs_df.empty:
                      player_vs_all_teams_career_avg = get_player_vs_all_teams_career_stats(career_game_logs_df)
                      if player_vs_all_teams_career_avg is not None and not player_vs_all_teams_career_avg.empty:
                           st.dataframe(player_vs_all_teams_career_avg.round(2)) # Round for display
                      else:
                           st.info("Could not calculate career statistics against opponent teams.")
                 else:
                      st.info("Career game logs not available to calculate player vs team stats.")

            with tab3:
                st.subheader("Detailed Statistics")

                # Determine the latest season played by the player
                latest_season_for_detailed_view = career_df_all_seasons['SEASON_ID'].iloc[-1]

                # Calculate Last Season Averages
                last_season_averages = None
                if len(career_df_all_seasons) > 1:
                     last_season_df = career_df_all_seasons.iloc[-2] # Second to last row is the previous season
                     last_season_id = last_season_df['SEASON_ID']
                     last_season_games = last_season_df['GP']
                     if last_season_games > 0:
                         valid_stats_columns_last = [col for col in stats_columns if col in last_season_df.index]
                         last_season_totals = last_season_df[valid_stats_columns_last]
                         last_season_averages = (last_season_totals / last_season_games).to_frame(name=f"Last Season ({last_season_id}) Avg").T
                     else:
                          st.info(f"Player played 0 games in their second to last season. Last season averages not available.")
                if last_season_averages is not None and not last_season_averages.empty:
                     st.write("Last Season Averages:")
                     st.dataframe(last_season_averages.round(2))


                # Calculate and display Recent Game Averages
                recent_averages_data = calculate_recent_game_averages(career_game_logs_df)

                st.write("Recent Game Averages (Across Seasons if Needed):")
                if recent_averages_data.get('last_5_games_avg') is not None and not recent_averages_data['last_5_games_avg'].empty:
                    with st.expander("Last 5 Games Average"):
                        st.dataframe(recent_averages_data['last_5_games_avg'].round(2))
                else:
                     st.write("Last 5 Games Averages Not Available (not enough recent games).")

                if recent_averages_data.get('last_10_games_avg') is not None and not recent_averages_data['last_10_games_avg'].empty:
                    with st.expander("Last 10 Games Average"):
                        st.dataframe(recent_averages_data['last_10_games_avg'].round(2))
                else:
                     st.write("Last 10 Games Averages Not Available (not enough recent games).")

                if recent_averages_data.get('last_20_games_avg') is not None and not recent_averages_data['last_20_games_avg'].empty:
                    with st.expander("Last 20 Games Average"):
                        st.dataframe(recent_averages_data['last_20_games_avg'].round(2))
                else:
                     st.write("Last 20 Games Averages Not Available (not enough recent games).")

                # Display individual last 5 games stats
                st.write("Last 5 Games (Individual Performance - Most Recent):")
                if recent_averages_data.get('last_5_games_individual') is not None and not recent_averages_data['last_5_games_individual'].empty:
                     st.dataframe(recent_averages_data['last_5_games_individual'])
                else:
                     st.write("Individual stats for the last 5 most recent games are not available.")

            with tab4:
                # --- Prediction Logic ---
                st.header(f"{player1_name_select} Next Game Prediction")
                # Use the fetched career_game_logs_df for prediction
                predicted_stats = predict_next_game_stats(career_game_logs_df, latest_season_for_detailed_view)

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
                         prediction_key = f"{player1_name_select} ({latest_season_for_detailed_view})" # Using latest season for key
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

            with tab5:
                 # Display saved predictions
                 if 'saved_predictions' in st.session_state and st.session_state.saved_predictions:
                     st.subheader("Saved Predictions")
                     # Convert saved predictions dictionary to a DataFrame for display
                     saved_predictions_df = pd.DataFrame.from_dict(st.session_state.saved_predictions, orient='index')
                     st.dataframe(saved_predictions_df)

                     if st.button("Clear Saved Predictions"):
                         st.session_state.saved_predictions = {}
                         st.success("Saved predictions cleared.")
                 else:
                     st.info("No predictions saved yet.")


        elif player1_id is not None: # Player selected but no career data found
            st.info(f"No data available for {player1_name_select}.")

    else: # No player selected
        st.info("Please select a player from the dropdown to view their stats.")
