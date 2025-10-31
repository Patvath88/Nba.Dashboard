import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playercareerstats, playergamelogs, teamgamelogs
import pandas as pd
import re
import numpy as np
import time # Import time for delays
# import plotly.graph_objects as go # Removed Plotly
# import requests # Removed requests for fetching logo image
from PIL import Image # Reintroduce Pillow for handling logo image
# from io import BytesIO # Removed BytesBytesIO for handling logo image bytes
import altair as alt # Reintroduce Altair


# Define core stats columns globally
stats_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

# Define prediction targets globally
prediction_targets = ['PTS', 'AST', 'REB', 'FG3M']

# Function to get NBA logo image (Fetching code removed, will instruct user on local file)
# @st.cache_data # Cache this data
# def get_nba_logo_image():
#     """
#     Attempts to get the NBA logo image from a common source.
#     Fetching code removed due to reliability issues.
#     """
#     # Example URL for NBA logo (this might need adjustment based on actual sources)
#     # logo_url = "https://upload.wikimedia.wikimedia.org/wikipedia/commons/thumb/b/ba/NBA_Logo.svg/800px-NBA_Logo.svg.png" # Example URL

#     # try:
#     #     response = requests.get(logo_url)
#     #     if response.status_code == 200:
#     #         return Image.open(BytesIO(response.content))
#     #     else:
#     #         st.warning(f"Could not fetch NBA logo image from {logo_url}. Status code: {response.status_code}")
#     #         return None
#     # except Exception as e:
#     #     st.warning(f"Error fetching NBA logo image: {e}")
#     return None # Return None as image fetching is removed


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


# Function to create a stat bar (using Streamlit progress)
def create_stat_bar(label, value, max_value):
    """Creates a simple horizontal bar for visualizing a stat using st.progress."""
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


# Removed Altair charting functions
# def create_stat_bar_chart(data_series, title, max_value):
#    """
#    Creates an Altair bar chart for a single stat from a pandas Series.
#    """
#    # ... (Altair code) ...
#    pass # Functionality removed

# def create_multiple_stat_bar_charts(data_series, stat_columns, max_values):
#    """
#    Creates multiple Altair bar charts for specified stats side-by-side.
#    """
#    # ... (Altair code) ...
#    pass # Functionality removed


# --- Streamlit App Layout and Logic ---

st.set_page_config(layout="wide", page_title="NBA Player Stats Dashboard", initial_sidebar_state="expanded") # Set wide layout, page title, and expanded sidebar

# Add some basic styling using markdown for black and blue theme and refined typography
st.markdown("""
<style>
    body {
        font-family: 'Segoe UI', Roboto, Arial, sans-serif; /* Professional sans-serif font */
    }
    .main {
        background-color: #121212; /* Darker black background */
        color: #e0e0e0; /* Light gray text for readability */
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 48px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a237e; /* Deep blue for inactive tabs */
        color: #e0e0e0; /* Light gray text */
        border-radius: 4px 4px 0px 0px;
        gap: 4px;
        padding-top: 10px;
        padding-bottom: 10px;
        margin: 0;
        cursor: pointer;
        font-weight: bold; /* Make tab text bold */
    }
    .stTabs [aria-selected="true"] {
        background-color: #2962ff; /* Brighter blue for active tab */
        color: white;
    }
    .stMarkdown h1, h2, h3 {
        color: #42a5f5; /* Lighter blue headings */
        font-weight: 600; /* Slightly bolder headings */
    }
    .stDataFrame {
        font-size: 0.95em; /* Slightly larger font for tables */
        color: #212121; /* Dark text for table content */
    }
     .stDataFrame tbody tr {
        background-color: #e3f2fd; /* Light blue for table rows */
     }
     .stDataFrame tbody tr:nth-child(odd) {
        background-color: #bbdefb; /* Slightly darker blue for odd rows */
     }
     .stDataFrame th {
         background-color: #1a237e; /* Deep blue for table headers */
         color: white;
         font-weight: bold; /* Bold table headers */
     }
     .stDataFrame td {
         padding: 8px; /* Add padding to table cells */
     }
    /* Style for stat bars (using st.progress) */
    .stProgress > div > div > div > div {
        background-color: #2962ff; /* Blue color for bars */
    }
    /* Style for text input and selectbox labels */
    .stTextInput label, .stSelectbox label {
        font-weight: bold;
        color: #e0e0e0; /* Light gray color for labels */
    }
    /* Style for buttons */
    .stButton > button {
        background-color: #2962ff;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #42a5f5;
    }
</style>
""", unsafe_allow_html=True)


# Removed logo column and related code
st.title("NBA Player Stats Dashboard")


# --- Sidebar for Player Selection ---
with st.sidebar:
    st.header("Select Player")
    # Get list of active players for dropdowns
    @st.cache_data # Cache this data to avoid refetching every time
    def get_active_nba_players():
        try:
            return players.get_active_players()
        except Exception as e:
            st.error(f"Error fetching active players: {e}")
            return []

    active_players = get_active_nba_players()

    # Get list of all teams for dropdowns (needed for team_name_to_id/abbr_to_id, maybe move this out if not used in sidebar?)
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
    player1_name_select = st.selectbox("Choose a player:", player_names) # Changed label for sidebar context


# --- Main Content Area ---
player1_id = None # Initialize player_id
career_df_all_seasons = None # Initialize to None
career_game_logs_df = None # Initialize career game logs

if player1_name_select:
    player1_id = player_name_to_id.get(player1_name_select)
    if player1_id is not None:
        # Fetch all data for the selected player
        career_df_all_seasons, career_game_logs_df = fetch_player_data(player1_id)

        if career_df_all_seasons is not None and not career_df_all_seasons.empty:
            st.header(f"{player1_name_select} Stats Dashboard") # Player name as header in main area


            # using a single column for stats now
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
                 # overall_career_averages_df = overall_career_averages.to_frame(name='Overall Career Avg').T # Convert to DataFrame for display


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

                 # Dictionary of max values for charting (still useful for scaling progress bars)
                 max_values_dict = {'PTS': max_pts, 'REB': max_reb, 'AST': max_ast, 'MIN': max_min}

                 # Display stat bars for key career averages using st.progress
                 st.write("Overall Career Averages:")
                 create_stat_bar("Points", overall_career_averages.get('PTS'), max_values_dict.get('PTS')) # Use create_stat_bar
                 create_stat_bar("Rebounds", overall_career_averages.get('REB'), max_values_dict.get('REB')) # Use create_stat_bar
                 create_stat_bar("Assists", overall_career_averages.get('AST'), max_values_dict.get('AST')) # Use create_stat_bar
                 create_stat_bar("Minutes", overall_career_averages.get('MIN'), max_values_dict.get('MIN')) # Use create_stat_bar


            else:
                 st.info("Overall Career Averages not available.")


            # Using Tabs for better organization in the main area
            tab1, tab2, tab3, tab4 = st.tabs(["Season Averages", "Recent Games", "Predictions", "Saved Predictions"])

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

                # --- Add Season Trends Visualization (Line Charts) ---
                st.subheader("Season Trends for Key Stats")
                if not display_career_df.empty and 'SEASON_ID' in display_career_df.columns:
                     # Select key stats for trend visualization
                     key_trend_stats = ['PTS', 'AST', 'REB', 'FG3M', 'MIN'] # Added MIN as it's a key stat

                     # Melt the DataFrame to long format for Altair
                     melted_df = display_career_df.melt(
                         id_vars='SEASON_ID',
                         value_vars=[col for col in key_trend_stats if col in display_career_df.columns],
                         var_name='Statistic',
                         value_name='Average per Game'
                     )

                     if not melted_df.empty:
                         # Convert SEASON_ID to string or ordered category for Altair axis
                         melted_df['SEASON_ID'] = melted_df['SEASON_ID'].astype(str)

                         chart = alt.Chart(melted_df).mark_line(point=True).encode(
                             x=alt.X('SEASON_ID', title='Season', sort=None), # Sort by season string
                             y=alt.Y('Average per Game', title='Average per Game'),
                             color='Statistic', # Different line for each statistic
                             tooltip=['SEASON_ID', 'Statistic', alt.Tooltip('Average per Game', format='.2f')]
                         ).properties(
                             title=f'{player1_name_select} Season Trends for Key Stats'
                         ).interactive() # Enable zooming and panning

                         st.altair_chart(chart, use_container_width=True)
                     else:
                         st.info("Data is not in the correct format for plotting season trends.")

                else:
                     st.info("Not enough season data available to plot trends.")


            with tab2: # This is now the Recent Games tab
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

            with tab3: # This is now the Predictions tab
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

            with tab4: # This is now the Saved Predictions tab
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
        st.info("Please select a player from the dropdown in the sidebar to view their stats.")
