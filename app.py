i# This code can be saved as a Python file (e.g., nba_dashboard.py) and run with streamlit run nba_dashboard.py

import streamlit as st
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats, playergamelogs
import pandas as pd
import re # Import regex for parsing matchup string

# Define the get_player_stats function
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
        # Filter out the row for the current season for historical career averages
        # Need to handle cases where career_df has only one row (current season)
        historical_career_averages = None
        if len(career_df) > 1:
            historical_career_df = career_df.iloc[:-1]
            total_games = historical_career_df['GP'].sum()
            stats_columns = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
            career_totals = historical_career_df[stats_columns].sum()
            historical_career_averages = career_totals / total_games
            historical_career_averages = historical_career_averages.to_frame(name='Historical Career Avg').T # Convert to DataFrame for display

        # Calculate overall career averages
        overall_total_games = career_df['GP'].sum()
        overall_career_totals = career_df[stats_columns].sum()
            # Ensure stats_columns only contains columns present in overall_career_totals
        valid_stats_columns = [col for col in stats_columns if col in overall_career_totals.index]
        overall_career_averages = overall_career_totals[valid_stats_columns] / overall_total_games
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
                 last_season_averages = last_season_df[stats_columns]
                 last_season_averages = last_season_averages.to_frame(name=f"Last Season ({last_season_df['SEASON_ID']}) Avg").T # Convert to DataFrame for display
            elif season_index == 0 and len(career_df) > 1:
                 # If selected season is the earliest in the career_df, take the next one if exists (unlikely scenario for 'last' season)
                 pass # Or handle as appropriate, perhaps indicate no prior season data
        else:
             st.warning(f"Selected season {season} not found in career history. Cannot determine last season's averages relative to this season.")


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
            'last_20_games_avg': last_20_games_avg
        }

    except Exception as e:
        st.error(f"An error occurred while fetching career stats: {e}")
        return None

# Define the get_head_to_head_stats function
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
