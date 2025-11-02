import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import json
import time

st.set_page_config(page_title="NBA Player Prediction Dashboard", layout="wide")
st.title("🏀 NBA Player Next Game Predictor (NBA Stats API Version)")

# =======================================
# 1. NBA API Setup
# =======================================
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
}

# =======================================
# 2. Helper Functions
# =======================================
@st.cache_data(ttl=3600)
def search_player_id(player_name: str):
    """Search NBA.com API for player ID."""
    url = "https://stats.nba.com/stats/leagueplayerlist"
    params = {
        "Season": "2024-25",
        "SeasonType": "Regular Season",
        "LeagueID": "00"
    }
    try:
        res = requests.get(url, headers=NBA_HEADERS, params=params, timeout=10)
        data = res.json()
        players = pd.DataFrame(data["resultSets"][0]["rowSet"], columns=data["resultSets"][0]["headers"])
        match = players[players["PLAYER_NAME"].str.contains(player_name, case=False, na=False)]
        if not match.empty:
            player = match.iloc[0]
            return int(player["PERSON_ID"]), player["PLAYER_NAME"]
    except Exception as e:
        st.error(f"Error fetching player ID: {e}")
    return None, None


@st.cache_data(ttl=600)
def get_player_game_logs(player_id: int):
    """Fetch last 100 games for the player."""
    url = "https://stats.nba.com/stats/playergamelog"
    params = {
        "PlayerID": player_id,
        "Season": "2024-25",
        "SeasonType": "Regular Season"
    }
    try:
        res = requests.get(url, headers=NBA_HEADERS, params=params, timeout=10)
        data = res.json()
        df = pd.DataFrame(data["resultSets"][0]["rowSet"], columns=data["resultSets"][0]["headers"])
        if df.empty:
            return pd.DataFrame()
        # Convert to numeric
        numeric_cols = ["PTS", "REB", "AST", "FG3M", "MIN"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df.sort_values("GAME_DATE")
    except Exception as e:
        st.error(f"Error fetching player stats: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def get_team_defensive_stats():
    """Pull team defensive stats to adjust predictions."""
    url = "https://stats.nba.com/stats/leaguedashteamstats"
    params = {
        "MeasureType": "Base",
        "PerMode": "PerGame",
        "PlusMinus": "N",
        "PaceAdjust": "N",
        "Rank": "N",
        "Season": "2024-25",
        "SeasonType": "Regular Season",
        "Outcome": "",
        "Location": "",
        "Month": 0,
        "SeasonSegment": "",
        "DateFrom": "",
        "DateTo": "",
        "OpponentTeamID": 0,
        "VsConference": "",
        "VsDivision": "",
        "GameSegment": "",
        "Period": 0,
        "LastNGames": 0,
        "LeagueID": "00"
    }
    res = requests.get(url, headers=NBA_HEADERS, params=params, timeout=10)
    data = res.json()
    df = pd.DataFrame(data["resultSets"][0]["rowSet"], columns=data["resultSets"][0]["headers"])
    return df[["TEAM_ID", "TEAM_NAME", "PTS_ALLOWED", "REB", "AST", "FG3M"]].rename(columns={"PTS_ALLOWED": "PTS_DEF"})

# =======================================
# 3. Regression (NumPy)
# =======================================
def linear_regression_predict(X, y, next_features):
    X = np.array(X)
    y = np.array(y)
    X = np.c_[np.ones(X.shape[0]), X]
    try:
        coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    except np.linalg.LinAlgError:
        coeffs = np.linalg.pinv(X.T @ X) @ X.T @ y
    next_features = np.array([1] + next_features)
    return float(next_features @ coeffs)

# =======================================
# 4. Feature Engineering
# =======================================
def prepare_features(df):
    df["PTS_AVG5"] = df["PTS"].rolling(5, min_periods=1).mean()
    df["REB_AVG5"] = df["REB"].rolling(5, min_periods=1).mean()
    df["AST_AVG5"] = df["AST"].rolling(5, min_periods=1).mean()
    df["FG3M_AVG5"] = df["FG3M"].rolling(5, min_periods=1).mean()
    df["MIN_AVG5"] = df["MIN"].rolling(5, min_periods=1).mean()
    features = ["MIN", "PTS_AVG5", "REB_AVG5", "AST_AVG5", "FG3M_AVG5"]
    df = df.dropna(subset=features)
    return df, features

# =======================================
# 5. Streamlit UI
# =======================================
player_name = st.text_input("Enter player name (e.g. LeBron James):", "LeBron James")

if player_name:
    player_id, full_name = search_player_id(player_name)
    if player_id:
        df = get_player_game_logs(player_id)

        if not df.empty:
            df, features = prepare_features(df)

            st.subheader(f"📊 {full_name} — Recent Games")
            st.dataframe(df[["GAME_DATE", "PTS", "REB", "AST", "FG3M", "MIN"]].tail(10))

            # Sidebar Inputs
            st.sidebar.header("Next Game Inputs")
            next_mins = st.sidebar.slider("Projected Minutes", 25, 45, 35)
            opponent_team = st.sidebar.text_input("Opponent Team Name (e.g. Warriors):", "Warriors")

            # Defensive adjustment
            team_def = get_team_defensive_stats()
            match = team_def[team_def["TEAM_NAME"].str.contains(opponent_team, case=False, na=False)]
            if not match.empty:
                opp = match.iloc[0]
                opp_factor = {
                    "PTS": opp["PTS_DEF"] / team_def["PTS_DEF"].mean(),
                    "REB": opp["REB"] / team_def["REB"].mean(),
                    "AST": opp["AST"] / team_def["AST"].mean(),
                    "FG3M": opp["FG3M"] / team_def["FG3M"].mean(),
                }
            else:
                opp_factor = {"PTS": 1, "REB": 1, "AST": 1, "FG3M": 1}

            last = df.iloc[-1]
            base_features = [
                next_mins,
                last["PTS_AVG5"],
                last["REB_AVG5"],
                last["AST_AVG5"],
                last["FG3M_AVG5"],
            ]

            preds = {}
            for stat in ["PTS", "REB", "AST", "FG3M"]:
                X = df[features].values
                y = df[stat].values
                raw_pred = linear_regression_predict(X, y, base_features)
                preds[stat] = raw_pred * opp_factor[stat]

            # Display predictions
            st.subheader("🎯 Predicted Next Game Stats (Opponent-Adjusted)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Points", f"{preds['PTS']:.1f}")
            col2.metric("Rebounds", f"{preds['REB']:.1f}")
            col3.metric("Assists", f"{preds['AST']:.1f}")
            col4.metric("3PT Made", f"{preds['FG3M']:.1f}")

            # Charts
            st.markdown("### Performance Trends")
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.line(df.tail(10), x="GAME_DATE", y="PTS", title="Points Trend"), use_container_width=True)
            c2.plotly_chart(px.line(df.tail(10), x="GAME_DATE", y="REB", title="Rebounds Trend"), use_container_width=True)
        else:
            st.warning("No game logs available for this player.")
    else:
        st.warning("Player not found in NBA Stats database.")
