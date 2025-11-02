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

# Safe request wrapper with retries
def safe_request(url, params=None, max_retries=8, sleep_seconds=3):
    """Keeps retrying until a valid JSON response is returned."""
    for attempt in range(max_retries):
        try:
            res = requests.get(url, headers=NBA_HEADERS, params=params, timeout=30)
            if res.status_code == 200:
                return res.json()
            else:
                time.sleep(sleep_seconds)
        except requests.exceptions.RequestException:
            time.sleep(sleep_seconds)
    st.error("⏳ NBA API is taking too long to respond. Please retry in a few moments.")
    return None

# =======================================
# 2. Helper Functions
# =======================================
@st.cache_data(ttl=3600)
def search_player_id(player_name: str):
    url = "https://stats.nba.com/stats/leagueplayerlist"
    params = {
        "Season": "2024-25",
        "SeasonType": "Regular Season",
        "LeagueID": "00"
    }
    st.info("🔍 Searching NBA database...")
    data = safe_request(url, params=params)
    if not data:
        return None, None
    players = pd.DataFrame(data["resultSets"][0]["rowSet"], columns=data["resultSets"][0]["headers"])
    match = players[players["PLAYER_NAME"].str.contains(player_name, case=False, na=False)]
    if not match.empty:
        player = match.iloc[0]
        return int(player["PERSON_ID"]), player["PLAYER_NAME"]
    return None, None


@st.cache_data(ttl=600)
def get_player_game_logs(player_id: int):
    url = "https://stats.nba.com/stats/playergamelog"
    params = {
        "PlayerID": player_id,
        "Season": "2024-25",
        "SeasonType": "Regular Season"
    }
    st.info("📊 Fetching player game logs...")
    data = safe_request(url, params=params)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data["resultSets"][0]["rowSet"], columns=data["resultSets"][0]["headers"])
    if df.empty:
        return pd.DataFrame()
    numeric_cols = ["PTS", "REB", "AST", "FG3M", "MIN"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values("GAME_DATE")


@st.cache_data(ttl=600)
def get_team_defensive_stats():
    url = "https://stats.nba.com/stats/leaguedashteamstats"
    params = {
        "MeasureType": "Base",
        "PerMode": "PerGame",
        "Season": "2024-25",
        "SeasonType": "Regular Season",
        "LeagueID": "00"
    }
    st.info("🛡️ Loading team defensive data...")
    data = safe_request(url, params=params)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data["resultSets"][0]["rowSet"], columns=data["resultSets"][0]["headers"])
    cols = ["TEAM_ID", "TEAM_NAME", "PTS", "REB", "AST", "FG3M"]
    df = df[cols].rename(columns={"PTS": "PTS_DEF"})
    return df

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
            st.subheader(f"📈 {full_name} Recent Games")
            st.dataframe(df[["GAME_DATE", "PTS", "REB", "AST", "FG3M", "MIN"]].tail(10))

            st.sidebar.header("Next Game Inputs")
            next_mins = st.sidebar.slider("Projected Minutes", 25, 45, 35)
            opponent_team = st.sidebar.text_input("Opponent Team Name (e.g. Warriors):", "Warriors")

            team_def = get_team_defensive_stats()
            if not team_def.empty:
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
            else:
                opp_factor = {"PTS": 1, "REB": 1, "AST": 1, "FG3M": 1}

            last = df.iloc[-1]
            base_features = [next_mins, last["PTS_AVG5"], last["REB_AVG5"], last["AST_AVG5"], last["FG3M_AVG5"]]

            preds = {}
            for stat in ["PTS", "REB", "AST", "FG3M"]:
                X = df[features].values
                y = df[stat].values
                raw_pred = linear_regression_predict(X, y, base_features)
                preds[stat] = raw_pred * opp_factor[stat]

            st.subheader("🎯 Predicted Next Game Stats (Opponent-Adjusted)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Points", f"{preds['PTS']:.1f}")
            c2.metric("Rebounds", f"{preds['REB']:.1f}")
            c3.metric("Assists", f"{preds['AST']:.1f}")
            c4.metric("3PT Made", f"{preds['FG3M']:.1f}")

            st.markdown("### Trends (Last 10 Games)")
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.line(df.tail(10), x="GAME_DATE", y="PTS", title="Points"), use_container_width=True)
            c2.plotly_chart(px.line(df.tail(10), x="GAME_DATE", y="REB", title="Rebounds"), use_container_width=True)
        else:
            st.warning("No game logs available for this player.")
    else:
        st.warning("Player not found in NBA Stats database.")
