import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px

st.set_page_config(page_title="NBA Player Prediction Dashboard", layout="wide")
st.title("🏀 NBA Player Next Game Predictor (Opponent-Adjusted Model)")

# =========================
# 1. Safe API Helper Calls
# =========================
@st.cache_data(ttl=3600)
def get_player_id(player_name):
    url = f"https://www.balldontlie.io/api/v1/players?search={player_name}"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        st.error("⚠️ Error fetching player info — please try again.")
        return None, None

    if data.get("data"):
        player = data["data"][0]
        return player["id"], f"{player['first_name']} {player['last_name']}"
    return None, None


@st.cache_data(ttl=300)
def get_player_stats(player_id):
    stats = []
    page = 1
    while True:
        url = f"https://www.balldontlie.io/api/v1/stats?player_ids[]={player_id}&per_page=100&page={page}"
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            data = res.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            st.warning("⚠️ Could not fetch all player stats (API limit or network issue).")
            break

        if not data.get("data"):
            break

        stats.extend(data["data"])
        if not data["meta"]["next_page"]:
            break
        page += 1

    if not stats:
        st.warning("⚠️ No player stats returned from API.")
        return pd.DataFrame()
    return pd.json_normalize(stats)


@st.cache_data(ttl=600)
def get_team_stats():
    """Get average team defense stats for opponent adjustment."""
    teams = {}
    for team_id in range(1, 31):
        url = f"https://www.balldontlie.io/api/v1/stats?team_ids[]={team_id}&per_page=100"
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            data = res.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            continue

        if not data.get("data"):
            continue

        df = pd.json_normalize(data["data"])
        teams[team_id] = {
            "team_name": df["team.full_name"].iloc[0],
            "opp_pts_allowed": df["pts"].mean(),
            "opp_reb_allowed": df["reb"].mean(),
            "opp_ast_allowed": df["ast"].mean(),
            "opp_3pt_allowed": df["fg3m"].mean(),
        }

    return pd.DataFrame.from_dict(teams, orient="index")

# =========================
# 2. Core Regression (NumPy)
# =========================
def linear_regression_predict(X, y, next_features):
    X = np.array(X)
    y = np.array(y)
    X = np.c_[np.ones(X.shape[0]), X]  # add bias
    try:
        coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    except np.linalg.LinAlgError:
        coeffs = np.linalg.pinv(X.T @ X) @ X.T @ y
    next_features = np.array([1] + next_features)
    prediction = next_features @ coeffs
    return float(prediction)

# =========================
# 3. Feature Engineering
# =========================
def prepare_features(df):
    df["min_played"] = df["min"].apply(lambda x: int(x.split(":")[0]) if isinstance(x, str) else 0)
    df["home_game"] = (df["game.home_team_id"] == df["team.id"]).astype(int)
    for stat in ["pts", "reb", "ast", "fg3m"]:
        df[stat] = df[stat].astype(float)
        df[f"{stat}_avg5"] = df[stat].rolling(5, min_periods=1).mean()
    features = ["min_played", "home_game", "pts_avg5", "reb_avg5", "ast_avg5", "fg3m_avg5"]
    df = df.dropna(subset=features)
    return df, features

# =========================
# 4. Streamlit UI
# =========================
player_name = st.text_input("Enter player name (e.g. Luka Doncic):", "Luka Doncic")

if player_name:
    player_id, full_name = get_player_id(player_name)
    if player_id:
        df = get_player_stats(player_id)
        if not df.empty:
            df, features = prepare_features(df)
            df = df.sort_values("game.date")

            st.subheader(f"📈 {full_name} Recent Performance")
            st.dataframe(df[["game.date", "pts", "reb", "ast", "fg3m", "min_played"]].tail(10))

            # Sidebar inputs
            st.sidebar.header("Next Game Inputs")
            next_mins = st.sidebar.slider("Projected Minutes", 25, 45, 35)
            home_game = 1 if st.sidebar.checkbox("Home Game?", value=True) else 0
            opponent_id = st.sidebar.number_input("Opponent Team ID (1–30)", min_value=1, max_value=30, value=10)

            team_def = get_team_stats()
            if opponent_id in team_def.index:
                opp_stats = team_def.loc[opponent_id]
                opp_factor = {
                    "pts": opp_stats["opp_pts_allowed"] / team_def["opp_pts_allowed"].mean(),
                    "reb": opp_stats["opp_reb_allowed"] / team_def["opp_reb_allowed"].mean(),
                    "ast": opp_stats["opp_ast_allowed"] / team_def["opp_ast_allowed"].mean(),
                    "fg3m": opp_stats["opp_3pt_allowed"] / team_def["opp_3pt_allowed"].mean(),
                }
            else:
                opp_factor = {"pts": 1, "reb": 1, "ast": 1, "fg3m": 1}

            last = df.iloc[-1]
            base_features = [
                next_mins, home_game,
                last["pts_avg5"], last["reb_avg5"], last["ast_avg5"], last["fg3m_avg5"]
            ]

            # Run regression + adjustment
            predictions = {}
            for stat in ["pts", "reb", "ast", "fg3m"]:
                X = df[features].values
                y = df[stat].values
                base_pred = linear_regression_predict(X, y, base_features)
                adjusted_pred = base_pred * opp_factor[stat]
                predictions[stat] = adjusted_pred

            st.subheader("🎯 Opponent-Adjusted Next Game Predictions")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Points", f"{predictions['pts']:.1f}")
            col2.metric("Rebounds", f"{predictions['reb']:.1f}")
            col3.metric("Assists", f"{predictions['ast']:.1f}")
            col4.metric("3PT Made", f"{predictions['fg3m']:.1f}")

            st.markdown(f"**Opponent Adjustment:** {team_def.loc[opponent_id]['team_name']}")

            # Trend charts
            st.markdown("### Recent Trends (Last 10 Games)")
            col1, col2 = st.columns(2)
            col1.plotly_chart(px.line(df.tail(10), x="game.date", y="pts", title="Points Trend"), use_container_width=True)
            col2.plotly_chart(px.line(df.tail(10), x="game.date", y="reb", title="Rebounds Trend"), use_container_width=True)
        else:
            st.warning("No game data found for this player.")
    else:
        st.warning("Player not found.")
