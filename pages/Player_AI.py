import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
import requests
from io import BytesIO
import os
from datetime import datetime
import datetime as dt

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Player AI", layout="wide")
st.markdown("<style>body{background-color:black;color:white;}</style>", unsafe_allow_html=True)
team_color = "#E50914"
contrast_color = "#00FFFF"

# ---------------------- UTILITIES ----------------------
@st.cache_data(show_spinner=False)
def get_games(player_id, season):
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        gl = gl.sort_values("GAME_DATE")
        return gl
    except Exception:
        return pd.DataFrame()

def enrich(df):
    if df.empty:
        return df
    df["P+R"] = df["PTS"] + df["REB"]
    df["P+A"] = df["PTS"] + df["AST"]
    df["R+A"] = df["REB"] + df["AST"]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

def get_player_photo(name):
    try:
        player = next((p for p in players.get_active_players() if p["full_name"] == name), None)
        if not player:
            return None
        player_id = player["id"]
        urls = [
            f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png",
            f"https://stats.nba.com/media/players/headshot/{player_id}.png"
        ]
        for url in urls:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                return Image.open(BytesIO(resp.content))
    except Exception:
        return None
    return None

# ---------------------- NEXT GAME INFO ----------------------
def get_next_game_info(player_id):
    """Use NBA’s public JSON feed to find next scheduled game."""
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season="2025-26").get_data_frames()[0]
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        team_code = gl.iloc[0]["MATCHUP"].split(" ")[0]

        today = dt.datetime.now().date()
        for offset in range(0, 7):
            date_check = today + dt.timedelta(days=offset)
            date_str = date_check.strftime("%Y-%m-%d")
            url = f"https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_{date_str}.json"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            games = resp.json().get("scoreboard", {}).get("games", [])
            for game in games:
                home = game["homeTeam"]["teamTricode"]
                away = game["awayTeam"]["teamTricode"]
                game_date = pd.to_datetime(game["gameTimeUTC"]).strftime("%Y-%m-%d")
                if home == team_code or away == team_code:
                    matchup = f"{home} vs {away}"
                    return game_date, matchup
        return "", ""
    except Exception:
        return "", ""

# ---------------------- HEADER ----------------------
nba_players = players.get_active_players()
player_list = sorted([p["full_name"] for p in nba_players])
player = st.selectbox("Search or Browse Player", [""] + player_list)
if not player:
    st.warning("Select a player from the dropdown above.")
    st.stop()
pid = next(p["id"] for p in nba_players if p["full_name"] == player)

CURRENT_SEASON = "2025-26"
PREVIOUS_SEASON = "2024-25"
current = enrich(get_games(pid, CURRENT_SEASON))
last = enrich(get_games(pid, PREVIOUS_SEASON))
blended = pd.concat([current, last], ignore_index=True)

# ---------------------- MODEL ----------------------
def train_model(df):
    if df is None or len(df) < 5:
        return None
    X = np.arange(len(df)).reshape(-1, 1)
    y = df.values
    m = RandomForestRegressor(n_estimators=150, random_state=42)
    m.fit(X, y)
    return m

def predict_next(df):
    if df is None or len(df) < 3:
        return 0
    X_pred = np.array([[len(df)]])
    m = train_model(df)
    return round(float(m.predict(X_pred)), 1) if m else 0

pred_next = {}
for stat in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA","P+R","P+A","R+A"]:
    pred_next[stat] = predict_next(current[stat]) if not current.empty else 0

# ---------------------- METRIC CARDS ----------------------
def metric_cards(stats: dict, color: str):
    cols = st.columns(4)
    for i, (key, val) in enumerate(stats.items()):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style="border:2px solid {color};
                            border-radius:10px;
                            background-color:#1e1e1e;
                            padding:12px;
                            text-align:center;
                            color:{color};
                            font-weight:bold;
                            box-shadow:0px 0px 10px {color};">
                    <div style='font-size:22px;margin-bottom:5px;'>{key}</div>
                    <div style='font-size:32px;margin-top:5px;'>{val}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ---------------------- LAYOUT ----------------------
photo = get_player_photo(player)
col1, col2 = st.columns([1, 3])
with col1:
    if photo:
        st.image(photo, width=180)
with col2:
    st.markdown(f"## **{player}**")

st.markdown("## 🧠 AI Predicted Next Game Stats")
metric_cards(pred_next, team_color)

# ---------------------- SAVE PROJECTIONS ----------------------
def save_projection(player_name, projections, game_date, opponent):
    df = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "player": player_name,
        "game_date": game_date,
        "opponent": opponent,
        **projections
    }])
    path = "saved_projections.csv"
    if os.path.exists(path):
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)

if st.button("💾 Save Current AI Projections"):
    next_game_date, next_matchup = get_next_game_info(pid)
    save_projection(player, pred_next, next_game_date, next_matchup)
    st.success(f"{player}'s projections saved for {next_matchup} on {next_game_date}!")
