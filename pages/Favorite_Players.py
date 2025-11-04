import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leaguegamefinder
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from io import BytesIO
import requests
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(page_title="⭐ Favorite Players", layout="wide")
st.title("⭐ Favorite Players Tracker")

FAV_PATH = "favorite_players.json"
PROJ_PATH = "saved_projections.csv"

# ---------------- UTILITIES ----------------
def load_favorites():
    if os.path.exists(FAV_PATH):
        with open(FAV_PATH, "r") as f:
            return json.load(f)
    return []

def save_favorites(favs):
    with open(FAV_PATH, "w") as f:
        json.dump(favs, f, indent=2)

def get_player_photo(pid):
    urls = [
        f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png",
        f"https://stats.nba.com/media/players/headshot/{pid}.png"
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                return Image.open(BytesIO(r.content))
        except Exception:
            continue
    return None

# ---------------- NBA HELPERS ----------------
@st.cache_data(ttl=3600)
def get_next_game_info(player_id):
    """Find next scheduled game and opponent using NBA API schedule."""
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season="2025-26").get_data_frames()[0]
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        gl = gl.sort_values("GAME_DATE")

        team_code = gl.iloc[0]["MATCHUP"].split(" ")[0]
        games = leaguegamefinder.LeagueGameFinder(season_nullable="2025-26").get_data_frames()[0]
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
        team_games = games[games["MATCHUP"].str.contains(team_code)]
        future_games = team_games[team_games["GAME_DATE"] > pd.Timestamp.now()]

        if not future_games.empty:
            next_game = future_games.sort_values("GAME_DATE").iloc[0]
            matchup = next_game["MATCHUP"]
            date_str = next_game["GAME_DATE"].strftime("%Y-%m-%d")
            return date_str, matchup
        return "", ""
    except Exception:
        return "", ""

@st.cache_data(ttl=3600)
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

def predict_next(df):
    if df is None or len(df) < 3:
        return 0
    X = np.arange(len(df)).reshape(-1, 1)
    y = df.values
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    return round(float(model.predict([[len(df)]])[0]), 1)

# ---------------- FAVORITE PLAYER MANAGEMENT ----------------
nba_players = players.get_active_players()
player_map = {p["full_name"]: p["id"] for p in nba_players}
player_list = sorted(player_map.keys())

favorites = load_favorites()

col1, col2 = st.columns([2, 1])
with col1:
    new_fav = st.selectbox("Add a player to your favorites:", [""] + player_list)
    if new_fav and st.button("➕ Add Player"):
        if new_fav not in favorites:
            favorites.append(new_fav)
            save_favorites(favorites)
            st.success(f"{new_fav} added to favorites!")
        else:
            st.info("Already in your favorites.")

with col2:
    if st.button("🗑️ Clear All Favorites"):
        save_favorites([])
        st.warning("All favorites cleared.")
        st.stop()

st.markdown("---")

if not favorites:
    st.info("No favorite players yet. Add some above!")
    st.stop()

# ---------------- DISPLAY FAVORITES ----------------
st.subheader("💫 Your Favorite Players")
for fav in favorites:
    pid = player_map.get(fav)
    photo = get_player_photo(pid)
    col_img, col_txt, col_btn = st.columns([1, 3, 1])
    with col_img:
        if photo:
            st.image(photo, width=100)
    with col_txt:
        st.markdown(f"### {fav}")
    with col_btn:
        if st.button(f"❌ Remove {fav}", key=fav):
            favorites.remove(fav)
            save_favorites(favorites)
            st.rerun()

st.markdown("---")
st.info("📡 Auto-updating projections for favorite players...")

# ---------------- AUTO-SAVE DAILY PROJECTIONS ----------------
def save_projection(player_name, projections, game_date, opponent):
    df = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "player": player_name,
        "game_date": game_date,
        "opponent": opponent,
        **projections
    }])
    if os.path.exists(PROJ_PATH):
        existing = pd.read_csv(PROJ_PATH)
        duplicate = existing[
            (existing["player"] == player_name) & (existing["game_date"] == game_date)
        ]
        if not duplicate.empty:
            return  # skip duplicates
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(PROJ_PATH, index=False)

# ---------------- UPDATE PROJECTIONS FOR FAVORITES ----------------
for fav in favorites:
    pid = player_map.get(fav)
    current = enrich(get_games(pid, "2025-26"))
    if current.empty:
        continue

    next_game_date, next_matchup = get_next_game_info(pid)
    if not next_game_date:
        continue

    pred_next = {}
    for stat in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA","P+R","P+A","R+A"]:
        pred_next[stat] = predict_next(current[stat])

    save_projection(fav, pred_next, next_game_date, next_matchup)

st.success("✅ Favorite player projections synced successfully!")
