import streamlit as st
import pandas as pd
import requests
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from PIL import Image
from io import BytesIO
import os
from datetime import datetime

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Projection Tracker", layout="wide")

st.markdown("""
<style>
body { background-color: black; color: white; }
.metric-card {
    background-color: #1e1e1e;
    border: 2px solid #E50914;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    color: #E50914;
    font-weight: bold;
    box-shadow: 0px 0px 10px #E50914;
}
.metric-card div.stat-name {
    font-size: 22px;
    margin-bottom: 5px;
}
.metric-card div.stat-value {
    font-size: 32px;
    margin-top: 5px;
}
.delete-btn {
    background-color: #B00020;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: bold;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

team_color = "#E50914"

st.markdown("# 🎯 Projection Tracker")

# ---------------------- IMAGE FETCH ----------------------
@st.cache_data(show_spinner=False)
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

# ---------------------- NEXT GAME FETCH ----------------------
@st.cache_data(ttl=3600)
def get_next_game_info(player_name):
    """Get next scheduled game (team, opponent, home/away, date)"""
    try:
        sched = requests.get("https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json", timeout=10).json()
        for date in sched["leagueSchedule"]["gameDates"]:
            for g in date["games"]:
                for side in ["homeTeam", "awayTeam"]:
                    if player_name.lower().split(" ")[-1] in g[side]["teamName"].lower():
                        team = g[side]["teamName"]
                        opp = g["awayTeam"]["teamName"] if side == "homeTeam" else g["homeTeam"]["teamName"]
                        loc = "Home" if side == "homeTeam" else "Away"
                        return {
                            "team": team,
                            "opponent": opp,
                            "location": loc,
                            "game_date": date["gameDate"]
                        }
        return None
    except Exception:
        return None

# ---------------------- DELETE PROJECTION FUNCTION ----------------------
def delete_projection(player_name):
    """Remove a player's projection from saved_projections.csv"""
    path = "saved_projections.csv"
    if not os.path.exists(path):
        st.warning("No projections file found.")
        return
    df = pd.read_csv(path)
    df = df[df["player"] != player_name]
    df.to_csv(path, index=False)
    st.success(f"🗑️ Deleted saved projection for {player_name}.")
    st.experimental_rerun()

# ---------------------- LOAD SAVED PROJECTIONS ----------------------
path = "saved_projections.csv"
try:
    data = pd.read_csv(path)
except FileNotFoundError:
    st.info("No projections saved yet.")
    st.stop()

nba_players = players.get_active_players()
player_map = {p["full_name"]: p["id"] for p in nba_players}

# ---------------------- DISPLAY EACH SAVED PROJECTION ----------------------
for player_name, group in data.groupby("player"):
    pid = player_map.get(player_name)
    if not pid:
        continue

    # Header with photo + name + delete
    photo = get_player_photo(player_name)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if photo:
            st.image(photo, width=120)
        else:
            st.markdown("🧍‍♂️")
    with col2:
        st.markdown(f"## 🏀 {player_name}")
    with col3:
        if st.button("❌ Delete Projection", key=f"delete_{player_name}"):
            delete_projection(player_name)

    # Next game info
    next_game = get_next_game_info(player_name)
    if next_game:
        game_date = datetime.strptime(next_game["game_date"].split("T")[0], "%Y-%m-%d").strftime("%B %d, %Y")
        st.markdown(
            f"**Next Game:** {next_game['team']} vs {next_game['opponent']} "
            f"({next_game['location']}) on {game_date}"
        )
    else:
        st.info("No upcoming game found for this player.")
        st.markdown("---")
        continue

    # --- Player's saved projection ---
    latest_proj = group.iloc[-1].to_dict()
    stats = [s for s in latest_proj.keys() if s not in ["timestamp", "player"]]

    # --- Fetch live or upcoming game stats ---
    try:
        gl = playergamelog.PlayerGameLog(player_id=pid, season="2025-26").get_data_frames()[0]
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        upcoming_games = gl[gl["GAME_DATE"] >= pd.Timestamp(datetime.now().date())]
        if upcoming_games.empty:
            st.info("Game has not started yet.")
            st.markdown("---")
            continue
        game = upcoming_games.iloc[0]
        live_stats = {
            "PTS": game["PTS"],
            "REB": game["REB"],
            "AST": game["AST"],
            "FG3M": game["FG3M"],
            "STL": game["STL"],
            "BLK": game["BLK"],
            "TOV": game["TOV"],
            "PRA": game["PTS"] + game["REB"] + game["AST"],
            "P+R": game["PTS"] + game["REB"],
            "P+A": game["PTS"] + game["AST"],
            "R+A": game["REB"] + game["AST"]
        }
    except Exception:
        st.info(f"No current game stats available yet for {player_name}.")
        st.markdown("---")
        continue

    # --- Metric cards section ---
    st.markdown("### 📊 Upcoming Game — AI Projection vs. Actual (Live or Final)")

    cols = st.columns(4)
    for i, stat in enumerate(stats):
        proj_val = latest_proj[stat]
        live_val = live_stats.get(stat, 0)
        hit = live_val >= proj_val
        border_color = "#00FF00" if hit else "#E50914"
        emoji = "✅" if hit else "❌"
        with cols[i % 4]:
            st.markdown(
                f"""
                <div class="metric-card" style="border-color:{border_color};color:{border_color};">
                    <div class="stat-name">{stat} {emoji}</div>
                    <div class="stat-value">{live_val}</div>
                    <div style="font-size:14px;margin-top:4px;">Proj: {proj_val}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")
