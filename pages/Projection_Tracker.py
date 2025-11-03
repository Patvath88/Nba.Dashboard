import streamlit as st
import pandas as pd
import requests
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.static import players
from PIL import Image
from io import BytesIO
import os
from datetime import datetime
import plotly.graph_objects as go

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

# ---------------------- GET PLAYER TEAM ----------------------
@st.cache_data(show_spinner=False)
def get_team_tricode(player_id):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        return info["TEAM_ABBREVIATION"].iloc[0]
    except Exception:
        return None

# ---------------------- FETCH NEXT GAME ----------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_next_game_info(team_tricode):
    """Find next scheduled game for a given team."""
    try:
        sched = requests.get("https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json", timeout=10).json()
        today = datetime.now().date()
        for date in sched["leagueSchedule"]["gameDates"]:
            for g in date["games"]:
                if g["homeTeam"]["teamTricode"] == team_tricode or g["awayTeam"]["teamTricode"] == team_tricode:
                    game_date = datetime.strptime(g["gameDateEst"].split("T")[0], "%Y-%m-%d").date()
                    if game_date >= today:
                        loc = "Home" if g["homeTeam"]["teamTricode"] == team_tricode else "Away"
                        opp = g["awayTeam"]["teamTricode"] if loc == "Home" else g["homeTeam"]["teamTricode"]
                        return {
                            "game_date": game_date.strftime("%B %d, %Y"),
                            "opponent": opp,
                            "location": loc
                        }
        return None
    except Exception:
        return None

# ---------------------- DELETE PROJECTION FUNCTION ----------------------
def delete_projection(player_name):
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
    with col2:
        st.markdown(f"## 🏀 {player_name}")
    with col3:
        if st.button("❌ Delete Projection", key=f"delete_{player_name}"):
            delete_projection(player_name)

    team_tricode = get_team_tricode(pid)
    next_game = get_next_game_info(team_tricode)

    if next_game:
        st.markdown(
            f"**Next Game:** {team_tricode} vs {next_game['opponent']} "
            f"({next_game['location']}) — {next_game['game_date']}"
        )
    else:
        st.info("No upcoming game found for this player.")
        st.markdown("---")
        continue

    # --- Player's saved projection ---
    latest_proj = group.iloc[-1].to_dict()
    stats = [s for s in latest_proj.keys() if s not in ["timestamp", "player"]]

    # --- Get last (or live) stats if game has started ---
    try:
        gl = playergamelog.PlayerGameLog(player_id=pid, season="2025-26").get_data_frames()[0]
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        last_game = gl.sort_values("GAME_DATE", ascending=False).iloc[0]
        live_stats = {
            "PTS": last_game["PTS"],
            "REB": last_game["REB"],
            "AST": last_game["AST"],
            "FG3M": last_game["FG3M"],
            "STL": last_game["STL"],
            "BLK": last_game["BLK"],
            "TOV": last_game["TOV"],
            "PRA": last_game["PTS"] + last_game["REB"] + last_game["AST"],
            "P+R": last_game["PTS"] + last_game["REB"],
            "P+A": last_game["PTS"] + last_game["AST"],
            "R+A": last_game["REB"] + last_game["AST"]
        }
    except Exception:
        live_stats = {}

    # --- Metric cards section ---
    st.markdown("### 📊 AI Prediction vs. Actual (Upcoming Game)")

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

    # --- Add side-by-side bar chart for visualization ---
    try:
        proj_vals = [latest_proj[s] for s in stats]
        live_vals = [live_stats.get(s, 0) for s in stats]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stats, y=proj_vals, name="AI Projection", marker_color="#E50914"))
        fig.add_trace(go.Bar(x=stats, y=live_vals, name="Actual", marker_color="#00BFFF"))
        fig.update_layout(
            title=f"{player_name} — AI Projection vs. Actual",
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    st.markdown("---")
