import streamlit as st
import pandas as pd
import requests
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.static import players
from PIL import Image
from io import BytesIO
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Projection Tracker", layout="wide")

st.markdown("""
<style>
body { background-color: black; color: white; }
.metric-card {
    background-color: #1e1e1e;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
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

st.markdown("# 🎯 Projection Tracker")

# ---------------------- IMAGE FETCH ----------------------
@st.cache_data(show_spinner=False)
def get_player_photo(name):
    try:
        player = next((p for p in players.get_active_players() if p["full_name"] == name), None)
        if not player:
            return None
        pid = player["id"]
        urls = [
            f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png",
            f"https://stats.nba.com/media/players/headshot/{pid}.png"
        ]
        for url in urls:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                return Image.open(BytesIO(resp.content))
    except Exception:
        return None
    return None

# ---------------------- GET TEAM CODE ----------------------
@st.cache_data(show_spinner=False)
def get_team_tricode(player_id):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        return info["TEAM_ABBREVIATION"].iloc[0]
    except Exception:
        return None

# ---------------------- FETCH NEXT GAME ----------------------
@st.cache_data(ttl=600, show_spinner=False)
def get_next_game_info(team_tricode):
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
                            "game_id": g["gameId"],
                            "game_date": game_date,
                            "opponent": opp,
                            "location": loc
                        }
        return None
    except Exception:
        return None

# ---------------------- FETCH LIVE OR FINAL BOX SCORE ----------------------
def get_live_stats(game_id, player_name):
    """Fetch live or final stats for the player's next game."""
    try:
        url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "game" not in data:
            return None, "not_started"
        players_data = data["game"]["players"]
        for p in players_data:
            full_name = f"{p['firstName']} {p['familyName']}"
            if player_name.lower() == full_name.lower():
                stats = {
                    "PTS": int(p["statistics"].get("points", 0)),
                    "REB": int(p["statistics"].get("reboundsTotal", 0)),
                    "AST": int(p["statistics"].get("assists", 0)),
                    "FG3M": int(p["statistics"].get("threePointersMade", 0)),
                    "STL": int(p["statistics"].get("steals", 0)),
                    "BLK": int(p["statistics"].get("blocks", 0)),
                    "TOV": int(p["statistics"].get("turnovers", 0)),
                    "PRA": int(p["statistics"].get("points", 0))
                           + int(p["statistics"].get("reboundsTotal", 0))
                           + int(p["statistics"].get("assists", 0))
                }
                status = data["game"]["gameStatusText"].lower()
                if "final" in status:
                    return stats, "final"
                elif "q" in status or "half" in status:
                    return stats, "in_progress"
                else:
                    return stats, "not_started"
        return None, "not_started"
    except Exception:
        return None, "not_started"

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

    if not next_game:
        st.info("No upcoming game found for this player.")
        st.markdown("---")
        continue

    game_date = next_game["game_date"]
    status = "not_started"
    live_stats = {}

    # If game is today or in progress, fetch live data
    if game_date <= datetime.now().date() + timedelta(days=1):
        live_stats, status = get_live_stats(next_game["game_id"], player_name)

    st.markdown(
        f"**Next Game:** {team_tricode} vs {next_game['opponent']} "
        f"({next_game['location']}) — {game_date.strftime('%B %d, %Y')}"
    )

    latest_proj = group.iloc[-1].to_dict()
    stats = [s for s in latest_proj.keys() if s not in ["timestamp", "player"]]

    # Define color logic
    if status == "not_started":
        border_color = "#808080"  # Gray
    elif status == "in_progress":
        border_color = "#FFD700"  # Yellow
    else:
        border_color = "#00FF00"  # Default green/red handled below

    st.markdown("### 📊 AI Prediction vs. Actual Progress")

    cols = st.columns(4)
    for i, stat in enumerate(stats):
        proj_val = latest_proj[stat]
        live_val = live_stats.get(stat, 0)
        if status == "final":
            hit = live_val >= proj_val
            color = "#00FF00" if hit else "#E50914"
        else:
            color = border_color
        with cols[i % 4]:
            st.markdown(
                f"""
                <div class="metric-card" style="border:2px solid {color};color:{color};">
                    <div class="stat-name">{stat}</div>
                    <div class="stat-value">{live_val}</div>
                    <div style="font-size:14px;margin-top:4px;">Proj: {proj_val}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- Add side-by-side bar chart if game started or final ---
    if live_stats:
        proj_vals = [latest_proj[s] for s in stats]
        live_vals = [live_stats.get(s, 0) for s in stats]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stats, y=proj_vals, name="AI Projection", marker_color="#E50914"))
        fig.add_trace(go.Bar(x=stats, y=live_vals, name="Actual", marker_color="#00BFFF"))
        fig.update_layout(
            title=f"{player_name} — AI Projection vs. Actual ({status.replace('_',' ').title()})",
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
