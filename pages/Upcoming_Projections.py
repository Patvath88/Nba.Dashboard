import streamlit as st
import pandas as pd
import time
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="🎯 Upcoming Game Projections", layout="wide")
st.title("🏀 Upcoming Game Projections")

# ---------------------- REFRESH ----------------------
REFRESH_INTERVAL = 300
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
if time.time() - st.session_state["last_refresh"] > REFRESH_INTERVAL:
    st.session_state["last_refresh"] = time.time()
    st.rerun()

st.caption(f"🔄 Auto-refresh every 5 minutes | Last updated {time.strftime('%H:%M:%S')}")
if st.button("🔁 Manual Refresh Now"):
    st.session_state["last_refresh"] = time.time()
    st.rerun()

# ---------------------- LOAD DATA ----------------------
path = "saved_projections.csv"
try:
    data = pd.read_csv(path)
except FileNotFoundError:
    st.info("No saved projections yet.")
    st.stop()

if data.empty:
    st.info("No projection data available.")
    st.stop()

nba_players = players.get_active_players()
player_map = {p["full_name"]: p["id"] for p in nba_players}

# ---------------------- HELPERS ----------------------
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

# ---------------------- DISPLAY UPCOMING ----------------------
today = pd.Timestamp.now().normalize()
upcoming_games = []

for _, row in data.iterrows():
    proj_date = pd.to_datetime(row.get("game_date"), errors="coerce")
    if pd.isna(proj_date) or proj_date >= today:
        upcoming_games.append(row)

if not upcoming_games:
    st.info("No upcoming games with saved projections.")
    st.stop()

df_upcoming = pd.DataFrame(upcoming_games)

for player_name, group in df_upcoming.groupby("player"):
    pid = player_map.get(player_name)
    if not pid:
        continue

    latest_proj = group.iloc[-1].to_dict()
    game_date = latest_proj.get("game_date", "")
    opponent = latest_proj.get("opponent", "")

    st.markdown("---")
    col_photo, col_info = st.columns([1, 3])
    with col_photo:
        photo = get_player_photo(pid)
        if photo:
            st.image(photo, width=180)
    with col_info:
        st.subheader(player_name)
        st.caption(f"📅 **Game Date:** {game_date or 'TBD'} | 🆚 **Opponent:** {opponent or 'TBD'}")

    compare_stats = ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]
    cols = st.columns(4)
    for i, stat in enumerate(compare_stats):
        val = latest_proj.get(stat, 0)
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style="border:1px solid #00FFFF;
                            border-radius:12px;
                            background:#111;
                            padding:10px;
                            text-align:center;
                            box-shadow:0 0 15px #00FFFF55;
                            margin-bottom:10px;">
                    <b>{stat}</b><br>
                    <span style='color:#00FFFF'>Proj: {val}</span><br>
                    <small>Pending ⏳</small>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.info("🕒 Upcoming game — awaiting actual stats after tip-off.")
