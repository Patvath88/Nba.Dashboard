import streamlit as st
import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

st.set_page_config(page_title="Projection Tracker", layout="wide")
st.markdown("# 🎯 Projection Tracker")

# Auto-refresh every 5 minutes (300000 ms)
st_autorefresh = st.experimental_data_editor  # placeholder to prevent lint warning
st_autorefresh_interval = 300000  # 5 minutes

# Proper Streamlit auto-refresh
count = st.experimental_data_editor if False else None
st_autorefresh = st.autorefresh(interval=st_autorefresh_interval, key="refresh")

# Manual refresh button
if st.button("🔄 Refresh Now"):
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

# ---------------------- DISPLAY TRACKED PROJECTIONS ----------------------
for player_name, group in data.groupby("player"):
    pid = player_map.get(player_name)
    if not pid:
        continue

    st.markdown(f"## {player_name}")
    latest_proj = group.iloc[-1].to_dict()

    # --- Get last game stats ---
    try:
        gl = playergamelog.PlayerGameLog(player_id=pid, season="2025-26").get_data_frames()[0]
        gl = gl.sort_values("GAME_DATE", ascending=False).iloc[0]
        live_stats = {
            "PTS": gl["PTS"],
            "REB": gl["REB"],
            "AST": gl["AST"],
            "FG3M": gl["FG3M"],
            "STL": gl["STL"],
            "BLK": gl["BLK"],
            "TOV": gl["TOV"],
            "PRA": gl["PTS"] + gl["REB"] + gl["AST"],
        }
    except Exception:
        st.info("Live data unavailable.")
        continue

    cols = st.columns(4)
    i = 0
    for stat, proj_val in latest_proj.items():
        if stat in ["timestamp", "player"]:
            continue
        live_val = live_stats.get(stat, 0)
        hit = "✅" if live_val >= proj_val else "❌"
        with cols[i % 4]:
            st.metric(f"{stat} {hit}", f"{live_val}", f"Proj: {proj_val}")
        i += 1
    st.markdown("---")
