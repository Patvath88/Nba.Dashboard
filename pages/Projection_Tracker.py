import streamlit as st
import pandas as pd
import time
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

st.set_page_config(page_title="🎯 Projection Tracker", layout="wide")
st.title("🏀 Live NBA Projection Tracker")

# --- Auto-refresh every 5 minutes ---
st_autorefresh = st.experimental_data_editor  # placeholder so Streamlit knows it exists
refresh_interval_ms = 300000  # 5 minutes
count = st.experimental_data_editor if False else None

# Streamlit's built-in refresher
from streamlit_autorefresh import st_autorefresh
count = st_autorefresh(interval=refresh_interval_ms, key="refresh_counter")

st.caption(f"🔄 Auto-refreshed every 5 minutes. Last refreshed: {time.strftime('%H:%M:%S')}")

if st.button("🔁 Manual Refresh Now"):
    st.experimental_rerun()

# --- Load projections ---
path = "saved_projections.csv"
try:
    data = pd.read_csv(path)
except FileNotFoundError:
    st.info("No projections saved yet.")
    st.stop()

nba_players = players.get_active_players()
player_map = {p["full_name"]: p["id"] for p in nba_players}

@st.cache_data(ttl=120)
def get_latest_stats(pid):
    """Fetch latest player stats (cached 2 mins)."""
    try:
        gl = playergamelog.PlayerGameLog(player_id=pid, season="2025-26").get_data_frames()[0]
        latest = gl.sort_values("GAME_DATE", ascending=False).iloc[0]
        live_stats = {
            "PTS": latest["PTS"],
            "REB": latest["REB"],
            "AST": latest["AST"],
            "FG3M": latest["FG3M"],
            "STL": latest["STL"],
            "BLK": latest["BLK"],
            "TOV": latest["TOV"],
            "PRA": latest["PTS"] + latest["REB"] + latest["AST"],
        }
        return live_stats
    except Exception:
        return None

# --- Main Loop ---
for player_name, group in data.groupby("player"):
    pid = player_map.get(player_name)
    if not pid:
        continue

    st.subheader(player_name)
    latest_proj = group.iloc[-1].to_dict()

    live_stats = get_latest_stats(pid)
    if not live_stats:
        st.warning("Live data unavailable or game not started.")
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

st.success("✅ Dashboard updated successfully.")
