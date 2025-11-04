import streamlit as st
import pandas as pd
import time
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

st.set_page_config(page_title="🎯 Projection Tracker", layout="wide")
st.title("🏀 Live NBA Projection Tracker")

# --- Auto-refresh logic ---
REFRESH_INTERVAL = 300  # seconds
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()

if time.time() - st.session_state["last_refresh"] > REFRESH_INTERVAL:
    st.session_state["last_refresh"] = time.time()
    st.rerun()

st.caption(f"🔄 Auto-refresh every 5 minutes | Last updated: {time.strftime('%H:%M:%S')}")
if st.button("🔁 Manual Refresh Now"):
    st.session_state["last_refresh"] = time.time()
    st.rerun()

# --- Load saved projections ---
path = "saved_projections.csv"
try:
    data = pd.read_csv(path)
except FileNotFoundError:
    st.info("No saved projections found. Go to the **AI Player Page** to create them.")
    st.stop()

if data.empty:
    st.info("No projections available yet.")
    st.stop()

nba_players = players.get_active_players()
player_map = {p["full_name"]: p["id"] for p in nba_players}

@st.cache_data(ttl=120)
def get_latest_game_stats(pid):
    """Fetch most recent game log for a player."""
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
            "P+R": gl["PTS"] + gl["REB"],
            "P+A": gl["PTS"] + gl["AST"],
            "R+A": gl["REB"] + gl["AST"],
        }
        return live_stats
    except Exception:
        return None

# --- Display players from saved projections ---
for player_name, group in data.groupby("player"):
    pid = player_map.get(player_name)
    if not pid:
        continue

    st.subheader(f"📊 {player_name}")

    # Get the latest saved projection
    latest_proj = group.iloc[-1].to_dict()
    live_stats = get_latest_game_stats(pid)

    if not live_stats:
        st.warning("Live or recent game data unavailable.")
        continue

    # --- Compare live vs projection ---
    cols = st.columns(4)
    for i, (stat, proj_val) in enumerate(latest_proj.items()):
        if stat in ["timestamp", "player"]:
            continue
        live_val = live_stats.get(stat, 0)
        delta = round(live_val - proj_val, 1)
        hit = "✅" if live_val >= proj_val else "❌"
        with cols[i % 4]:
            st.metric(f"{stat} {hit}", live_val, f"Proj: {proj_val} ({'+' if delta >= 0 else ''}{delta})")

    st.markdown("---")

st.success("✅ Dashboard updated successfully — synced with AI projections.")
