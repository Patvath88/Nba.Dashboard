import streamlit as st
import pandas as pd
import requests
import datetime
import time
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.static import players

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="⭐ Favorite Players Tracker", layout="wide")
st.title("⭐ Favorite Players Live Tracker")

REFRESH_INTERVAL = 60
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
if time.time() - st.session_state["last_refresh"] > REFRESH_INTERVAL:
    st.session_state["last_refresh"] = time.time()
    st.rerun()

st.caption(f"🔄 Auto-refresh every {REFRESH_INTERVAL}s | Last updated: {datetime.datetime.now().strftime('%I:%M:%S %p')}")

# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------
@st.cache_data(ttl=3600)
def get_player_id_map():
    return {p["full_name"]: p["id"] for p in players.get_active_players()}

@st.cache_data(ttl=600)
def get_live_boxscores():
    """Pull live box scores from ESPN feed."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data.get("events", [])
    except Exception:
        return []

@st.cache_data(ttl=3600)
def get_player_projection_data():
    """Load projections dataset (use your research/projection CSV)."""
    try:
        return pd.read_csv("data/results_history.csv")
    except:
        return pd.DataFrame()

def get_player_stats(player_name):
    """Fetch season game log and compute last-game + averages."""
    pid = get_player_id_map().get(player_name)
    if not pid:
        return {}
    logs = playergamelog.PlayerGameLog(player_id=pid, season="2025-26").get_data_frames()[0]
    last_game = logs.iloc[0]
    last_5 = logs.head(5)
    season_avg = logs.mean(numeric_only=True)

    return {
        "PTS": {"Last": last_game["PTS"], "L5": last_5["PTS"].mean(), "Season": season_avg["PTS"]},
        "REB": {"Last": last_game["REB"], "L5": last_5["REB"].mean(), "Season": season_avg["REB"]},
        "AST": {"Last": last_game["AST"], "L5": last_5["AST"].mean(), "Season": season_avg["AST"]},
        "FG3M": {"Last": last_game["FG3M"], "L5": last_5["FG3M"].mean(), "Season": season_avg["FG3M"]},
        "BLK": {"Last": last_game["BLK"], "L5": last_5["BLK"].mean(), "Season": season_avg["BLK"]},
        "STL": {"Last": last_game["STL"], "L5": last_5["STL"].mean(), "Season": season_avg["STL"]}
    }

def get_live_stats(player_name):
    """Scrape live stats for given player from ESPN feed."""
    games = get_live_boxscores()
    for g in games:
        for comp in g.get("competitions", []):
            for team in comp.get("competitors", []):
                for p in team.get("leaders", []):
                    leader = p.get("leaders", [])
                    if not leader:
                        continue
                    for entry in leader:
                        athlete = entry.get("athlete", {})
                        if athlete.get("displayName", "").lower() == player_name.lower():
                            return entry
    return {}

# ------------------------------------------------------
# PLAYER SELECTION
# ------------------------------------------------------
player_id_map = get_player_id_map()
projection_df = get_player_projection_data()

if projection_df.empty:
    st.warning("No projection data found. Please ensure `results_history.csv` is loaded.")
else:
    player_names = sorted(projection_df["PLAYER"].unique())
    selected_players = st.multiselect("Select Favorite Players", player_names, default=player_names[:3])

# ------------------------------------------------------
# DISPLAY LIVE METRIC CARDS
# ------------------------------------------------------
if selected_players:
    for player in selected_players:
        st.markdown(f"<h2 style='color:#FF3B3B;text-shadow:0 0 8px #0066FF;'>{player}</h2>", unsafe_allow_html=True)

        player_stats = get_player_stats(player)
        player_proj = projection_df[projection_df["PLAYER"] == player].iloc[-1] if not projection_df.empty else None

        if not player_proj.empty:
            cols = st.columns(6)
            metrics = ["PTS", "REB", "AST", "FG3M", "BLK", "STL"]

            for i, m in enumerate(metrics):
                live_stat = get_live_stats(player)
                live_value = live_stat.get("value", player_stats[m]["Last"])
                projected = player_proj.get(m, 0)

                with cols[i]:
                    st.metric(label=f"{m}", value=f"{live_value:.1f}", delta=f"Proj: {projected:.1f}")

            # Subtext rows: Last Game / L5 / Season averages
            sub_cols = st.columns(6)
            for i, m in enumerate(metrics):
                with sub_cols[i]:
                    st.caption(
                        f"🕐 Last: {player_stats[m]['Last']:.1f} | 📊 L5: {player_stats[m]['L5']:.1f} | 🔁 Season: {player_stats[m]['Season']:.1f}"
                    )
        else:
            st.warning("No projection data available for this player.")
else:
    st.info("Select players to begin tracking their live stats.")

st.markdown("---")
st.caption("⚡ Hot Shot Props — Favorite Players Live Tracker © 2025")
