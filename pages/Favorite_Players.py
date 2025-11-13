import streamlit as st
import pandas as pd
import requests
import datetime
import time
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from PIL import Image
from io import BytesIO

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="⭐ Favorite Players Live Tracker", layout="wide")
st.title("⭐ Favorite Players — Live Tracker")

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
def get_live_games():
    """Get ESPN live scoreboard data."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    try:
        r = requests.get(url, timeout=10)
        return r.json().get("events", [])
    except Exception:
        return []

@st.cache_data(ttl=3600)
def get_player_projections():
    """Pull projections data (same CSV used in research/upcoming projections)."""
    try:
        return pd.read_csv("data/results_history.csv")
    except Exception:
        return pd.DataFrame()

def get_player_photo(pid):
    """Get official NBA player photo."""
    urls = [
        f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png",
        f"https://stats.nba.com/media/players/headshot/{pid}.png"
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                return Image.open(BytesIO(resp.content))
        except Exception:
            continue
    return None

def get_player_stats(pid):
    """Fetch last game, last 5 avg, and season avg."""
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

def get_live_opponent_and_side(player_name):
    """Find opponent and whether home or away from ESPN live data."""
    games = get_live_games()
    for g in games:
        for comp in g.get("competitions", []):
            competitors = comp.get("competitors", [])
            for team in competitors:
                for leader_group in team.get("leaders", []):
                    for leader in leader_group.get("leaders", []):
                        athlete = leader.get("athlete", {})
                        if athlete.get("displayName", "").lower() == player_name.lower():
                            opp = [c for c in competitors if c != team][0]
                            return opp["team"]["displayName"], team["homeAway"].capitalize()
    return None, None

# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
projection_df = get_player_projections()
player_map = get_player_id_map()

if projection_df.empty:
    st.warning("⚠️ No projection data found. Please ensure `results_history.csv` is available.")
    st.stop()

player_names = sorted(projection_df["PLAYER"].unique())
selected_players = st.multiselect("Select Favorite Players", player_names, default=player_names[:3])

# ------------------------------------------------------
# DISPLAY PLAYER SECTIONS
# ------------------------------------------------------
if selected_players:
    for player_name in selected_players:
        pid = player_map.get(player_name)
        if not pid:
            st.warning(f"Player not found: {player_name}")
            continue

        # --- Player Header ---
        st.markdown(f"<h2 style='color:#FF3B3B;text-shadow:0 0 8px #0066FF;'>{player_name}</h2>", unsafe_allow_html=True)
        photo = get_player_photo(pid)
        opp_team, side = get_live_opponent_and_side(player_name)

        col1, col2 = st.columns([1, 3])
        with col1:
            if photo:
                st.image(photo, width=180)
            else:
                st.image("https://cdn-icons-png.flaticon.com/512/847/847969.png", width=180)
            if opp_team:
                st.caption(f"🆚 **{opp_team}** ({side})")
            else:
                st.caption("🕒 No game in progress")

        # --- Pull stats & projections ---
        proj = projection_df[projection_df["PLAYER"] == player_name].iloc[-1]
        stat_summary = get_player_stats(pid)
        metrics = ["PTS", "REB", "AST", "FG3M", "BLK", "STL"]

        cols = st.columns(6)
        for i, stat in enumerate(metrics):
            proj_val = proj.get(stat, 0)
            live_val = stat_summary[stat]["Last"]
            with cols[i]:
                st.markdown(
                    f"""
                    <div style="border:1px solid #0066FF;border-radius:10px;background:#0B0B0B;
                                text-align:center;padding:10px;margin-bottom:5px;
                                box-shadow:0 0 15px #0066FF55;">
                        <b style='color:#FF3B3B;font-size:1.2rem'>{stat}</b><br>
                        <span style='color:#EAEAEA;font-size:1rem;'>Proj: {proj_val:.1f}</span><br>
                        <span style='color:#66B3FF;font-size:1.2rem;'>Live: {live_val:.1f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # --- Historical Data Rows ---
        sub_cols = st.columns(6)
        for i, stat in enumerate(metrics):
            with sub_cols[i]:
                st.caption(
                    f"🕐 Last: {stat_summary[stat]['Last']:.1f} | 📊 L5: {stat_summary[stat]['L5']:.1f} | 🔁 Season: {stat_summary[stat]['Season']:.1f}"
                )

        st.markdown("<hr style='border:1px solid #222;'>", unsafe_allow_html=True)
else:
    st.info("👆 Select your favorite players to start tracking their live performance.")

st.markdown("---")
st.caption("⚡ Hot Shot Props — Favorite Players Live Tracker © 2025")
