# app.py — Hot Shot Props | NBA Player Analytics Dashboard
# Clean / fast: League Leaders home, player page with metrics/charts,
# deep-linking via ?player_id=..., robust headers+retries to avoid timeouts.

import os, json, time, math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from nba_api.stats.static import players
from nba_api.stats.endpoints import leagueleaders, playercareerstats, playergamelogs

# ---- CRITICAL: make nba_api requests reliable (headers + retries) ----
# nba.com blocks generic UAs; we update the internal session headers & add retries.
try:
    from nba_api.stats.library.http import NBAStatsHTTP
    _UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
    NBAStatsHTTP._COMMON_HEADERS.update({
        "User-Agent": _UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    })
    # force a new session with updated headers
    NBAStatsHTTP._session = None
except Exception:
    # if library internals change, we still have retries below
    pass

REQUEST_TIMEOUT = 25
MAX_RETRIES = 4
BACKOFF = 0.7

def _with_retries(func, *args, **kwargs):
    """Run nba_api call with retries + exponential backoff."""
    for i in range(MAX_RETRIES):
        try:
            return func(*args, timeout=REQUEST_TIMEOUT, **kwargs)
        except Exception as e:
            if i == MAX_RETRIES - 1:
                raise
            sleep_s = BACKOFF * (2 ** i) + np.random.rand() * 0.3
            time.sleep(sleep_s)

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Hot Shot Props • NBA Analytics", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {background:#000;color:#f4f4f4;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#000 0%,#111 100%)!important;}
h1,h2,h3,h4,h5 {color:#ff5555;font-weight:700;}
[data-testid="stMetric"] {background:#111;border-radius:12px;padding:10px;border:1px solid #222;}
[data-testid="stMetric"] label{color:#ff7777;}
[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#fff;font-size:1.4em;}
.leader-card{display:flex;align-items:center;gap:14px;background:#0d0d0d;border:1px solid #222;
padding:10px;border-radius:10px;margin-bottom:8px;}
.leader-img img{width:55px;height:55px;border-radius:8px;}
.leader-info{display:flex;flex-direction:column;}
.leader-info a{color:#ffb4b4;text-decoration:none;font-weight:bold;}
.leader-stat{color:#ccc;font-size:0.9em;}
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def current_season():
    today = dt.date.today()
    y = today.year if today.month >= 10 else today.year - 1
    return f"{y}-{str(y+1)[-2:]}"

SEASON = current_season()

# ---------------- Favorites Persistence ----------------
DEFAULT_TMP = "/tmp" if os.access("/", os.W_OK) else "."
FAV_PATH = os.path.join(DEFAULT_TMP, "favorites.json")

def load_favorites() -> list:
    try:
        if os.path.exists(FAV_PATH):
            with open(FAV_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_favorites(favs: list):
    try:
        with open(FAV_PATH, "w") as f:
            json.dump(sorted(set(favs)), f)
    except Exception:
        pass

if "favorites" not in st.session_state:
    st.session_state.favorites = load_favorites()

# Build active players maps once (also used for query-param deep links)
ACTIVE_PLAYERS = players.get_active_players()
ID_TO_NAME = {p["id"]: p["full_name"] for p in ACTIVE_PLAYERS}
NAME_TO_ID = {v: k for k, v in ID_TO_NAME.items()}

# ---------------- Cached data fetchers (fast + robust) ----------------
@st.cache_data(ttl=300, show_spinner=False)
def get_league_leaders_df(season: str) -> pd.DataFrame:
    resp = _with_retries(leagueleaders.LeagueLeaders, season=season, per_mode48="PerGame")
    return resp.get_data_frames()[0]

@st.cache_data(ttl=600, show_spinner=False)
def get_player_career_df(player_id: int) -> pd.DataFrame:
    resp = _with_retries(playercareerstats.PlayerCareerStats, player_id=player_id)
    return resp.get_data_frames()[0]

@st.cache_data(ttl=300, show_spinner=False)
def get_player_gamelogs_df(player_id: int, season: str) -> pd.DataFrame:
    resp = _with_retries(
        playergamelogs.PlayerGameLogs,
        player_id_nullable=player_id,
        season_nullable=season
    )
    return resp.get_data_frames()[0]

# ---------------- Sidebar ----------------
def go_home():
    st.session_state.pop("selected_player", None)
    st.experimental_rerun()

with st.sidebar:
    st.button("🏠 Home Screen", on_click=go_home, type="primary", key="home_btn")
    st.markdown("---")

    st.header("Search Player")
    player_names = sorted([p["full_name"] for p in ACTIVE_PLAYERS])
    search_name = st.selectbox("Player", player_names, index=None, placeholder="Select player")

    st.markdown("### ⭐ Favorites")
    for fav in st.session_state["favorites"]:
        cols = st.columns([4,1])
        with cols[0]:
            if st.button(fav, key=f"fav_{fav}"):
                st.session_state["selected_player"] = fav
                st.experimental_rerun()
        with cols[1]:
            if st.button("❌", key=f"rm_{fav}"):
                st.session_state["favorites"].remove(fav)
                save_favorites(st.session_state["favorites"])
                st.experimental_rerun()

# ---------------- Query param deep link ----------------
# If URL has ?player_id=XXXX -> open that player directly
try:
    qp = st.query_params
    pid_param = qp.get("player_id")
    if pid_param:
        if isinstance(pid_param, (list, tuple)):
            pid_param = pid_param[0]
        pid = int(pid_param)
        name_from_qp = ID_TO_NAME.get(pid)
        if name_from_qp:
            st.session_state["selected_player"] = name_from_qp
except Exception:
    pass

# ---------------- Home Screen ----------------
def show_home():
    st.title("🏀 NBA League Leaders")
    st.subheader(f"Season {SEASON}")
    st.info("Tip: Use the sidebar to search any player, or click a league leader below to open their page.")

    try:
        leaders = get_league_leaders_df(SEASON)
        for stat in ["PTS", "REB", "AST", "STL", "BLK", "FG3M"]:
            top = leaders.sort_values(stat, ascending=False).iloc[0]
            player_id = int(top["PLAYER_ID"])
            href = f"?player_id={player_id}"
            st.markdown(f"""
            <div class='leader-card'>
              <div class='leader-img'><img src='https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png'></div>
              <div class='leader-info'>
                <a href='{href}'>{top["PLAYER"]}</a>
                <div>{top["TEAM"]}</div>
                <div class='leader-stat'>{stat}: <b>{round(float(top[stat]),2)}</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load league leaders: {e}")

# Route to home if no player is selected via search or session
if "selected_player" not in st.session_state and not search_name:
    show_home()
    st.stop()

# ---------------- Player Detail ----------------
selected = search_name or st.session_state.get("selected_player")
player_id = NAME_TO_ID.get(selected)
if not player_id:
    st.error("Player not found.")
    st.stop()

# Update URL so the page is shareable once a player is selected
try:
    st.query_params.update({"player_id": str(player_id)})
except Exception:
    pass

st.title(f"📊 {selected}")
if st.button("⭐ Add to Favorites"):
    if selected not in st.session_state["favorites"]:
        st.session_state["favorites"].append(selected)
        save_favorites(st.session_state["favorites"])
        st.success(f"Added {selected} to favorites")

# Fetch player data (robust + cached)
load_ph = st.empty()
with load_ph.container():
    st.markdown("Loading player data…")

try:
    career_df = get_player_career_df(player_id)
    gamelogs = get_player_gamelogs_df(player_id, SEASON)
    load_ph.empty()
except Exception as e:
    load_ph.empty()
    st.error(f"Failed to load player data: {e}")
    st.stop()

# Headshot
img_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
st.image(img_url, width=240)

# Career per-game (basic)
st.subheader("Career Averages (Per Game)")
if not career_df.empty:
    df = career_df.copy()
    # Avoid div-by-zero
    df["PTS"] = df["PTS"] / df["GP"].replace(0, np.nan)
    df["REB"] = df["REB"] / df["GP"].replace(0, np.nan)
    df["AST"] = df["AST"] / df["GP"].replace(0, np.nan)
    out = df[["SEASON_ID", "TEAM_ABBREVIATION", "PTS", "REB", "AST"]].round(2)
    # stretch for Streamlit 1.39+ (no deprecation)
    st.dataframe(out, width='stretch')
else:
    st.info("No career data.")

# Last game
st.subheader("Last Game")
if not gamelogs.empty:
    last = gamelogs.iloc[0]
    cols = st.columns(4)
    for stat in ["PTS", "REB", "AST", "FG3M"]:
        cols.pop(0).metric(stat, int(last.get(stat, 0)))
else:
    st.info("No game logs found.")

# Recent form
st.subheader("Recent Form (Last 5)")
if len(gamelogs) >= 5:
    avg = gamelogs.head(5).mean(numeric_only=True)
    cols = st.columns(4)
    for stat, val in zip(["PTS", "REB", "AST", "FG3M"],
                         [avg.get("PTS", np.nan), avg.get("REB", np.nan), avg.get("AST", np.nan), avg.get("FG3M", np.nan)]):
        v = "N/A" if (pd.isna(val)) else round(float(val), 2)
        cols.pop(0).metric(stat, v)
else:
    st.info("Not enough games.")

# Simple weighted prediction
st.subheader("Predicted Next Game (Weighted Avg)")
if not gamelogs.empty:
    n = min(10, len(gamelogs))
    w = np.arange(n, 0, -1)  # most recent gets highest weight
    preds = {}
    for s in ["PTS", "REB", "AST", "FG3M"]:
        vals = gamelogs[s].head(n).astype(float).values
        preds[s] = float(np.average(vals, weights=w)) if len(vals) else np.nan
    cols = st.columns(4)
    for stat, v in preds.items():
        val = "N/A" if (pd.isna(v)) else f"{v:.1f}"
        cols.pop(0).metric(f"{stat} (ML)", val)
else:
    st.info("No logs to predict from.")

# Bar chart: last 10 mean values
st.subheader("Stat Breakdown (Last 10 Games)")
if not gamelogs.empty:
    last10 = gamelogs.head(10)[["GAME_DATE", "PTS", "REB", "AST", "FG3M"]].copy()
    df_long = last10.melt("GAME_DATE", var_name="Stat", value_name="Value")
    chart = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(x="Stat:N", y="mean(Value):Q", color="Stat:N")
        .properties(width=900, height=300)
    )
    st.altair_chart(chart, theme=None)
else:
    st.info("No recent games to chart.")

st.markdown("---")
st.caption("Hot Shot Props • NBA Analytics Dashboard ©2025")
