# app.py — Hot Shot Props | NBA Player Analytics (NBA-only, fast, reliable)
# - Pure nba_api (no fallbacks)
# - Parallel fetch + caching + hardened headers & timeouts
# - Home: league leaders with deep-links (?player_id=...)
# - Sidebar: Home button, search, favorites
# - Player page: last game, last 5 avg, weighted prediction, bar chart

import os
import json
import time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from nba_api.stats.static import players
from nba_api.stats.endpoints import (
    leagueleaders,
    playercareerstats,
    playergamelogs,
)

# ───────────────────── Streamlit page/theme ─────────────────────
st.set_page_config(page_title="Hot Shot Props • NBA Analytics", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {background:#000;color:#f4f4f4;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#000 0%,#111 100%)!important;}
h1,h2,h3,h4,h5 {color:#ff5555;font-weight:700;}
[data-testid="stMetric"] {background:#111;border-radius:12px;padding:10px;border:1px solid #222;}
[data-testid="stMetric"] label{color:#ff7777;}
[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#fff;font-size:1.35em;}
.leader-card{display:flex;align-items:center;gap:14px;background:#0d0d0d;border:1px solid #222;
padding:10px;border-radius:10px;margin-bottom:8px;}
.leader-img img{width:55px;height:55px;border-radius:8px;}
.leader-info{display:flex;flex-direction:column;}
.leader-info a{color:#ffb4b4;text-decoration:none;font-weight:bold;}
.leader-stat{color:#ccc;font-size:0.9em;}
.hint{background:#0e0e0e;border:1px solid #222;border-radius:10px;padding:.75rem 1rem;color:#ddd;}
</style>
""", unsafe_allow_html=True)

# ───────────────────── Season helpers ─────────────────────
def current_season():
    today = dt.date.today()
    y = today.year if today.month >= 10 else today.year - 1
    return f"{y}-{str(y+1)[-2:]}"
SEASON = current_season()

# ───────────────── nba_api hardening (headers/timeouts) ─────────────
# Setting browser-like headers reduces blocks from stats.nba.com
try:
    from nba_api.stats.library.http import NBAStatsHTTP
    NBAStatsHTTP._COMMON_HEADERS.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/123.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    })
    NBAStatsHTTP._session = None  # force rebuild with new headers
except Exception:
    pass

NBA_TIMEOUT = 10
NBA_RETRIES = 2
NBA_BACKOFF = 0.45

def _nba_call(endpoint_cls, **kwargs):
    """Call an nba_api endpoint with short timeout + tiny retries."""
    last_err = None
    for i in range(NBA_RETRIES + 1):
        try:
            return endpoint_cls(timeout=NBA_TIMEOUT, **kwargs)
        except Exception as e:
            last_err = e
            if i < NBA_RETRIES:
                time.sleep(NBA_BACKOFF * (2**i) + 0.1*np.random.rand())
    raise last_err

# ───────────────── Favorites persistence ────────────────
DEFAULT_TMP = "/tmp" if os.access("/", os.W_OK) else "."
FAV_PATH = os.path.join(DEFAULT_TMP, "favorites.json")

def load_favorites() -> list:
    try:
        if os.path.exists(FAV_PATH):
            with open(FAV_PATH, "r") as f: return json.load(f)
    except Exception:
        pass
    return []

def save_favorites(favs: list):
    try:
        with open(FAV_PATH, "w") as f: json.dump(sorted(set(favs)), f)
    except Exception:
        pass

if "favorites" not in st.session_state:
    st.session_state.favorites = load_favorites()

# ───────────────── Active players index ─────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_active_maps():
    aps = players.get_active_players()
    id2name = {p["id"]: p["full_name"] for p in aps}
    name2id = {v: k for k, v in id2name.items()}
    names_sorted = sorted(name2id.keys())
    return aps, id2name, name2id, names_sorted

ACTIVE_PLAYERS, ID_TO_NAME, NAME_TO_ID, PLAYER_NAMES = get_active_maps()

# ───────────────── Cached nba_api fetchers ──────────────
@st.cache_data(ttl=900, show_spinner=False)
def get_league_leaders_df(season: str) -> pd.DataFrame:
    df = _nba_call(leagueleaders.LeagueLeaders,
                   season=season, per_mode48="PerGame").get_data_frames()[0]
    return df

@st.cache_data(ttl=900, show_spinner=False)
def get_player_career_df(player_id: int) -> pd.DataFrame:
    return _nba_call(playercareerstats.PlayerCareerStats,
                     player_id=player_id).get_data_frames()[0]

@st.cache_data(ttl=900, show_spinner=False)
def get_player_gamelogs_df(player_id: int, season: str) -> pd.DataFrame:
    gl = _nba_call(playergamelogs.PlayerGameLogs,
                   player_id_nullable=player_id,
                   season_nullable=season,
                   season_type_nullable="Regular Season").get_data_frames()[0]
    # Normalize date sort: newest first
    if "GAME_DATE" in gl.columns:
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        gl = gl.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    return gl

# ───────────────── Sidebar (Home / search / favorites) ─────────────
def go_home():
    st.session_state.pop("selected_player", None)
    try:
        st.query_params.clear()
    except Exception:
        pass
    st.rerun()  # Streamlit ≥ 1.40

with st.sidebar:
    st.button("🏠 Home Screen", on_click=go_home, type="primary", key="home_btn")
    st.markdown("---")
    st.header("Search Player")
    search_name = st.selectbox("Player", PLAYER_NAMES, index=None, placeholder="Select player")
    st.markdown("### ⭐ Favorites")
    if not st.session_state["favorites"]:
        st.caption("No favorites yet.")
    for fav in st.session_state["favorites"]:
        c1, c2 = st.columns([4,1])
        if c1.button(fav, key=f"fav_{fav}"):
            st.session_state["selected_player"] = fav
            st.rerun()
        if c2.button("❌", key=f"rm_{fav}"):
            st.session_state["favorites"].remove(fav)
            save_favorites(st.session_state["favorites"])
            st.rerun()

# ───────────── Query param → open player directly ─────────────
try:
    pid_param = st.query_params.get("player_id")
    if pid_param:
        if isinstance(pid_param, (list, tuple)): pid_param = pid_param[0]
        pid = int(pid_param)
        if pid in ID_TO_NAME:
            st.session_state["selected_player"] = ID_TO_NAME[pid]
except Exception:
    pass

# ───────────────── Home Screen ─────────────────
def show_home():
    st.title("🏀 NBA League Leaders")
    st.subheader(f"Season {SEASON}")
    st.markdown("<div class='hint'>💡 Use the sidebar to search any player, "
                "or click a league leader below to open their page.</div>",
                unsafe_allow_html=True)
    try:
        leaders = get_league_leaders_df(SEASON)
        for stat in ["PTS", "REB", "AST", "STL", "BLK", "FG3M"]:
            top = leaders.sort_values(stat, ascending=False).iloc[0]
            player_id = int(top["PLAYER_ID"])
            href = f"?player_id={player_id}"
            st.markdown(f"""
            <div class='leader-card'>
              <div class='leader-img'>
                <img src='https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png'>
              </div>
              <div class='leader-info'>
                <a href='{href}'>{top["PLAYER"]}</a>
                <div>{top["TEAM"]}</div>
                <div class='leader-stat'>{stat}: <b>{round(float(top[stat]),2)}</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load league leaders: {e}")

if "selected_player" not in st.session_state and not search_name:
    show_home()
    st.stop()

# ───────────────── Player Detail ─────────────────
selected = search_name or st.session_state.get("selected_player")
player_id = NAME_TO_ID.get(selected)

# Update URL for deep-link
try:
    if player_id:
        st.query_params.update({"player_id": str(player_id)})
    else:
        st.query_params.clear()
except Exception:
    pass

st.title(f"📊 {selected}")

if st.button("⭐ Add to Favorites"):
    if selected not in st.session_state["favorites"]:
        st.session_state["favorites"].append(selected)
        save_favorites(st.session_state["favorites"])
        st.success(f"Added {selected} to favorites")

spinner = st.empty()
spinner.info("Loading player data…")

def load_player_fast(pid: int):
    """Fetch career & gamelogs in parallel to reduce latency."""
    def _career():  return get_player_career_df(pid)
    def _glogs():   return get_player_gamelogs_df(pid, SEASON)
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_career): "career", ex.submit(_glogs): "logs"}
        career_df, glogs = None, None
        for fut in as_completed(futs):
            kind = futs[fut]
            data = fut.result()
            if kind == "career": career_df = data
            else: glogs = data
    return career_df, glogs

if player_id is None:
    spinner.empty()
    st.error("Player not found in active list. Pick a name from the sidebar list.")
    st.stop()

try:
    career_df, gamelogs = load_player_fast(player_id)
except Exception as e:
    spinner.empty()
    st.error(f"Failed to load from nba_api (likely a temporary block): {e}")
    st.stop()

spinner.empty()

# Headshot
st.image(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png", width=230)

# Career per-game table (compact)
st.subheader("Career Averages (Per Game)")
if not career_df.empty:
    df = career_df.copy()
    # per-season per-game from totals / GP
    if "GP" in df.columns:
        gp = df["GP"].replace(0, np.nan)
        for c in ("PTS", "REB", "AST"):
            if c in df.columns:
                df[c] = (df[c] / gp).round(2)
    display_cols = [c for c in ["SEASON_ID","TEAM_ABBREVIATION","PTS","REB","AST"] if c in df.columns]
    st.dataframe(df[display_cols].fillna(0), width='stretch')
else:
    st.info("No career/per-season data.")

# Last game
st.subheader("Last Game")
if isinstance(gamelogs, pd.DataFrame) and not gamelogs.empty:
    last = gamelogs.iloc[0]
    cols = st.columns(4)
    for i, s in enumerate(("PTS","REB","AST","FG3M")):
        val = last.get(s, np.nan)
        cols[i].metric(s, "N/A" if pd.isna(val) else int(val))
else:
    st.info("No recent game logs found.")

# Recent form (5)
st.subheader("Recent Form (Last 5)")
if isinstance(gamelogs, pd.DataFrame) and len(gamelogs) >= 5:
    avg = gamelogs.head(5).mean(numeric_only=True)
    cols = st.columns(4)
    stats = [("PTS", avg.get("PTS", np.nan)),
             ("REB", avg.get("REB", np.nan)),
             ("AST", avg.get("AST", np.nan)),
             ("FG3M", avg.get("FG3M", np.nan))]
    for i, (s, v) in enumerate(stats):
        cols[i].metric(s, "N/A" if pd.isna(v) else round(float(v),2))
else:
    st.info("Not enough games.")

# Weighted prediction (10 → 1)
st.subheader("Predicted Next Game (Weighted Avg)")
if isinstance(gamelogs, pd.DataFrame) and not gamelogs.empty:
    n = min(10, len(gamelogs))
    w = np.arange(n, 0, -1)
    preds = {}
    for s in ("PTS","REB","AST","FG3M"):
        if s in gamelogs.columns:
            vals = gamelogs[s].head(n).astype(float).values
            preds[s] = float(np.average(vals, weights=w)) if len(vals) else np.nan
        else:
            preds[s] = np.nan
    cols = st.columns(4)
    for i, s in enumerate(("PTS","REB","AST","FG3M")):
        v = preds.get(s, np.nan)
        cols[i].metric(f"{s} (ML)", "N/A" if pd.isna(v) else f"{v:.1f}")
else:
    st.info("No logs to predict from.")

# Bar chart — last 10 games (mean per stat)
st.subheader("Stat Breakdown (Last 10 Games)")
if isinstance(gamelogs, pd.DataFrame) and not gamelogs.empty:
    keep_cols = [c for c in ["GAME_DATE","PTS","REB","AST","FG3M"] if c in gamelogs.columns]
    last10 = gamelogs[keep_cols].head(10).copy()
    long = last10.melt("GAME_DATE", var_name="Stat", value_name="Value")
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(x="Stat:N", y="mean(Value):Q", color="Stat:N")
        .properties(width=900, height=300)
    )
    st.altair_chart(chart, theme=None)
else:
    st.info("No recent games to chart.")

st.markdown("---")
st.caption("Hot Shot Props • NBA Analytics Dashboard ©2025")
