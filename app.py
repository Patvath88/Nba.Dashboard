# app.py — Hot Shot Props | NBA Player Analytics (nba_api + balldontlie fallback)
# - Home: League Leaders (deep-links to player via ?player_id=...)
# - Player page: metrics, recent form, weighted prediction, bar charts
# - Robust: nba_api with headers+retries (fast), automatic fallback to balldontlie
# - Caching + parallel fetch, small timeouts, clear source badge

import os, json, time, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from nba_api.stats.static import players
from nba_api.stats.endpoints import (
    leagueleaders, playercareerstats, playergamelogs
)

# ─────────────────────────── Page / Theme ───────────────────────────
st.set_page_config(page_title="Hot Shot Props • NBA Analytics", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {background:#000;color:#f4f4f4;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#000 0%,#111 100%)!important;}
h1,h2,h3,h4,h5 {color:#ff5555;font-weight:700;}
[data-testid="stMetric"] {background:#111;border-radius:12px;padding:10px;border:1px solid #222;}
[data-testid="stMetric"] label{color:#ff7777;}
[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#fff;font-size:1.35em;}
.badge{display:inline-block;padding:3px 8px;border:1px solid #333;border-radius:999px;
background:#121212;color:#ddd;font-size:.78rem;margin-left:8px}
.badge.ok{color:#9ae6b4;border-color:#22543d;background:#0f2e1f}
.badge.warn{color:#fbd38d;border-color:#7b341e;background:#2a1a12}
.leader-card{display:flex;align-items:center;gap:14px;background:#0d0d0d;border:1px solid #222;
padding:10px;border-radius:10px;margin-bottom:8px;}
.leader-img img{width:55px;height:55px;border-radius:8px;}
.leader-info{display:flex;flex-direction:column;}
.leader-info a{color:#ffb4b4;text-decoration:none;font-weight:bold;}
.leader-stat{color:#ccc;font-size:0.9em;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────── Season / Helpers ───────────────────────────
def current_season():
    today = dt.date.today()
    y = today.year if today.month >= 10 else today.year - 1
    return f"{y}-{str(y+1)[-2:]}", y

SEASON, SEASON_START_YEAR = current_season()

# ───────────────────── nba_api Hardening ────────────────────────────
# Proper headers; shorter timeout; retries; and we expose a wrapper.
try:
    from nba_api.stats.library.http import NBAStatsHTTP
    NBAStatsHTTP._COMMON_HEADERS.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    })
    NBAStatsHTTP._session = None
except Exception:
    pass

NBA_TIMEOUT = 10
NBA_RETRIES = 2
NBA_BACKOFF = 0.5

def _nba_call(fn, **kwargs):
    err = None
    for i in range(NBA_RETRIES + 1):
        try:
            return fn(timeout=NBA_TIMEOUT, **kwargs)
        except Exception as e:
            err = e
            if i < NBA_RETRIES:
                time.sleep(NBA_BACKOFF * (2**i) + 0.1*np.random.rand())
    raise err

# ───────────────────── balldontlie Fallback ─────────────────────────
# Optional key (paid/GOAT or free). If present -> set Authorization header.
BDL_BASE = "https://api.balldontlie.io/v1"
BDL_KEY  = os.environ.get("BALLDONTLIE_API_KEY", "").strip()

def _bdl_session():
    s = requests.Session()
    if BDL_KEY:
        s.headers["Authorization"] = f"Bearer {BDL_KEY}"
    s.headers["User-Agent"] = "HotShotProps/1.0"
    return s

BDL_TIMEOUT = 8
BDL_RETRIES = 2

def _bdl_get(path, params=None):
    s = _bdl_session()
    url = f"{BDL_BASE}{path}"
    err = None
    for i in range(BDL_RETRIES + 1):
        try:
            r = s.get(url, params=params or {}, timeout=BDL_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            err = e
            if i < BDL_RETRIES:
                time.sleep(0.4 * (2**i))
    raise err

# ───────────────────── Favorites Persistence ────────────────────────
DEFAULT_TMP = "/tmp" if os.access("/", os.W_OK) else "."
FAV_PATH = os.path.join(DEFAULT_TMP, "favorites.json")

def load_favorites() -> list:
    try:
        if os.path.exists(FAV_PATH):
            with open(FAV_PATH, "r") as f: return json.load(f)
    except Exception: pass
    return []

def save_favorites(favs: list):
    try:
        with open(FAV_PATH, "w") as f: json.dump(sorted(set(favs)), f)
    except Exception: pass

if "favorites" not in st.session_state:
    st.session_state.favorites = load_favorites()

# ───────────────────── Active players index ─────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_active_maps():
    aps = players.get_active_players()
    id2name = {p["id"]: p["full_name"] for p in aps}
    name2id = {v: k for k, v in id2name.items()}
    names_sorted = sorted(name2id.keys())
    return aps, id2name, name2id, names_sorted

ACTIVE_PLAYERS, ID_TO_NAME, NAME_TO_ID, PLAYER_NAMES = get_active_maps()

# ───────────────────────── Cached fetchers ──────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def get_league_leaders_df(season: str) -> pd.DataFrame:
    df = _nba_call(leagueleaders.LeagueLeaders, season=season, per_mode48="PerGame").get_data_frames()[0]
    return df

@st.cache_data(ttl=600, show_spinner=False)
def get_player_career_df_nba(player_id: int) -> pd.DataFrame:
    return _nba_call(playercareerstats.PlayerCareerStats, player_id=player_id).get_data_frames()[0]

@st.cache_data(ttl=600, show_spinner=False)
def get_player_gamelogs_df_nba(player_id: int, season: str) -> pd.DataFrame:
    gl = _nba_call(
        playergamelogs.PlayerGameLogs,
        player_id_nullable=player_id,
        season_nullable=season,
        season_type_nullable="Regular Season",
    ).get_data_frames()[0]
    return gl

# ─────────────────── balldontlie helpers (fallback) ─────────────────
@st.cache_data(ttl=1200, show_spinner=False)
def bdl_find_player_id_by_name(name: str) -> int | None:
    # Try exact full_name match first; otherwise first search result.
    data = _bdl_get("/players", params={"search": name, "per_page": 25})
    d = data.get("data", [])
    if not d: return None
    # Best match
    for p in d:
        full = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        if full.lower() == name.lower():
            return int(p["id"])
    return int(d[0]["id"])

@st.cache_data(ttl=600, show_spinner=False)
def bdl_season_averages(player_id: int, season_year: int) -> dict:
    # https://api.balldontlie.io/v1/season_averages?season=2025&player_ids[]=237
    js = _bdl_get("/season_averages", params={"season": season_year, "player_ids[]": player_id})
    arr = js.get("data", [])
    return arr[0] if arr else {}

@st.cache_data(ttl=600, show_spinner=False)
def bdl_last_n_games(player_id: int, season_year: int, n: int = 10) -> pd.DataFrame:
    # /games?player_ids[]=...&seasons[]=2025&per_page=100
    js = _bdl_get("/games", params={"player_ids[]": player_id, "seasons[]": season_year, "per_page": 100})
    rows = js.get("data", [])
    if not rows: return pd.DataFrame()
    # newest first to match nba_api behavior
    rows = sorted(rows, key=lambda r: r.get("date",""), reverse=True)
    rows = rows[:n]
    out = []
    for g in rows:
        stat = g.get("player", {}) or {}
        # balldontlie returns stats in 'stats' for some tiers; for v2 it’s flattened under 'pts', 'ast', etc in game object
        # We'll handle common fields with safe defaults
        out.append({
            "GAME_DATE": g.get("date","")[:10],
            "PTS": g.get("pts", 0) or 0,
            "REB": g.get("reb", 0) or 0,
            "AST": g.get("ast", 0) or 0,
            "FG3M": g.get("fg3m", 0) or 0,
        })
    return pd.DataFrame(out)

# ───────────────────────────── Sidebar ──────────────────────────────
def go_home():
    st.session_state.pop("selected_player", None)
try:
    st.query_params.clear()
except Exception:
    pass
st.rerun()

with st.sidebar:
    st.button("🏠 Home Screen", on_click=go_home, type="primary", key="home_btn")
    st.markdown("---")
    st.header("Search Player")
    search_name = st.selectbox("Player", PLAYER_NAMES, index=None, placeholder="Select player")
    st.markdown("### ⭐ Favorites")
    for fav in st.session_state["favorites"]:
        cols = st.columns([4,1])
        if cols[0].button(fav, key=f"fav_{fav}"):
            st.session_state["selected_player"] = fav
            st.experimental_rerun()
        if cols[1].button("❌", key=f"rm_{fav}"):
            st.session_state["favorites"].remove(fav)
            save_favorites(st.session_state["favorites"])
            st.experimental_rerun()

# ─────────────── Query param -> open player directly ────────────────
try:
    qp = st.query_params
    pid_param = qp.get("player_id")
    if pid_param:
        if isinstance(pid_param, (list, tuple)): pid_param = pid_param[0]
        pid = int(pid_param)
        if pid in ID_TO_NAME:
            st.session_state["selected_player"] = ID_TO_NAME[pid]
except Exception:
    pass

# ─────────────────────────── Home Screen ────────────────────────────
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

if "selected_player" not in st.session_state and not search_name:
    show_home()
    st.stop()

# ───────────────────────── Player Detail ────────────────────────────
selected = search_name or st.session_state.get("selected_player")
player_id = NAME_TO_ID.get(selected)  # nba_api id (if any)

# update URL for deep-link
try: 
    if player_id: st.query_params.update({"player_id": str(player_id)})
    else: st.query_params.clear()
except Exception: 
    pass

st.title(f"📊 {selected}")

if st.button("⭐ Add to Favorites"):
    if selected not in st.session_state["favorites"]:
        st.session_state["favorites"].append(selected)
        save_favorites(st.session_state["favorites"])
        st.success(f"Added {selected} to favorites")

# Try nba_api first (fast + parallel). If it fails, transparently fallback to balldontlie.
spinner = st.empty()
spinner.info("Loading player data…")

source_label = st.empty()

def load_from_nba(pid: int):
    def _career():   return get_player_career_df_nba(pid)
    def _glogs():    return get_player_gamelogs_df_nba(pid, SEASON)

    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_career): "career", ex.submit(_glogs): "logs"}
        career_df, glogs = None, None
        for fut in as_completed(futs):
            kind = futs[fut]
            data = fut.result()
            if kind == "career": career_df = data
            else: glogs = data
    return career_df, glogs

def load_from_bdl(name: str):
    bdl_pid = bdl_find_player_id_by_name(name)
    if not bdl_pid:
        raise RuntimeError("balldontlie: player not found")

    # Season averages (for per-game display) + last N games for recent/prediction
    season_avg = bdl_season_averages(bdl_pid, SEASON_START_YEAR)
    glogs = bdl_last_n_games(bdl_pid, SEASON_START_YEAR, n=10)

    # Build a pseudo career_df with just this season (per-game)
    if season_avg:
        cdf = pd.DataFrame([{
            "SEASON_ID": SEASON,
            "TEAM_ABBREVIATION": "",  # bdl does not return abbr in season_averages
            "GP": season_avg.get("games_played", 0),
            "PTS": season_avg.get("pts", 0),
            "REB": season_avg.get("reb", 0),
            "AST": season_avg.get("ast", 0),
        }])
    else:
        cdf = pd.DataFrame()

    return cdf, glogs

# Decide source
career_df, gamelogs = None, None
used_source = "nba_api"
try:
    if player_id is None:
        # Name was typed that isn't in nba_api active list → go straight to bdl
        raise RuntimeError("Unknown nba_api id")
    career_df, gamelogs = load_from_nba(player_id)
    source_label.markdown("<span class='badge ok'>Source: nba_api</span>", unsafe_allow_html=True)
except Exception:
    # fallback
    used_source = "balldontlie"
    try:
        career_df, gamelogs = load_from_bdl(selected)
        source_label.markdown("<span class='badge warn'>Source: balldontlie</span>", unsafe_allow_html=True)
    except Exception as e2:
        spinner.empty()
        st.error(f"Failed to load from nba_api and balldontlie: {e2}")
        st.stop()

spinner.empty()

# Headshot (will work only for nba_api ids; okay if it 404s)
head_id = player_id if player_id is not None else 237  # default to a valid id (LeBron) to avoid blank
st.image(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{head_id}.png", width=230)

# Career per-game (lean)
st.subheader("Career Averages (Per Game)")
if not career_df.empty:
    df = career_df.copy()
    if "GP" in df.columns:
        gp = df["GP"].replace(0, np.nan)
        for c in ("PTS","REB","AST"):
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
    for s in ("PTS","REB","AST","FG3M"):
        val = last.get(s, np.nan)
        cols.pop(0).metric(s, "N/A" if pd.isna(val) else int(val))
else:
    st.info("No recent game logs found.")

# Recent form (5)
st.subheader("Recent Form (Last 5)")
if isinstance(gamelogs, pd.DataFrame) and len(gamelogs) >= 5:
    avg = gamelogs.head(5).mean(numeric_only=True)
    cols = st.columns(4)
    vals = [avg.get("PTS", np.nan), avg.get("REB", np.nan), avg.get("AST", np.nan), avg.get("FG3M", np.nan)]
    for s, v in zip(("PTS","REB","AST","FG3M"), vals):
        cols.pop(0).metric(s, "N/A" if pd.isna(v) else round(float(v),2))
else:
    st.info("Not enough games.")

# Weighted prediction (10 -> 1)
st.subheader("Predicted Next Game (Weighted Avg)")
if isinstance(gamelogs, pd.DataFrame) and not gamelogs.empty:
    n = min(10, len(gamelogs))
    w = np.arange(n, 0, -1)
    preds = {}
    for s in ("PTS","REB","AST","FG3M"):
        vals = gamelogs[s].head(n).astype(float).values if s in gamelogs.columns else np.array([])
        preds[s] = float(np.average(vals, weights=w)) if len(vals) else np.nan
    cols = st.columns(4)
    for s in ("PTS","REB","AST","FG3M"):
        v = preds.get(s, np.nan)
        cols.pop(0).metric(f"{s} (ML)", "N/A" if pd.isna(v) else f"{v:.1f}")
else:
    st.info("No logs to predict from.")

# Bar chart — last 10 mean values
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
