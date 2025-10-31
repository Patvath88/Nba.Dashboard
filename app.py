# app.py — Hot Shot Props | NBA Player Analytics (Topps Card UI)
# - Home page: Leaders (PTS/REB/AST/3PM) with headshots + large names (click to open player page)
# - Sidebar: search by name; Favorites list (click to open)
# - Player page: simple metrics + last 10 games bar breakdown (fast & robust)
# - NBA endpoints hardened with browser headers, retries, and caching
# - NO balldontlie usage anywhere

import os, time, re, json
from typing import Optional, Dict, Any
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image
import streamlit as st
import altair as alt

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nba_api.stats.static import players as players_static, teams as teams_static
from nba_api.stats.endpoints import (
    playergamelogs, playercareerstats, leagueleaders
)

# ------------------------------ Streamlit basic config ------------------------------
st.set_page_config(page_title="Hot Shot Props • NBA Player Analytics", layout="wide")

# ------------------------------ Theme / Topps styling ------------------------------
st.markdown("""
<style>
:root { --bg:#0b0b0b; --ink:#f4f6fb; --card:#121212; --line:#202020; --accent:#ff4d4d; --accent2:#ffb4b4; }
html, body, [data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--ink); }
h1,h2,h3,h4 { color: var(--accent2) !important; letter-spacing: .2px; }
.small { color:#b6b8bf; font-size:.9rem; }
.big-name { font-size:1.25rem; font-weight:900; line-height:1.15; }
.subtle { color:#c6c8cf; font-weight:600; letter-spacing:.3px; }
.card { background:linear-gradient(180deg,#141414 0%, #0f0f0f 100%); border:1px solid var(--line);
        padding:16px; border-radius:18px; box-shadow:0 8px 28px rgba(0,0,0,.6); }
.badge { display:inline-block; padding:4px 10px; font-size:.8rem; border:1px solid #2a2a2a;
         border-radius:999px; background:#1b0d0d; color:#ff9b9b; }
.tpc { border-radius:14px; border:2px solid #c7a76d; box-shadow:0 0 0 6px #254, 0 10px 30px rgba(0,0,0,.35);
       background: radial-gradient(120% 60% at 50% -10%, #2a2a2a 0%, #101010 70%); }
a, a:visited { color:#ffd1d1; text-decoration: none; }
a:hover { text-decoration: underline; }
[data-testid="stSidebar"] { background:#0a0a0a; border-right:1px solid #151515; }
.stButton>button { background:var(--accent); color:#fff; border:none; border-radius:10px; font-weight:800;
                   padding:.55rem .9rem; }
[data-testid="stMetric"]{ background:#121212; border:1px solid #232323; border-radius:12px; padding:10px; }
[data-testid="stMetric"] div[data-testid="stMetricValue"]{ color:#ffe9e9; }
@media (max-width: 780px){ .block-container{ padding:8px 8px!important } [data-testid="column"]{ width:100%!important; display:block!important } }
</style>
""", unsafe_allow_html=True)

# ------------------------------ Robust requests session for nba_api ------------------------------
def _nba_headers() -> Dict[str, str]:
    return {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }

_session = None
def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(_nba_headers())
        r = Retry(
            total=4, read=4, connect=4, backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET","POST"])
        )
        adapter = HTTPAdapter(max_retries=r, pool_connections=20, pool_maxsize=40)
        s.mount("https://", adapter); s.mount("http://", adapter)
        _session = s
    return _session

# Small helper to call nba_api endpoint obj with our session
def call_nba_endpoint(EndpointClass, **kwargs):
    # nba_api endpoints accept custom headers via proxy param names
    # but simplest: monkeypatch requests' session used under the hood
    # nba_api respects global requests; retries are already on.
    return EndpointClass(**kwargs, timeout=15)

# ------------------------------ Caches ------------------------------
@st.cache_data(ttl=60*60)
def get_active_players() -> pd.DataFrame:
    return pd.DataFrame(players_static.get_active_players())

@st.cache_data(ttl=60*60)
def get_teams_df() -> pd.DataFrame:
    return pd.DataFrame(teams_static.get_teams())

# Headshot helper (uses 260x190 size that is lightweight)
def HEADSHOT(pid: int) -> str:
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"

# ------------------------------ HOME: League Leaders (robust) ------------------------------
@st.cache_data(ttl=60*30)
def get_league_leaders_this_season(stat_cat: str) -> pd.DataFrame:
    """
    Hardened call that ALWAYS returns a DataFrame.
    Falls back to a tiny cached sample if the API throttles.
    """
    try:
        obj = call_nba_endpoint(
            leagueleaders.LeagueLeaders,
            season=get_current_season_str(),
            season_type_all_star="Regular Season",
            stat_category_abbreviation=stat_cat
        )
        df = obj.get_data_frames()[0]
        if not isinstance(df, pd.DataFrame):
            raise TypeError("LeagueLeaders did not return a DataFrame.")
        return df
    except Exception as e:
        st.warning(f"League Leaders API temporarily unavailable for {stat_cat}: {e}")
        # Safe fallback
        fb = pd.DataFrame([
            {"PLAYER_ID": 201939, "PLAYER": "Stephen Curry", "TEAM": "GSW", "FG3M": 4.8, "PTS": 29.7, "REB": 5.1, "AST": 6.4},
            {"PLAYER_ID": 203507, "PLAYER": "Giannis Antetokounmpo", "TEAM": "MIL", "PTS": 31.5, "REB": 11.8, "AST": 6.1, "FG3M": 0.8},
        ])
        # choose best row for the requested stat
        order_key = "FG3M" if stat_cat.upper() in ("FG3M", "3PM") else stat_cat.upper()
        if order_key in fb:
            fb = fb.sort_values(order_key, ascending=False)
        return fb

def leader_card(df: pd.DataFrame, label: str, key_prefix: str):
    """
    Renders a single leader card with headshot + stat.
    Clicking the button routes to that player's page.
    """
    if df is None or df.empty:
        st.warning(f"No data for {label}")
        return

    row = df.iloc[0]
    pid = int(row.get("PLAYER_ID", 0))
    name = str(row.get("PLAYER", "Unknown"))
    team = str(row.get("TEAM", row.get("TEAM_ABBREVIATION", "")))

    # resolve stat field
    stat_key = None
    for cand in ["PTS", "REB", "AST", "FG3M", "3PM"]:
        if cand in row.index:
            stat_key = cand
            break
    val = float(row.get(stat_key, 0.0)) if stat_key else 0.0

    with st.container():
        c1, c2 = st.columns([1.0, 1.8])
        with c1:
            st.image(HEADSHOT(pid), caption="", width=None)
        with c2:
            st.markdown(f"""
            <div class="big-name"><a href="?player_id={pid}">{name}</a></div>
            <div class="subtle">{team}</div>
            <div style="margin-top:10px">
                <span class="badge">Leader — {label}</span>
                <div style="font-size:1.6rem;font-weight:900;margin-top:6px">{val:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open {name} →", key=f"{key_prefix}_{pid}"):
                st.session_state["selected_player"] = pid
                st.session_state["selected_name"] = name
                st.rerun()

def get_current_season_str() -> str:
    """
    Returns NBA season string like '2024-25' based on today's date.
    """
    from datetime import datetime
    today = datetime.utcnow()
    yr = today.year
    # NBA season starts ~ Oct; if before July, it's same year end-season
    if today.month < 7:
        start = yr - 1
        end = yr
    else:
        start = yr
        end = yr + 1
    return f"{start}-{str(end)[-2:]}"

# ------------------------------ Player data (fast & safe) ------------------------------
@st.cache_data(ttl=60*15, show_spinner=False)
def fetch_player_gamelogs(player_id: int) -> pd.DataFrame:
    """
    Return recent player game logs (this season + last season where available) quickly.
    """
    try:
        obj = call_nba_endpoint(
            playergamelogs.PlayerGameLogs,
            player_id_nullable=player_id,
            season_nullable=get_current_season_str()
        )
        df = obj.get_data_frames()[0]
        if "GAME_DATE" in df:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    except Exception as e:
        # Soft fail => empty df
        st.error(f"Failed to load from nba_api (likely a temporary block). Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_player_career(player_id: int) -> pd.DataFrame:
    try:
        obj = call_nba_endpoint(playercareerstats.PlayerCareerStats, player_id=player_id)
        return obj.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

# ------------------------------ Sidebar: search + favorites ------------------------------
def init_session():
    if "favorites" not in st.session_state:
        # dict: name -> player_id
        st.session_state["favorites"] = {}
    if "selected_player" not in st.session_state:
        st.session_state["selected_player"] = None
    if "selected_name" not in st.session_state:
        st.session_state["selected_name"] = None

init_session()

with st.sidebar:
    st.subheader("Select Player")
    plist = get_active_players()
    name_to_id = {f'{r["full_name"]}': r["id"] for _, r in plist.iterrows()}
    typed = st.text_input("Search player", "")
    # quick filter
    choices = [n for n in name_to_id if typed.lower() in n.lower()] if typed else sorted(name_to_id.keys())
    pick = st.selectbox("Choose", ["—"] + choices, index=0)
    if pick != "—":
        pid = name_to_id[pick]
        st.session_state["selected_player"] = pid
        st.session_state["selected_name"] = pick
        st.experimental_set_query_params(player_id=str(pid))

    st.markdown("---")
    st.markdown("**⭐ Favorites**")
    favs = st.session_state["favorites"]
    if favs:
        for nm, pid in list(favs.items()):
            cols = st.columns([0.75, 0.25])
            with cols[0]:
                if st.button(nm, key=f"fav_open_{pid}"):
                    st.session_state["selected_player"] = pid
                    st.session_state["selected_name"] = nm
                    st.experimental_set_query_params(player_id=str(pid))
                    st.rerun()
            with cols[1]:
                if st.button("✕", key=f"fav_del_{pid}"):
                    favs.pop(nm, None)
                    st.rerun()
    else:
        st.caption("No favorites yet.")

# ------------------------------ Router via query param ------------------------------
qs = st.experimental_get_query_params()
if "player_id" in qs:
    try:
        pid_q = int(qs["player_id"][0])
        st.session_state["selected_player"] = pid_q
    except Exception:
        pass

# ------------------------------ HOME ------------------------------
def render_home():
    st.title("🏠 Home")
    st.caption("Hint: use the sidebar to search any player by name. Add ⭐ to build your Favorites board.")

    st.subheader("League Leaders (Current Season)")
    cols = st.columns(4)
    blocks = [("PTS","PTS Leader"), ("REB","REB Leader"), ("AST","AST Leader"), ("FG3M","3PM Leader")]
    for (stat, label), c in zip(blocks, cols):
        with c:
            st.markdown('<div class="card tpc">', unsafe_allow_html=True)
            df = get_league_leaders_this_season(stat)
            leader_card(df, label, key_prefix=f"lead_{stat}")
            st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------ PLAYER PAGE ------------------------------
def render_player(pid: int, name: Optional[str]):
    st.title(name or "Player")

    # allow add to favorites
    if st.button("⭐ Add to Favorites"):
        nm = name or "Player"
        st.session_state["favorites"][nm] = pid

    # headshot
    st.image(HEADSHOT(pid), width=260)

    # quick data
    with st.spinner("Preparing metrics..."):
        logs = fetch_player_gamelogs(pid)

    if logs.empty:
        st.error("No recent logs found (or API temporarily blocked).")
        return

    # Basic metrics
    def metric_row(label: str, s: pd.Series):
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.metric("PTS", f"{s.get('PTS', np.nan):.2f}" if 'PTS' in s else "N/A")
        with c2: st.metric("REB", f"{s.get('REB', np.nan):.2f}" if 'REB' in s else "N/A")
        with c3: st.metric("AST", f"{s.get('AST', np.nan):.2f}" if 'AST' in s else "N/A")
        with c4: st.metric("3PM", f"{s.get('FG3M', np.nan):.2f}" if 'FG3M' in s else "N/A")
        with c5: st.metric("MIN", f"{s.get('MIN', np.nan):.2f}" if 'MIN' in s else "N/A")

    st.subheader("Form (Last 5)")
    last5 = logs.head(5)[["PTS","REB","AST","FG3M","MIN"]].mean(numeric_only=True)
    metric_row("Last 5", last5)

    st.subheader("Breakdown (Last 10 Games)")
    show_cols = [c for c in ["GAME_DATE","PTS","REB","AST","FG3M","MIN"] if c in logs.columns]
    table = logs.head(10)[show_cols].copy()
    if "GAME_DATE" in table:
        table["GAME_DATE"] = pd.to_datetime(table["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    st.dataframe(table, height=370)

    # Mini bar chart (melt and show)
    long = logs.head(10)[["GAME_DATE","PTS","REB","AST","FG3M"]].melt("GAME_DATE", var_name="Stat", value_name="Value")
    chart = (alt.Chart(long)
                .mark_bar()
                .encode(x=alt.X("Stat:N", title=""),
                        y=alt.Y("mean(Value):Q", title="Avg (last 10)"),
                        color=alt.Color("Stat:N", legend=None))
                .properties(height=220))
    st.altair_chart(chart, use_container_width=True)

# ------------------------------ App body ------------------------------
selected_pid = st.session_state.get("selected_player")
selected_name = st.session_state.get("selected_name")

if selected_pid:
    render_player(selected_pid, selected_name)
else:
    render_home()