# app.py — Hot Shot Props | NBA Player Analytics (with Home dashboard)
# New: Home widgets (last night results, standings by conference, league leaders)
# Notes:
# - Clicking a leader's name uses ?pid=<player_id> query param to open their page.
# - Optional email/SMS for "Share this page" via st.secrets (see comments below).

import os, json, re, time, threading, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import smtplib, ssl
from email.message import EmailMessage
import streamlit.components.v1 as components

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playercareerstats, playergamelogs, scoreboardv2, leagueleaders, leaguedashteamstats
)
# leaguestandingsv3 may not exist in some nba_api versions; we import guarded later.

# ------------------ Optional ML deps ------------------
try:
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    joblib = None
    Ridge = None
    SKLEARN_OK = False

# ------------------ Page & global CSS -----------------
st.set_page_config(page_title="Hot Shot Props • NBA Player Analytics (Free)",
                   layout="wide", initial_sidebar_state="expanded")

# Hide Streamlit multipage sidebar nav (in case /pages exists)
st.markdown("""
<style>
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"],
[data-testid="stSidebarHeader"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# Mobile viewport & iOS notch support
components.html("""
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
""", height=0)

# Global theme + mobile tweaks (black bg, red accents)
st.markdown("""
<style>
:root { --bg:#000; --panel:#0b0b0b; --ink:#f3f4f6; --line:#171717; --red:#ef4444; }
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink)!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#000 0%,#0b0b0b 100%)!important;border-right:1px solid #111;}
.stButton>button{background:var(--red)!important;color:#fff!important;border:none!important;border-radius:10px!important;padding:.55rem .95rem!important;font-weight:700!important;}
h1,h2,h3,h4{color:#ffb4b4!important;letter-spacing:.2px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 8px 28px rgba(0,0,0,.5);}
.badge{display:inline-block;padding:4px 10px;font-size:.8rem;border:1px solid var(--line);border-radius:9999px;background:#140606;color:#fca5a5;}
.hr{border:0;border-top:1px solid var(--line);margin:.75rem 0;}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.75rem;border:1px solid var(--line);margin-left:6px;}
.tag.ok{color:#bbf7d0;border-color:#064e3b;background:#052e16;}
.tag.warn{color:#fde68a;border-color:#7c2d12;background:#3b0a0a;}
.tag.dim{color:#e5e7eb;border-color:#374151;background:#111827;}
[data-testid="stMetric"]{background:#0e0e0e;border:1px solid #181818;border-radius:14px;padding:14px;box-shadow:0 10px 30px rgba(0,0,0,.6);}
[data-testid="stMetric"] label{color:#fda4a4;}
[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#ffe4e6;font-size:1.45rem;}
/* Mobile */
@media (max-width: 780px){
  .block-container { padding: 0.6rem 0.6rem !important; }
  section.main > div { padding-top: 0 !important; }
  [data-testid="column"] { width: 100% !important; display: block !important; }
  [data-testid="stMetric"]{ margin-bottom: 10px !important; }
  .stButton>button{ width: 100% !important; padding: 14px !important; font-size: 1.05rem !important; }
  .stTextInput>div>div>input, .stSelectbox>div>div>div { font-size: 1.05rem !important; }
  .st-emotion-cache-1wmy9hl, .st-emotion-cache-1r6slb0 { height: 260px !important; }
}
[data-testid="stSidebar"] .stButton>button, [data-testid="stSidebar"] .stSelectbox { font-size: 1.02rem !important; }
.leader-card{display:flex;gap:14px;align-items:center;background:#0e0e0e;border:1px solid #181818;border-radius:14px;padding:10px;margin-bottom:10px;}
.leader-img{width:56px;height:56px;border-radius:10px;overflow:hidden;border:1px solid #222;}
.leader-img img{width:56px;height:56px;object-fit:cover;}
.leader-name a{color:#ffb4b4;text-decoration:none;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ------------------ API robustness (fast defaults) ----
NBA_TIMEOUT = 15
NBA_RETRIES = 2
NBA_BACKOFF_BASE = 1.6
API_SLEEP = 0.15
FAST_MODE = True
UI_SEASON_LIMIT = 2

# ------------------ Constants & storage ---------------
STATS_COLS   = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']
PREDICT_COLS = ['PTS','AST','REB','FG3M']

DEFAULT_TMP_DIR = "/tmp" if os.access("/", os.W_OK) else "."
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(DEFAULT_TMP_DIR, "models"))
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILES  = {t: os.path.join(MODEL_DIR, f"model_{t}.pkl") for t in PREDICT_COLS}
FEAT_TEMPLATE = lambda tgt: [f'{tgt}_r5', f'{tgt}_r10', f'{tgt}_r20', f'{tgt}_season_mean', 'IS_HOME', 'DAYS_REST']

USER_ROOT = os.path.join(DEFAULT_TMP_DIR, "userdata", "guest")
os.makedirs(USER_ROOT, exist_ok=True)
FAV_FILE = os.path.join(USER_ROOT, "favorites.json")
PRED_HISTORY_FILE = os.path.join(USER_ROOT, "pred_history.json")

CACHE_ROOT = os.path.join(USER_ROOT, "cache")
os.makedirs(CACHE_ROOT, exist_ok=True)
CACHE_TTL_HOURS = 6

def _cache_path_for_player(player_id: int) -> str:
    return os.path.join(CACHE_ROOT, f"player_{player_id}_logs.parquet")
def _cache_path_csv(player_id: int) -> str:
    return os.path.join(CACHE_ROOT, f"player_{player_id}_logs.csv")

def read_logs_cache(player_id: int) -> Optional[pd.DataFrame]:
    p_parq = _cache_path_for_player(player_id)
    p_csv  = _cache_path_csv(player_id)
    path = p_parq if os.path.exists(p_parq) else (p_csv if os.path.exists(p_csv) else None)
    if not path: return None
    try:
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(path))
        if (dt.datetime.now() - mtime).total_seconds() > CACHE_TTL_HOURS * 3600:
            return None
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return None

def write_logs_cache(player_id: int, df: pd.DataFrame) -> None:
    try:
        if df is not None and not df.empty:
            try:
                df.to_parquet(_cache_path_for_player(player_id), index=False)
            except Exception:
                df.to_csv(_cache_path_csv(player_id), index=False)
    except Exception:
        pass

# ------------------ Small helpers ---------------------
def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def extract_opp_from_matchup(matchup: str) -> Optional[str]:
    if not isinstance(matchup, str): return None
    m = re.search(r'@\s*([A-Z]{3})|vs\.\s*([A-Z]{3})|VS\.\s*([A-Z]{3})', matchup, re.IGNORECASE)
    return (m.group(1) or m.group(2) or m.group(3)).upper() if m else None

def cdn_headshot(player_id: int, size: str = "1040x760") -> Optional[Image.Image]:
    url = f"https://cdn.nba.com/headshots/nba/latest/{size}/{player_id}.png"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200: return Image.open(BytesIO(r.content)).convert("RGBA")
    except Exception: pass
    return None

def cdn_team_logo(team_id: int) -> Optional[Image.Image]:
    for url in [
        f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.png",
        f"https://cdn.nba.com/logos/nba/{team_id}/global/D/logo.png",
    ]:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200: return Image.open(BytesIO(r.content)).convert("RGBA")
        except Exception:
            continue
    return None

def overlay_logo_top_right(headshot: Image.Image, logo: Optional[Image.Image],
                           padding_ratio=0.035, logo_width_ratio=0.22) -> Image.Image:
    base = headshot.copy()
    if not logo: return base
    W, H = base.size
    pad = int(W * padding_ratio)
    logo_w = int(W * logo_width_ratio)
    aspect = logo.size[1] / logo.size[0]
    logo_h = int(logo_w * aspect)
    logo_resized = logo.resize((logo_w, logo_h), Image.LANCZOS)
    base.alpha_composite(logo_resized, (W - logo_w - pad, pad))
    return base

def load_favorites() -> list:
    try:
        if os.path.exists(FAV_FILE):
            with open(FAV_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []

def save_favorites(favs: list) -> None:
    try:
        with open(FAV_FILE, "w") as f: json.dump(favs, f)
    except Exception:
        pass

def _load_pred_history() -> dict:
    try:
        if os.path.exists(PRED_HISTORY_FILE):
            with open(PRED_HISTORY_FILE, "r") as f: return json.load(f)
    except Exception:
        pass
    return {}

def _save_pred_history(data: dict) -> None:
    try:
        with open(PRED_HISTORY_FILE, "w") as f: json.dump(data, f)
    except Exception:
        pass

def record_prediction(player_id: int, player_name: str, pred_date: Optional[str],
                      engine: str, preds: dict) -> None:
    if not preds: return
    db = _load_pred_history()
    key = str(player_id)
    entries = db.get(key, [])
    if pred_date: entries = [e for e in entries if e.get("pred_date") != pred_date]
    entry = {"player_id": player_id, "player_name": player_name, "pred_date": pred_date,
             "engine": engine, "preds": {k: float(v) for k,v in preds.items() if k in PREDICT_COLS}}
    entries.append(entry)
    entries = sorted(entries, key=lambda x: (x.get("pred_date") or ""), reverse=True)[:30]
    db[key] = entries
    _save_pred_history(db)

def get_player_history(player_id: int) -> list:
    return _load_pred_history().get(str(player_id), [])

# ------------------ Cached fetchers -------------------
@st.cache_data(ttl=6*3600)
def get_active_players_fast():
    try:
        return players.get_active_players()
    except Exception as e:
        st.error(f"Error fetching active players: {e}")
        return []

@st.cache_data(ttl=12*3600)
def get_all_teams():
    try:
        return teams.get_teams()
    except Exception as e:
        st.error(f"Error fetching teams: {e}")
        return []

def get_frames_with_retry(endpoint_cls, label: str, **kwargs):
    last_err = None
    for attempt in range(NBA_RETRIES):
        try:
            ep = endpoint_cls(timeout=NBA_TIMEOUT, **kwargs)
            return ep.get_data_frames()
        except Exception as e:
            last_err = e
            time.sleep(NBA_BACKOFF_BASE ** attempt)
    st.error(f"{label}:
