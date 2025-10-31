# app.py — Hot Shot Props | NBA Player Analytics
# Home page widgets (last night results, standings by conference, league leaders)
# Clean sidebar (search + favorites + Home button), fast fetch (cache + concurrency),
# ML predictions (global background + player ad-hoc) with fallback,
# media-day headshots with overlay team logo, metric rows, bar charts,
# "Last predictions vs results", mobile-first UI, and Download PNG snapshot.

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
import streamlit.components.v1 as components

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playercareerstats, playergamelogs, scoreboardv2, leagueleaders, leaguedashteamstats
)

# ---------- NEW: robust requests session with browser headers + retries ----------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _nba_headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

_SESSION = None
def _get_session():
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update(_nba_headers())
        r = Retry(
            total=4, connect=4, read=4,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"])
        )
        adapter = HTTPAdapter(max_retries=r, pool_connections=20, pool_maxsize=40)
        s.mount("https://", adapter); s.mount("http://", adapter)
        _SESSION = s
    return _SESSION
# -------------------------------------------------------------------------------

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

# Hide Streamlit multipage sidebar nav (in case a /pages dir exists)
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
os.makedirs(MODE)
