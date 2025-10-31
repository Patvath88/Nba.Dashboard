# app.py — Hot Shot Props | NBA Player Analytics (Free, Premium One-Page Dashboard)
# - Uses nba_api (no paid key)
# - Auto-refresh every 60s (no toggles)
# - Single-page layout w/ header, headshot, metric rows, trend charts
# - NBA.com-inspired dark theme
# - Robust caching and small delays to be gentle on nba_api

import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playercareerstats, playergamelogs, scoreboardv3
import pandas as pd
import numpy as np
import re, time, datetime as dt
import altair as alt
import requests
from io import BytesIO

# ---------------------------------
# Page config + global constants
# ---------------------------------
st.set_page_config(
    page_title="Hot Shot Props • NBA Player Analytics (Free)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto refresh every 60s (silent background reload)
st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

STATS_COLS = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']
PREDICT_COLS = ['PTS','AST','REB','FG3M']
API_SLEEP = 0.25  # be gentle with endpoints

# ---------------------------------
# Styling (NBA.com-ish)
# ---------------------------------
st.markdown("""
<style>
:root {
  --bg: #0f1116; --panel:#121722; --ink:#e5e7eb; --muted:#9aa3b2;
  --blue:#1d4ed8; --blue2:#1e3a8a; --line:#1f2a44;
}
html, body, [data-testid="stAppViewContainer"] { background:var(--bg); color:var(--ink); }
h1,h2,h3,h4 { color:#b3d1ff !important; letter-spacing:.2px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0f1629 0%,#0f1116 100%); border-right:1px solid #1f2937; }
.stButton > button { background:var(--blue); color:white; border:none; border-radius:10px; padding:.6rem 1rem; font-weight:700; }
.stButt
