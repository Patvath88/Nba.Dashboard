# app.py — Hot Shot Props | NBA Player Projections (Reliable Version)
# - Adds retry + fallback to data.nba.com
# - Fully deployable on Streamlit Cloud

import os
import time
import datetime as dt
import json
import urllib3
import requests
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from nba_api.stats.static import players as static_players, teams as static_teams
from nba_api.stats.endpoints import (
    playergamelogs, playercareerstats, commonplayerinfo,
    leaguegamelog, scoreboardv2, leaguedashteamstats
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --------------------------- Streamlit Config ---------------------------
st.set_page_config(
    page_title="NBA Live Player Projections — Hot Shot Props",
    page_icon="🏀",
    layout="wide",
)

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
}

# --------------------------- Retry + Fallback ---------------------------
def _retry_nba_api(func, *args, tries=5, sleep=2.0, **kwargs):
    """Retries NBA API calls with timeout and fallback logic."""
    last_exc = None
    for i in range(tries):
        try:
            res = func(*args, headers=HEADERS, timeout=60, **kwargs)
            return res
        except Exception as e:
            last_exc = e
            time.sleep(sleep * (i + 1))

    st.warning(f"NBA API timeout after {tries} tries: {last_exc}")
    return None

def fallback_player_game_logs(player_id: int) -> pd.DataFrame:
    """Fallback to data.nba.com (faster, public JSON endpoint)"""
    try:
        url = f"https://data.nba.com/data/v2015/json/mobile_teams/nba/2024/players/player_{player_id}_gamelog.json"
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        j = r.json()
        if "gamelog" not in j:
            return pd.DataFrame()
        games = j["gamelog"]["game"]
        df = pd.DataFrame(games)
        # Normalize keys
        df.rename(columns={
            "pts": "PTS", "reb": "REB", "ast": "AST", "fg3m": "FG3M",
            "game_date": "GAME_DATE", "matchup": "MATCHUP"
        }, inplace=True)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df.sort_values("GAME_DATE", ascending=False, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

# --------------------------- Cache Helpers ---------------------------
@st.cache_data(ttl=60 * 30)
def get_players_df():
    return pd.DataFrame(static_players.get_players())

@st.cache_data(ttl=60 * 30)
def get_teams_df():
    return pd.DataFrame(static_teams.get_teams())

@st.cache_data(ttl=60 * 15)
def get_player_info(player_id: int):
    res = _retry_nba_api(commonplayerinfo.CommonPlayerInfo, player_id=player_id)
    if res is None:
        return {}
    df = res.get_data_frames()[0]
    return df.iloc[0].to_dict()

@st.cache_data(ttl=60 * 15)
def get_player_game_logs(player_id: int, season: str):
    gl = _retry_nba_api(
        playergamelogs.PlayerGameLogs,
        player_id_nullable=str(player_id),
        season_nullable=season,
        season_type_nullable="Regular Season",
    )
    if gl is None:
        # fallback
        return fallback_player_game_logs(player_id)
    return gl.get_data_frames()[0] if gl.get_data_frames() else pd.DataFrame()

# --------------------------- Helper Functions ---------------------------
def to_season_string(date: dt.date) -> str:
    year = date.year
    if date.month >= 8:
        start = year
        end = (year + 1) % 100
    else:
        start = year - 1
        end = year % 100
    return f"{start}-{end:02d}"

def safe_mean(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.mean() if len(s) else 0.0

def recent_mean(df, col, n=5):
    if df.empty or col not in df.columns:
        return 0.0
    return safe_mean(df.head(n)[col])

# --------------------------- Sidebar ---------------------------
st.sidebar.title("🏀 NBA Live Dashboard")
st.sidebar.caption("Pick a player, get live stats + projections")

players_df = get_players_df()
teams_df = get_teams_df()

query = st.sidebar.text_input("Search player", placeholder="e.g., Luka Doncic")
if query:
    matches = players_df[players_df["full_name"].str.contains(query, case=False, na=False)]
else:
    matches = players_df

player = st.sidebar.selectbox(
    "Select player",
    sorted(matches["full_name"].tolist()),
    index=0 if not matches.empty else None,
)

if not player:
    st.stop()

# --------------------------- Player Info ---------------------------
player_id = int(players_df.loc[players_df["full_name"] == player, "id"].iloc[0])
info = get_player_info(player_id)
team_name = info.get("TEAM_NAME", "—")

st.header(player)
st.caption(f"{team_name}")

# --------------------------- Game Logs ---------------------------
season_str = to_season_string(dt.date.today())
logs = get_player_game_logs(player_id, season_str)

if logs.empty:
    st.error("Could not fetch recent games (timeout or no data). Try another player.")
    st.stop()

logs = logs.sort_values("GAME_DATE", ascending=False)
show_cols = [c for c in ["GAME_DATE","MATCHUP","PTS","REB","AST","FG3M","MIN"] if c in logs.columns]
st.dataframe(logs[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

# --------------------------- Simple Projection ---------------------------
pts = recent_mean(logs, "PTS", 5)
reb = recent_mean(logs, "REB", 5)
ast = recent_mean(logs, "AST", 5)
fg3 = recent_mean(logs, "FG3M", 5)

st.subheader("Next Game Projection (based on recent form)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Points", round(pts, 2))
col2.metric("Rebounds", round(reb, 2))
col3.metric("Assists", round(ast, 2))
col4.metric("3PM", round(fg3, 2))

# --------------------------- Chart ---------------------------
tidy = logs.melt("GAME_DATE", value_vars=["PTS","REB","AST","FG3M"], var_name="Stat", value_name="Value")
chart = (
    alt.Chart(tidy)
    .mark_line(point=True)
    .encode(
        x="GAME_DATE:T",
        y="Value:Q",
        color="Stat:N",
        tooltip=["GAME_DATE:T","Stat","Value:Q"]
    )
    .properties(height=300)
)
st.altair_chart(chart, use_container_width=True)

st.caption("Built with ❤️ for Hot Shot Props — Streamlit + nba_api")
