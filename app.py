# -------------------------------------------------
# HOT SHOT PROPS – NBA PLAYER DASHBOARD (FUTURISTIC PATCH)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, playercareerstats
from PIL import Image
import requests
from io import BytesIO

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Hot Shot Props | NBA Dashboard", page_icon="🔥", layout="wide")

CURRENT_SEASON = "2025-26"
LAST_SEASON = "2024-25"

ASSETS_DIR = "assets"
PLAYER_PHOTO_DIR = os.path.join(ASSETS_DIR, "player_photos")
TEAM_LOGO_DIR = os.path.join(ASSETS_DIR, "team_logos")
os.makedirs(PLAYER_PHOTO_DIR, exist_ok=True)
os.makedirs(TEAM_LOGO_DIR, exist_ok=True)

# -------------------------------------------------
# STYLE (FUTURISTIC)
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color:#0d0d0d;
    color:#F5F5F5;
    font-family:'Roboto',sans-serif;
}
h1,h2,h3,h4 {
    font-family:'Oswald',sans-serif;
    color:#E50914;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    margin-bottom: 15px;
}
.metric-card {
    background: linear-gradient(145deg, #1b1b1b, #121212);
    border: 1px solid #2a2a2a;
    border-radius: 15px;
    padding: 12px;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 0 10px rgba(229,9,20,0.3);
}
.metric-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 0 20px rgba(229,9,20,0.6);
}
.metric-value {
    font-size: 1.6em;
    color: #ffffff;
    font-weight: bold;
}
.metric-label {
    font-size: 0.9em;
    color: #aaa;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
@st.cache_data(ttl=3600)
def get_player_photo(player_name, player_id=None):
    safe = player_name.replace(" ", "_").lower()
    local_path = os.path.join(PLAYER_PHOTO_DIR, f"{safe}.png")
    if os.path.exists(local_path):
        return local_path
    try:
        if player_id:
            url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content))
                img.save(local_path)
                return local_path
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def get_team_logo(team_abbr):
    safe = team_abbr.lower()
    path = os.path.join(TEAM_LOGO_DIR, f"{safe}.png")
    if os.path.exists(path):
        return path
    try:
        url = f"https://loodibee.com/wp-content/uploads/nba-{safe}-logo.png"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content))
            img.save(path)
            return path
    except Exception:
        pass
    return None

@st.cache_data(ttl=900)
def get_games(player_id, season):
    try:
        log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        return log.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def get_career(player_id):
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        return career.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

def enrich_stats(df):
    if df.empty:
        return df
    df["P+R"] = df["PTS"] + df["REB"]
    df["P+A"] = df["PTS"] + df["AST"]
    df["R+A"] = df["REB"] + df["AST"]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

def get_last_game(df):
    if df.empty:
        return {}
    g = df.iloc[0]
    return {
        "GAME_DATE": g["GAME_DATE"], "PTS": g["PTS"], "REB": g["REB"], "AST": g["AST"],
        "FG3M": g["FG3M"], "STL": g["STL"], "BLK": g["BLK"], "TOV": g["TOV"], "MIN": g["MIN"]
    }

def compute_avg(df):
    if df.empty:
        return {}
    stats = ["PTS","REB","AST","FG3M","STL","BLK","TOV","P+R","P+A","R+A","PRA","MIN"]
    return {s: round(df[s].mean(),1) for s in stats if s in df.columns}

# -------------------------------------------------
# PLAYER SELECTION
# -------------------------------------------------
nba_players = players.get_active_players()
nba_teams = teams.get_teams()
team_lookup = {t["id"]: t for t in nba_teams}

team_map = {}
for p in nba_players:
    tid = p.get("team_id")
    abbr = team_lookup[tid]["abbreviation"] if tid in team_lookup else "FA"
    team_map.setdefault(abbr, []).append(p["full_name"])

team_options = []
for t, plist in sorted(team_map.items()):
    team_options.append(f"=== {t} ===")
    team_options.extend(sorted(plist))

st.title("🏀 Hot Shot Props — NBA Player Dashboard")
st.subheader("Live Stats, Splits, and Career Averages")

selected_player = st.selectbox(
    "Search or Browse by Team ↓",
    options=team_options,
    index=None,
    placeholder="Select an NBA player"
)
if selected_player is None or selected_player == "" or selected_player.startswith("==="):
    st.stop()

pinfo = next((p for p in nba_players if p["full_name"] == selected_player), None)
if not pinfo:
    st.error("Player not found.")
    st.stop()

player_id = pinfo["id"]
team_abbr = "FA"
if pinfo.get("team_id") in team_lookup:
    team_abbr = team_lookup[pinfo["team_id"]]["abbreviation"]

photo = get_player_photo(selected_player, player_id)
logo = get_team_logo(team_abbr)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
games_current = enrich_stats(get_games(player_id, CURRENT_SEASON))
if games_current.empty:
    games_current = enrich_stats(get_games(player_id, LAST_SEASON))
games_last = enrich_stats(get_games(player_id, LAST_SEASON))
career_df = get_career(player_id)

# -------------------------------------------------
# TOP BANNER
# -------------------------------------------------
st.markdown("### Player Summary")
col1, col2 = st.columns([1, 3])
latest = get_last_game(games_current)
with col1:
    if photo: st.image(photo, width="stretch")
with col2:
    if logo: st.image(logo, width=100)
    st.markdown(f"## {selected_player} ({team_abbr})")
    if latest:
        st.markdown(f"**Last Game:** {latest['GAME_DATE']}")
        st.markdown(
            f"PTS: {latest['PTS']} | REB: {latest['REB']} | AST: {latest['AST']} | "
            f"3PM: {latest['FG3M']} | STL: {latest['STL']} | BLK: {latest['BLK']} | "
            f"TOV: {latest['TOV']} | MIN: {latest['MIN']}"
        )

# -------------------------------------------------
# EXPANDER RENDERING (WITH SELECTOR + FUTURISTIC CARDS)
# -------------------------------------------------
def render_expander(title, df):
    if df.empty:
        st.warning(f"No data available for {title}")
        return

    avg = compute_avg(df)

    # Futuristic metric cards
    st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
    for stat, val in avg.items():
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{stat}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Dropdown metric selector
    selectable_stats = [s for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","MIN","P+R","P+A","R+A","PRA"] if s in df.columns]
    metric_choice = st.selectbox("📊 Choose a metric to visualize:", selectable_stats, key=f"metric_select_{title}")

    if "GAME_DATE" in df.columns and metric_choice in df.columns:
        x = df["GAME_DATE"].iloc[::-1]
        y = df[metric_choice].iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=y, name=metric_choice, marker_color="#E50914", opacity=0.6))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers",
                                 line=dict(color="#29B6F6", width=3),
                                 marker=dict(size=8)))
        fig.update_layout(
            paper_bgcolor="#0d0d0d",
            plot_bgcolor="#0d0d0d",
            font_color="#F5F5F5",
            title=f"{metric_choice} Trend — {title.title()}",
            yaxis_title=metric_choice,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom"),
        )
        st.plotly_chart(fig, width="stretch", key=f"chart_{title}_{metric_choice}")
    else:
        st.caption("No game-level data available to chart for this section.")

# -------------------------------------------------
# EXPANDERS
# -------------------------------------------------
with st.expander("📅 Last 5 Games", expanded=False):
    render_expander("last5", games_current.head(5))
with st.expander("📅 Last 10 Games", expanded=False):
    render_expander("last10", games_current.head(10))
with st.expander("📅 Last 20 Games", expanded=False):
    df20 = games_current.copy()
    if len(df20) < 20:
        need = 20 - len(df20)
        df20 = pd.concat([df20, games_last.head(need)], ignore_index=True)
    render_expander("last20", df20)
if not games_current.empty:
    with st.expander("📊 Current Season Averages", expanded=False):
        render_expander("currentSeason", games_current)
if not games_last.empty:
    with st.expander("🕰️ Last Season Averages", expanded=False):
        render_expander("lastSeason", games_last)
if not career_df.empty:
    career_avg = career_df.groupby("PLAYER_ID").agg({
        "PTS":"mean","REB":"mean","AST":"mean","FG3M":"mean",
        "STL":"mean","BLK":"mean","TOV":"mean","MIN":"mean"
    }).reset_index()
    career_avg = enrich_stats(career_avg)
    with st.expander("🏆 Career Averages", expanded=False):
        render_expander("career", career_avg)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(f"""
---
<div style='text-align:center;color:#777;font-size:13px;margin-top:20px;'>
Hot Shot Props © {datetime.now().year} | Powered by NBA API + Streamlit
</div>
""", unsafe_allow_html=True)
