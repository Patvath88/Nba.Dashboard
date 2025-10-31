# app.py — Hot Shot Props | NBA Player Analytics Dashboard
# Streamlit Cloud-ready, 2025-26 Season
# Clean black/red UI • League Leaders • Player Detail • PNG Snapshot

import os, json, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from nba_api.stats.static import players
from nba_api.stats.endpoints import (
    playercareerstats, playergamelogs, leagueleaders
)
from PIL import ImageGrab

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
def get_current_season():
    today = dt.date.today()
    year = today.year if today.month >= 10 else today.year - 1
    return f"{year}-{str(year+1)[-2:]}"

CURRENT_SEASON = get_current_season()

# ---------------- Favorites Persistence ----------------
DEFAULT_TMP_DIR = "/tmp" if os.access("/", os.W_OK) else "."
FAV_PATH = os.path.join(DEFAULT_TMP_DIR, "favorites.json")

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

# ---------------- Sidebar ----------------
def go_home():
    st.session_state.pop("selected_player", None)
    st.experimental_rerun()

with st.sidebar:
    st.button("🏠 Home Screen", on_click=go_home, type="primary", key="home_btn")
    st.markdown("---")

    st.header("Search Player / Team")
    all_players = players.get_active_players()
    player_names = sorted([p["full_name"] for p in all_players])
    search_name = st.selectbox("Player Search", player_names, index=None, placeholder="Select player")

    st.markdown("### ⭐ Favorites")
    for fav in st.session_state["favorites"]:
        cols = st.columns([4,1])
        with cols[0]:
            if st.button(fav, key=fav):
                st.session_state["selected_player"] = fav
                st.experimental_rerun()
        with cols[1]:
            if st.button("❌", key=f"rm_{fav}"):
                st.session_state["favorites"].remove(fav)
                save_favorites(st.session_state["favorites"])
                st.experimental_rerun()

# ---------------- Home Screen (League Leaders Only) ----------------
def show_home():
    st.title("🏀 NBA League Leaders")
    st.subheader(f"Season {CURRENT_SEASON}")

    try:
        leaders = leagueleaders.LeagueLeaders(season=CURRENT_SEASON, per_mode48="PerGame").get_data_frames()[0]
        for stat in ["PTS","REB","AST","STL","BLK","FG3M"]:
            top = leaders.sort_values(stat, ascending=False).iloc[0]
            st.markdown(f"""
            <div class='leader-card'>
              <div class='leader-img'><img src='https://cdn.nba.com/headshots/nba/latest/1040x760/{int(top["PLAYER_ID"])}.png'></div>
              <div class='leader-info'>
                <a href='?player={int(top["PLAYER_ID"])}'>{top["PLAYER"]}</a>
                <div>{top["TEAM"]}</div>
                <div class='leader-stat'>{stat}: <b>{round(top[stat],2)}</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load leaders: {e}")

if "selected_player" not in st.session_state and not search_name:
    show_home()
    st.stop()

# ---------------- Player Detail ----------------
selected = search_name or st.session_state.get("selected_player")
pid = next((p["id"] for p in players.get_active_players() if p["full_name"] == selected), None)
if not pid:
    st.error("Player not found.")
    st.stop()

st.title(f"📊 {selected}")
if st.button("⭐ Add to Favorites"):
    if selected not in st.session_state["favorites"]:
        st.session_state["favorites"].append(selected)
        save_favorites(st.session_state["favorites"])
        st.success(f"Added {selected} to favorites")

try:
    career_df = playercareerstats.PlayerCareerStats(player_id=pid).get_data_frames()[0]
    gamelogs = playergamelogs.PlayerGameLogs(player_id_nullable=pid, season_nullable=CURRENT_SEASON).get_data_frames()[0]
except Exception as e:
    st.error(f"Failed to load player data: {e}")
    st.stop()

img_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
st.image(img_url, width=250)

st.subheader("Career Averages (Per Game)")
career_df["PTS"] = career_df["PTS"]/career_df["GP"]
career_df["REB"] = career_df["REB"]/career_df["GP"]
career_df["AST"] = career_df["AST"]/career_df["GP"]
st.dataframe(career_df[["SEASON_ID","TEAM_ABBREVIATION","PTS","REB","AST"]], width='stretch')

st.subheader("Last Game")
if not gamelogs.empty:
    last = gamelogs.iloc[0]
    cols = st.columns(4)
    for stat in ["PTS","REB","AST","FG3M"]:
        cols.pop(0).metric(stat, last[stat])
else:
    st.info("No game logs found.")

st.subheader("Recent Form (Last 5)")
if len(gamelogs)>=5:
    avg = gamelogs.head(5).mean(numeric_only=True)
    cols = st.columns(4)
    for stat,val in zip(["PTS","REB","AST","FG3M"], [avg["PTS"],avg["REB"],avg["AST"],avg["FG3M"]]):
        cols.pop(0).metric(stat, round(val,2))
else:
    st.info("Not enough games.")

st.subheader("Predicted Next Game (Weighted Avg)")
if not gamelogs.empty:
    w = np.arange(10,0,-1)
    preds = {s: np.average(gamelogs[s].head(10), weights=w) for s in ["PTS","REB","AST","FG3M"]}
    cols = st.columns(4)
    for stat,v in preds.items():
        cols.pop(0).metric(f"{stat} (ML)", f"{v:.1f}")

st.subheader("Stat Breakdown (Last 10 Games)")
df_long = gamelogs.head(10)[["GAME_DATE","PTS","REB","AST","FG3M"]].melt("GAME_DATE", var_name="Stat", value_name="Value")
chart = alt.Chart(df_long).mark_bar().encode(x="Stat:N", y="mean(Value):Q", color="Stat:N").properties(width='stretch', height=300)
st.altair_chart(chart, theme=None)

# ---------------- PNG Snapshot ----------------
st.markdown("---")
st.subheader("📸 Save Dashboard Snapshot")
st.write("Click below to save your current dashboard view as a PNG image.")

if st.button("💾 Save as PNG"):
    try:
        img = ImageGrab.grab()
        save_path = os.path.join(DEFAULT_TMP_DIR, f"{selected.replace(' ','_')}_snapshot.png")
        img.save(save_path)
        with open(save_path, "rb") as f:
            st.download_button("⬇️ Download Snapshot", f, file_name=os.path.basename(save_path))
    except Exception as e:
        st.error(f"Screenshot not supported in this environment. Error: {e}")

st.markdown("---")
st.caption("Hot Shot Props • NBA Analytics Dashboard ©2025")
