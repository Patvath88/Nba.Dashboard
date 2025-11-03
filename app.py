import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog
from PIL import Image
import requests
from io import BytesIO

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Hot Shot Props | NBA Dashboard", page_icon="🔥", layout="wide")

ODDS_API_KEY = "e11d4159145383afd3a188f99489969e"
ASSETS_DIR = "assets"
PLAYER_PHOTO_DIR = os.path.join(ASSETS_DIR, "player_photos")
TEAM_LOGO_DIR = os.path.join(ASSETS_DIR, "team_logos")
os.makedirs(PLAYER_PHOTO_DIR, exist_ok=True)
os.makedirs(TEAM_LOGO_DIR, exist_ok=True)

# -------------------------------------------------
# STYLE
# -------------------------------------------------
st.markdown("""
<style>
body {background-color:#121212;color:#F5F5F5;font-family:'Roboto',sans-serif;}
h1,h2,h3,h4{font-family:'Oswald',sans-serif;color:#E50914;}
.player-card{background:linear-gradient(180deg,#1E1E1E 0%,#191919 100%);
  border-radius:18px;padding:15px;text-align:center;
  transition:transform .3s ease,box-shadow .3s ease;}
.player-card:hover{transform:scale(1.03);box-shadow:0 0 25px rgba(229,9,20,.4);}
.chart-box{background:#1C1C1C;padding:20px;border-radius:15px;margin-top:15px;
  box-shadow:0 0 10px rgba(0,0,0,.3);}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# UTILITIES
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_predictions():
    if os.path.exists("predictions.csv"):
        return pd.read_csv("predictions.csv")
    # fallback sample
    return pd.DataFrame({
        "player_name":["Jayson Tatum","Donovan Mitchell","Luka Doncic"],
        "team":["BOS","CLE","DAL"],
        "prop_type":["Points","Points","Points"],
        "line":[27.5,26.5,31.5],
        "projection":[31.2,28.8,35.1],
        "ev":[0.14,0.08,0.09],
        "confidence":[0.87,0.79,0.83],
        "sportsbook_line":["FanDuel","FanDuel","FanDuel"],
        "edge_value":[14,8,9]
    })

@st.cache_data(ttl=3600)
def get_player_photo(player_name):
    safe = player_name.replace(" ", "_").lower()
    local_path = os.path.join(PLAYER_PHOTO_DIR, f"{safe}.jpg")
    if os.path.exists(local_path):
        return local_path
    try:
        url = f"https://nba-players.herokuapp.com/players/{safe.split('_')[-1]}/{safe.split('_')[0]}"
        img = Image.open(BytesIO(requests.get(url, timeout=5).content))
        img.save(local_path)
        return local_path
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_team_logo(team_abbr):
    safe = team_abbr.lower()
    local_path = os.path.join(TEAM_LOGO_DIR, f"{safe}.png")
    if os.path.exists(local_path):
        return local_path
    try:
        url = f"https://loodibee.com/wp-content/uploads/nba-{safe}-logo.png"
        img = Image.open(BytesIO(requests.get(url, timeout=5).content))
        img.save(local_path)
        return local_path
    except Exception:
        return None

@st.cache_data(ttl=900)
def get_recent_games(player_id):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25')
        df = gamelog.get_data_frames()[0]
        return df.head(5)[["GAME_DATE","PTS","REB","AST"]]
    except Exception:
        return pd.DataFrame(columns=["GAME_DATE","PTS","REB","AST"])

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
# -------------------------------------------------
# LOAD DATA (FIXED)
# -------------------------------------------------
preds = load_predictions()
nba_players = players.get_active_players()
nba_teams = teams.get_teams()

# Build team lookup by ID
team_lookup = {t["id"]: t for t in nba_teams}

# Build team-player map safely
team_map = {}
for p in nba_players:
    team_id = p.get("team_id")
    team_abbr = team_lookup[team_id]["abbreviation"] if team_id in team_lookup else "FA"
    team_map.setdefault(team_abbr, []).append(p["full_name"])

# Create flattened dropdown list
team_options = []
for team, plist in sorted(team_map.items()):
    team_options.append(f"=== {team} ===")
    team_options.extend(sorted(plist))

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🏀 Hot Shot Props — NBA ESPN-Style Dashboard")
st.subheader("AI-Powered Player Prop Insights with Live NBA Data")

# -------------------------------------------------
# PLAYER SEARCH / SELECT (patched)
# -------------------------------------------------
selected_player = st.selectbox(
    "Search or Browse by Team ↓",
    options=team_options,
    index=None,
    placeholder="Select an NBA player"
)

# Stop until a player is chosen
if selected_player is None or selected_player == "":
    st.stop()

# Guard against team header lines
if isinstance(selected_player, str) and selected_player.startswith("==="):
    st.warning("Please select an actual player from the list.")
    st.stop()

metric = st.selectbox("Select Metric", ["Points","Rebounds","Assists","PRA"])

# -------------------------------------------------
# FETCH DATA
# -------------------------------------------------
pinfo = next((p for p in nba_players if p["full_name"] == selected_player), None)
if not pinfo:
    st.error("Player not found in NBA API.")
    st.stop()

player_id = pinfo["id"]
recent = get_recent_games(player_id)
if recent.empty:
    st.warning("No recent game data available.")
    st.stop()

recent["PRA"] = recent["PTS"] + recent["REB"] + recent["AST"]

# -------------------------------------------------
# MODEL MATCH
# -------------------------------------------------
from difflib import get_close_matches

# -------------------------------------------------
# MODEL MATCH (robust fuzzy matching)
# -------------------------------------------------
model_row = pd.DataFrame()
if "player_name" in preds.columns:
    names_lower = preds["player_name"].str.lower().tolist()
    matches = get_close_matches(selected_player.lower(), names_lower, n=1, cutoff=0.6)
    if matches:
        model_row = preds[preds["player_name"].str.lower() == matches[0]]

model_val = None
if not model_row.empty:
    row = model_row.iloc[0]
    if row["prop_type"].lower() == metric.lower():
        model_val = row["projection"]
else:
    row = {}


# -------------------------------------------------
# VISUALS
# -------------------------------------------------
photo = get_player_photo(selected_player)
team_logo = get_team_logo(row.get("team","nba") if row else "nba")

col1, col2 = st.columns([1,2])
with col1:
    if photo: st.image(photo, use_column_width=True)
with col2:
    if team_logo: st.image(team_logo, width=90)
    st.markdown(f"### {selected_player} — {metric}")
    if model_row.empty:
        st.markdown("No model data available for this player.")
    else:
        st.metric("Model Projection", f"{row['projection']:.1f}")
        st.metric("Sportsbook Line", f"{row['line']}")
        st.metric("Edge", f"{row['edge_value']}%")

# -------------------------------------------------
# CHART
# -------------------------------------------------
st.markdown("### Recent Performance (last 5 games)")

# map display name to the correct dataframe column
col_map = {
    "Points": "PTS",
    "Rebounds": "REB",
    "Assists": "AST",
    "PRA": "PRA"
}
col_name = col_map.get(metric, "PTS")

x = recent["GAME_DATE"].iloc[::-1]
y_actual = recent[col_name].iloc[::-1]

fig = go.Figure()
fig.add_trace(go.Bar(x=x, y=y_actual, name="Actual", marker_color="#E50914"))

if model_val:
    fig.add_trace(go.Scatter(x=x, y=[model_val]*len(x),
                             name="Model Projection",
                             mode="lines", line=dict(color="#00E676", dash="dash")))

fig.update_layout(paper_bgcolor="#121212", plot_bgcolor="#121212",
                  font_color="#F5F5F5", legend=dict(orientation="h", yanchor="bottom"),
                  yaxis_title=metric)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# INSIGHT SUMMARY
# -------------------------------------------------
if not model_row.empty:
    line = row["line"]; proj = row["projection"]; edge = row["edge_value"]; conf = row["confidence"]
    status = "🔥 Over trend" if proj > line else "🧊 Under risk"
    st.info(f"{status}: projected **{proj:.1f} {metric}** vs line **{line}**  "
            f"({edge:+.0f}% edge, confidence {conf*100:.0f}%)")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(f"""
---
<div style='text-align:center;color:#777;font-size:13px;margin-top:20px;'>
Hot Shot Props © {datetime.now().year} | Powered by AI Sports Analytics
</div>
""", unsafe_allow_html=True)
