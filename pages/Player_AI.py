import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="NBA Player AI Dashboard", layout="wide", page_icon="🏀")

# -----------------------
# Utility Functions
# -----------------------

@st.cache_data(show_spinner=False)
def get_games(pid, season):
    """Fetch player game logs for a given season."""
    try:
        data = playergamelog.PlayerGameLog(player_id=pid, season=season).get_data_frames()[0]
        return data
    except Exception:
        return pd.DataFrame()

def enrich(df):
    """Add combined stats."""
    if df.empty:
        return df
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df.sort_values("GAME_DATE", inplace=True)
    df["P+R"] = df["PTS"] + df["REB"]
    df["P+A"] = df["PTS"] + df["AST"]
    df["R+A"] = df["REB"] + df["AST"]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

def prepare(df):
    """Add rolling averages as features."""
    df = df.copy()
    for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA","P+R","P+A","R+A"]:
        for w in [3,5]:
            df[f"{s}_avg{w}"] = df[s].rolling(w, min_periods=1).mean()
    return df

def get_player_photo(player_name):
    """Return official NBA headshot."""
    base_url = "https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/"
    player_data = players.find_players_by_full_name(player_name)
    if player_data and "id" in player_data[0]:
        pid = player_data[0]["id"]
        return f"{base_url}{pid}.png"
    return "https://cdn.nba.com/logos/nba/nba-logoman/2022/1x/logo.png"

def metric_cards(stats: dict):
    """Render stats in a 4-column grid."""
    cols = st.columns(4)
    for i, (key, val) in enumerate(stats.items()):
        with cols[i % 4]:
            st.metric(label=key, value=val)

# -----------------------
# Modeling Logic
# -----------------------

def model(df, pid):
    """Train AI model using blended dataset."""
    base = df.copy()

    # Fallbacks if not enough current-season data
    if len(base) < 15:
        try:
            pre = enrich(get_games(pid, "2025 Preseason"))
            base = pd.concat([base, pre])
        except:
            pass
    if len(base) < 15:
        try:
            last = enrich(get_games(pid, "2024-25"))
            base = pd.concat([base, last])
        except:
            pass

    base = prepare(base)
    feats = [c for c in base if "avg" in c]
    models = {}

    for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA","P+R","P+A","R+A"]:
        if s not in base:
            continue
        X = base[feats].dropna()
        y = base.loc[X.index, s].dropna()

        min_len = min(len(X), len(y))
        if min_len < 8:
            continue
        X, y = X.iloc[-min_len:], y.iloc[-min_len:]

        Xtr, Xte, Ytr, Yte = train_test_split(X, y, test_size=0.2, random_state=42)
        m = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        m.fit(Xtr, Ytr)
        models[s] = m

    return models, feats

# -----------------------
# Page Logic
# -----------------------

# Player selection
nba_players = players.get_players()
player_names = sorted([p["full_name"] for p in nba_players])

st.markdown("### 🔍 Search or Select an NBA Player")
selected_player = st.selectbox("Choose a player:", ["-- Select Player --"] + player_names)
player_name = selected_player if selected_player != "-- Select Player --" else None

if not player_name:
    st.info("Please select a player to view AI predictions and stats.")
    st.stop()

# Player info
player_info = next((p for p in nba_players if p["full_name"] == player_name), None)
if not player_info:
    st.error("Player not found.")
    st.stop()

pid = player_info["id"]

# Player header
photo_url = get_player_photo(player_name)
player_team = next((t["full_name"] for t in teams.get_teams() if t["id"] == player_info.get("team_id", None)), "NBA")

st.markdown(
    f"""
    <div style='text-align:center; margin-top:20px;'>
        <img src="{photo_url}" width="200" style='border-radius:10px; margin-bottom:10px;'>
        <h1 style='margin-bottom:0;'>{player_name}</h1>
        <h3 style='color:#999;'>{player_team}</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Data Loading
# -----------------------
current = enrich(get_games(pid, "2025-26"))
pre = enrich(get_games(pid, "2025 Preseason"))
last = enrich(get_games(pid, "2024-25"))

# Blend datasets based on availability
blended = current.copy()
if len(blended) < 20:
    blended = pd.concat([blended, pre])
if len(blended) < 20:
    blended = pd.concat([blended, last])
blended.sort_values("GAME_DATE", inplace=True)

if blended.empty:
    st.error("No data found for this player.")
    st.stop()

# Regular season for recent game display
regular_season = current[current["SEASON_TYPE"] == "Regular Season"] if "SEASON_TYPE" in current else current

# -----------------------
# AI Prediction
# -----------------------
models, feats = model(blended, pid)
if not models:
    st.warning("Not enough data for prediction.")
    st.stop()

latest_feats = prepare(blended).iloc[[-1]][feats]
pred = {s: round(float(models[s].predict(latest_feats)[0]), 1) for s in models if s in models}

st.markdown("## 🧠 AI Predicted Next Game Stats")
metric_cards(pred)

# -----------------------
# Most Recent Game
# -----------------------
if not regular_season.empty:
    latest = regular_season.iloc[-1]
    recent = {
        "PTS": latest["PTS"], "REB": latest["REB"], "AST": latest["AST"], "FG3M": latest["FG3M"],
        "STL": latest["STL"], "BLK": latest["BLK"], "TOV": latest["TOV"],
        "PRA": latest["PRA"], "P+R": latest["P+R"], "P+A": latest["P+A"], "R+A": latest["R+A"]
    }
    st.markdown("### 🔥 Most Recent Regular Season Game Stats")
    metric_cards(recent)
else:
    st.info("No recent regular season games found.")

# -----------------------
# Historical Stats
# -----------------------
def avg_section(title, df, n=None):
    """Display metric cards for selected timeframe."""
    subset = df.tail(n) if n else df
    if subset.empty:
        return
    st.markdown(f"### {title}")
    avg = subset.mean(numeric_only=True).round(1)
    metric_cards({
        "PTS": avg["PTS"], "REB": avg["REB"], "AST": avg["AST"], "FG3M": avg["FG3M"],
        "STL": avg["STL"], "BLK": avg["BLK"], "TOV": avg["TOV"], "PRA": avg["PRA"]
    })

avg_section("📈 Last 5 Games", blended, 5)
avg_section("📈 Last 10 Games", blended, 10)
avg_section("📈 Last 20 Games", blended, 20)
avg_section("📊 Current Season Averages", current)
avg_section("📊 Previous Season Averages", last)
avg_section("🏆 Career Totals", blended)

st.markdown("---")
st.caption("Powered by Hot Shot Props AI — Live NBA Analytics © 2025")
