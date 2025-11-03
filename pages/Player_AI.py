import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

st.set_page_config(page_title="NBA Player AI Dashboard", layout="wide", page_icon="🏀")

# -----------------------
# Utility Functions
# -----------------------

@st.cache_data(show_spinner=False)
def get_games(pid, season):
    """Fetch player game logs for a given season"""
    try:
        data = playergamelog.PlayerGameLog(player_id=pid, season=season).get_data_frames()[0]
        return data
    except Exception:
        return pd.DataFrame()

def enrich(df):
    """Add derived combined stats."""
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
    """Add rolling average features."""
    df = df.copy()
    for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA","P+R","P+A","R+A"]:
        for w in [3,5]:
            df[f"{s}_avg{w}"] = df[s].rolling(w, min_periods=1).mean()
    return df

def model(df, pid):
    """Train AI models; fallback to preseason/last season if limited data."""
    base = df.copy()

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

def metric_cards(stats: dict):
    """Render metrics in compact 4-column layout."""
    cols = st.columns(4)
    keys = list(stats.keys())
    for i, key in enumerate(keys):
        with cols[i % 4]:
            st.metric(label=key, value=stats[key])

def bar_chart(title, df):
    """Display grouped bar chart for major stats."""
    fig = go.Figure()
    colors = {"PTS": "#E50914", "REB": "#00E676", "AST": "#29B6F6", "FG3M": "#FFD700"}
    for col in ["PTS","REB","AST","FG3M"]:
        if col in df:
            fig.add_trace(go.Bar(x=df["GAME_DATE"], y=df[col], name=col, marker_color=colors.get(col, "#FFFFFF")))
    fig.update_layout(title=title, barmode="group", height=300, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Main Page
# -----------------------

# Fetch all NBA players for dropdown
nba_players = players.get_players()
player_names = sorted([p["full_name"] for p in nba_players])

# --- Player Selection ---
st.markdown("### 🔍 Search or Select an NBA Player")
selected_player = st.selectbox("Choose a player:", ["-- Select Player --"] + player_names)

# Determine target player
player_name = selected_player if selected_player != "-- Select Player --" else None

if not player_name:
    st.info("Please select a player to view AI predictions and stats.")
    st.stop()

# Player info lookup
player_info = next((p for p in nba_players if p["full_name"] == player_name), None)
if not player_info:
    st.error("Player not found.")
    st.stop()

pid = player_info["id"]

# -----------------------
# Data Loading
# -----------------------
current = enrich(get_games(pid, "2025-26"))
pre = enrich(get_games(pid, "2025 Preseason"))
last = enrich(get_games(pid, "2024-25"))
data = pd.concat([current, pre, last])

if data.empty:
    st.error("No data found for this player.")
    st.stop()

# -----------------------
# Model Training
# -----------------------
models, feats = model(data, pid)

if not models:
    st.warning("Not enough data for prediction.")
    st.stop()

# -----------------------
# Predictions
# -----------------------
latest_feats = prepare(data).iloc[[-1]][feats]
pred = {s: round(float(models[s].predict(latest_feats)[0]), 1) for s in models if s in models}

st.markdown(f"## 🏀 {player_name} — AI Predicted Next Game Stats")
metric_cards(pred)

# -----------------------
# Recent Game
# -----------------------
latest = data.iloc[-1]
recent = {
    "PTS": latest["PTS"],
    "REB": latest["REB"],
    "AST": latest["AST"],
    "FG3M": latest["FG3M"],
    "STL": latest["STL"],
    "BLK": latest["BLK"],
    "TOV": latest["TOV"],
    "PRA": latest["PRA"],
    "P+R": latest["P+R"],
    "P+A": latest["P+A"],
    "R+A": latest["R+A"],
}

st.markdown("### 🔥 Most Recent Game Stats")
metric_cards(recent)

# -----------------------
# Historical Sections
# -----------------------
st.markdown("### 📈 Last 5 Games")
bar_chart("Last 5 Games — Performance", data.tail(5))

st.markdown("### 📈 Last 10 Games")
bar_chart("Last 10 Games — Performance", data.tail(10))

st.markdown("### 📈 Last 20 Games")
bar_chart("Last 20 Games — Performance", data.tail(20))

# -----------------------
# Season and Career
# -----------------------
st.markdown("### 📊 This Season Averages")
season_avg = data.mean(numeric_only=True).round(1)
metric_cards({
    "PTS": season_avg["PTS"],
    "REB": season_avg["REB"],
    "AST": season_avg["AST"],
    "FG3M": season_avg["FG3M"],
    "STL": season_avg["STL"],
    "BLK": season_avg["BLK"],
    "TOV": season_avg["TOV"],
    "PRA": season_avg["PRA"],
})

st.markdown("### 🏆 Career Totals")
career_totals = data.sum(numeric_only=True).round(0)
metric_cards({
    "PTS": career_totals["PTS"],
    "REB": career_totals["REB"],
    "AST": career_totals["AST"],
    "FG3M": career_totals["FG3M"],
    "STL": career_totals["STL"],
    "BLK": career_totals["BLK"],
    "TOV": career_totals["TOV"],
    "PRA": career_totals["PRA"],
})
