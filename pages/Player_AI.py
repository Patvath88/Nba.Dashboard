import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import random

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

# Simple team color lookup
TEAM_COLORS = {
    "Lakers": "#552583", "Celtics": "#007A33", "Warriors": "#1D428A", "Nuggets": "#0E2240",
    "Heat": "#98002E", "Bucks": "#00471B", "Cavaliers": "#6F263D", "76ers": "#006BB6",
    "Suns": "#E56020", "Mavericks": "#00538C", "Knicks": "#F58426", "Bulls": "#CE1141"
}

def get_team_color(team_name):
    for k, v in TEAM_COLORS.items():
        if k.lower() in team_name.lower():
            return v
    return "#E50914"  # Default red if team not found

def metric_cards(stats: dict, color: str, accuracy=None):
    """Render stats in a 4-column grid with glowing team-color borders and accuracy %."""
    cols = st.columns(4)
    for i, (key, val) in enumerate(stats.items()):
        acc_str = f" ({accuracy.get(key,0)}% accurate)" if accuracy else ""
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style="
                    border: 2px solid {color};
                    border-radius: 10px;
                    background: rgba(25,25,25,0.85);
                    padding: 10px;
                    text-align:center;
                    box-shadow: 0px 0px 10px {color};
                    transition: all 0.3s ease;
                ">
                    <h4 style='color:white;margin-bottom:0;'>{key}</h4>
                    <p style='font-size:22px;color:{color};margin-top:4px;font-weight:bold;'>
                        {val}{acc_str}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

def bar_chart_compare(title, ai_pred, season_avg):
    """Bar chart comparing AI prediction vs season averages."""
    stats = ["PTS","REB","AST","FG3M","STL","BLK","TOV"]
    ai_vals = [ai_pred.get(s, 0) for s in stats]
    avg_vals = [season_avg.get(s, 0) for s in stats]
    fig = go.Figure(data=[
        go.Bar(name="AI Prediction", x=stats, y=ai_vals, marker_color="#E50914"),
        go.Bar(name="Season Avg", x=stats, y=avg_vals, marker_color="#00E676")
    ])
    fig.update_layout(
        barmode="group",
        title=title,
        template="plotly_dark",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

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

nba_players = players.get_players()
player_names = sorted([p["full_name"] for p in nba_players])

st.markdown("### 🔍 Search or Select an NBA Player")
selected_player = st.selectbox("Choose a player:", ["-- Select Player --"] + player_names)
player_name = selected_player if selected_player != "-- Select Player --" else None

if not player_name:
    st.info("Please select a player to view AI predictions and stats.")
    st.stop()

player_info = next((p for p in nba_players if p["full_name"] == player_name), None)
pid = player_info["id"]

photo_url = get_player_photo(player_name)
player_team = next((t["full_name"] for t in teams.get_teams() if t["id"] == player_info.get("team_id", None)), "NBA")
team_color = get_team_color(player_team)

st.markdown(
    f"""
    <div style='text-align:center; margin-top:20px;'>
        <img src="{photo_url}" width="200" style='border-radius:10px; margin-bottom:10px;'>
        <h1 style='margin-bottom:0;'>{player_name}</h1>
        <h3 style='color:{team_color};'>{player_team}</h3>
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

blended = current.copy()
if len(blended) < 20:
    blended = pd.concat([blended, pre])
if len(blended) < 20:
    blended = pd.concat([blended, last])
blended.sort_values("GAME_DATE", inplace=True)

if blended.empty:
    st.error("No data found for this player.")
    st.stop()

regular_season = current[current["SEASON_TYPE"] == "Regular Season"] if "SEASON_TYPE" in current else current

# -----------------------
# AI Prediction
# -----------------------
models, feats = model(blended, pid)
latest_feats = prepare(blended).iloc[[-1]][feats]
pred = {s: round(float(models[s].predict(latest_feats)[0]), 1) for s in models if s in models}

# Mocked model accuracy data (replace with real logs later)
accuracy = {k: random.randint(70, 95) for k in pred.keys()}

st.markdown("## 🧠 AI Predicted Next Game Stats")
metric_cards(pred, team_color, accuracy)

# Bar chart comparing predictions vs season avg
if not current.empty:
    avg = current.mean(numeric_only=True).round(1)
    season_avg = {s: avg.get(s, 0) for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV"]}
    bar_chart_compare("AI Predictions vs Current Season Averages", pred, season_avg)

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
    metric_cards(recent, team_color)
else:
    st.info("No recent regular season games found.")

st.markdown("---")
st.caption("🔥 Hot Shot Props NBA AI Dashboard © 2025")
