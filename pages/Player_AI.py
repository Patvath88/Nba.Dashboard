import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
import requests
from io import BytesIO

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Player AI", layout="wide")
st.markdown("<style>body{background-color:black;color:white;}</style>", unsafe_allow_html=True)
team_color = "#E50914"

# ---------------------- UTILITIES ----------------------
@st.cache_data(show_spinner=False)
def get_games(player_id, season):
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        gl = gl.sort_values("GAME_DATE")
        return gl
    except Exception:
        return pd.DataFrame()

def enrich(df):
    if df.empty:
        return df
    df["P+R"] = df["PTS"] + df["REB"]
    df["P+A"] = df["PTS"] + df["AST"]
    df["R+A"] = df["REB"] + df["AST"]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

def get_player_photo(name):
    try:
        url = f"https://nba-players-directory.vercel.app/api/player/{name.replace(' ', '_')}.png"
        resp = requests.get(url)
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content))
    except Exception:
        pass
    return None

# ---------------------- HEADER ----------------------
nba_players = players.get_active_players()
player_list = sorted([p["full_name"] for p in nba_players])
player = st.selectbox("Search or Browse Player", [""] + player_list)

if not player:
    st.warning("Select a player from the dropdown above.")
    st.stop()

pid = next(p["id"] for p in nba_players if p["full_name"] == player)

CURRENT_SEASON = "2025-26"
PREVIOUS_SEASON = "2024-25"

# ---------------------- DATA ----------------------
current = enrich(get_games(pid, CURRENT_SEASON))
last = enrich(get_games(pid, PREVIOUS_SEASON))
blended = pd.concat([current, last], ignore_index=True)

# ---------------------- MODEL ----------------------
def train_model(df):
    if df is None or len(df) < 5:
        return None
    X = np.arange(len(df)).reshape(-1, 1)
    y = df.values
    min_len = min(len(X), len(y))
    X, y = X[:min_len], y[:min_len]
    m = RandomForestRegressor(n_estimators=150, random_state=42)
    m.fit(X, y)
    return m

def predict_next(df):
    if df is None or len(df) < 3:
        return 0
    X_pred = np.array([[len(df)]])
    m = train_model(df)
    return round(float(m.predict(X_pred)), 1) if m else 0

pred_next = {}
if not current.empty:
    for stat in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA","P+R","P+A","R+A"]:
        pred_next[stat] = predict_next(current[stat])
else:
    for stat in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA","P+R","P+A","R+A"]:
        pred_next[stat] = 0

# ---------------------- METRIC CARDS ----------------------
def metric_cards(stats: dict, color: str, accuracy=None, predictions=False):
    """Render stats in a 4-column grid with dark gray background and red text."""
    cols = st.columns(4)
    for i, (key, val) in enumerate(stats.items()):
        acc_str = ""
        if accuracy and predictions:
            acc_val = accuracy.get(key, 0)
            acc_str = f"<div style='font-size:13px; color:gray; font-style:italic;'>(Accuracy: {acc_val}%)</div>"

        # Fix: explicitly convert val to text, not HTML
        val_str = str(val)

        card_html = f"""
        <div style="
            border: 2px solid {color};
            border-radius: 10px;
            background-color: #1e1e1e;
            padding: 14px;
            text-align:center;
            color:{color};
            box-shadow: 0px 0px 10px {color};
        ">
            <h4 style='margin-bottom:2px;'>{key}</h4>
            {acc_str}
            <div style='font-size:30px;font-weight:bold;margin-top:6px;'>{val_str}</div>
        </div>
        """
        with cols[i % 4]:
            st.markdown(card_html, unsafe_allow_html=True)

# ---------------------- BAR CHART ----------------------
def bar_chart_recent(title, df):
    if df.empty:
        return
    stats = ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]
    avg = df[stats].mean(numeric_only=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=avg.index, y=avg.values, marker_color=team_color))
    fig.update_layout(
        title=title,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=300,
        margin=dict(l=30, r=30, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width='stretch')

# ---------------------- LAYOUT ----------------------
photo = get_player_photo(player)
col1, col2 = st.columns([1, 3])
with col1:
    if photo:
        st.image(photo, width=180)
with col2:
    st.markdown(f"## **{player}**")
    st.markdown(f"**Team:** (Auto-detected)")
st.markdown("---")

# ---------------------- AI PREDICTION ----------------------
st.markdown("## 🧠 AI Predicted Next Game Stats")
metric_cards(pred_next, team_color, predictions=True)
bar_chart_recent("AI Prediction vs. Season Average", current)

# ---------------------- MOST RECENT GAME ----------------------
if not current.empty:
    recent = current.iloc[-1]
    st.markdown("## 🔥 Most Recent Regular Season Game Stats")
    stats_recent = {
        "PTS": int(recent.get("PTS", 0)), "REB": int(recent.get("REB", 0)), "AST": int(recent.get("AST", 0)),
        "FG3M": int(recent.get("FG3M", 0)), "STL": int(recent.get("STL", 0)), "BLK": int(recent.get("BLK", 0)),
        "TOV": int(recent.get("TOV", 0)), "PRA": int(recent.get("PRA", 0)),
        "P+R": int(recent.get("P+R", 0)), "P+A": int(recent.get("P+A", 0)), "R+A": int(recent.get("R+A", 0))
    }
    metric_cards(stats_recent, team_color)
    bar_chart_recent("Most Recent Game Breakdown", pd.DataFrame([recent]))
else:
    st.info("No recent game data available.")

# ---------------------- HISTORICAL PERFORMANCE ----------------------
st.markdown("### 📊 Player Historical Performance")
timeframe = st.selectbox(
    "Select your time frame to view player's historical stats:",
    ["", "Last 5 Games", "Last 10 Games", "Last 20 Games",
     "Current Season Averages", "Previous Season Averages",
     "Career Averages", "Career Totals"],
    index=0
)

if timeframe:
    if timeframe == "Last 5 Games":
        df = blended.tail(5)
        title = "📈 Last 5 Games"
    elif timeframe == "Last 10 Games":
        df = blended.tail(10)
        title = "📈 Last 10 Games"
    elif timeframe == "Last 20 Games":
        df = blended.tail(20)
        title = "📈 Last 20 Games"
    elif timeframe == "Current Season Averages":
        df = current
        title = "📊 Current Season Averages"
    elif timeframe == "Previous Season Averages":
        df = last
        title = "📊 Previous Season Averages"
    elif timeframe == "Career Averages":
        df = blended
        title = "🏆 Career Averages"
    else:
        df = blended
        title = "🏆 Career Totals"

    if not df.empty:
        st.markdown(f"### {title}")
        avg = df.mean(numeric_only=True).round(1)
        metric_cards({
            "PTS": avg.get("PTS", 0), "REB": avg.get("REB", 0),
            "AST": avg.get("AST", 0), "FG3M": avg.get("FG3M", 0),
            "STL": avg.get("STL", 0), "BLK": avg.get("BLK", 0),
            "TOV": avg.get("TOV", 0), "PRA": avg.get("PRA", 0)
        }, team_color)
        bar_chart_recent(f"{title} — Performance Overview", df)
    else:
        st.info("No data available for this timeframe.")
