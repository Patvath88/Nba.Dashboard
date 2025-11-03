# -------------------------------------------------
# HOT SHOT PROPS — NBA PLAYER MODEL DASHBOARD (PAGE)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests

st.set_page_config(page_title="Hot Shot Props | Player AI Model",
                   page_icon="📊", layout="wide")

CURRENT_SEASON = "2025-26"
LAST_SEASON = "2024-25"
PRESEASON = "2025 Preseason"

# -------------------------------------------------
# STYLE
# -------------------------------------------------
st.markdown("""
<style>
body { background-color:#000; color:#F5F5F5; font-family:'Roboto',sans-serif; }
h1,h2,h3,h4 { font-family:'Oswald',sans-serif; color:#ff6f00; text-shadow:0 0 10px #ff9f43; }
.metric-grid {
    display:grid;
    grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
    gap:10px;
    margin-bottom:10px;
    justify-items:center;
}
.metric-card {
    width:100%;
    max-width:150px;
    background:linear-gradient(145deg,#1e1e1e,#121212);
    border:1px solid rgba(255,111,0,0.4);
    border-radius:10px;
    padding:10px;
    text-align:center;
    box-shadow:0 0 10px rgba(255,111,0,0.2);
    transition:all .3s ease;
}
.metric-card:hover{transform:scale(1.04);box-shadow:0 0 16px rgba(255,111,0,0.6);}
.metric-value{font-size:1.1em;font-weight:700;}
.metric-label{font-size:.8em;color:#bbb;}
.player-header { display:flex;align-items:center;gap:20px;margin-bottom:20px; }
.player-img { width:120px;height:120px;border-radius:50%;border:3px solid #ff6f00;object-fit:cover; }
.team-logo { width:80px;height:80px;object-fit:contain; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
@st.cache_data(ttl=600)
def get_games(player_id, season):
    try:
        log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        return log.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

def enrich(df):
    if df.empty: return df
    df = df.copy()
    df["P+R"] = df["PTS"] + df["REB"]
    df["P+A"] = df["PTS"] + df["AST"]
    df["R+A"] = df["REB"] + df["AST"]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

@st.cache_data(ttl=600)
def prepare(df):
    if df.empty: return df
    for w in [5,10,20]:
        for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV"]:
            df.loc[:, f"{s}_avg_{w}"] = df[s].rolling(w).mean()
            df.loc[:, f"{s}_std_{w}"] = df[s].rolling(w).std()
    return df.dropna().reset_index(drop=True)

def train_model(df):
    df = prepare(df)
    feats = [c for c in df.columns if "avg" in c or "std" in c]
    stats = ["PTS","REB","AST","FG3M","STL","BLK","TOV","P+R","P+A","R+A","PRA"]
    models = {}
    for s in stats:
        if s not in df: continue
        X, y = df[feats], df[s]
        if len(X) < 8: continue
        Xtr, Xte, Ytr, Yte = train_test_split(X, y, test_size=.2, random_state=42)
        m = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        m.fit(Xtr, Ytr)
        models[s] = (m, feats)
    return models

def metric_cards(stats):
    html = "<div class='metric-grid'>"
    for k,v in stats.items():
        html += f"<div class='metric-card'><div class='metric-value'>{v}</div><div class='metric-label'>{k}</div></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def bar_compare(actual, pred, title):
    stats = list(actual.keys())
    a_vals = [actual[s] for s in stats]
    p_vals = [pred.get(s, 0) for s in stats]
    fig = go.Figure()
    fig.add_bar(x=stats, y=a_vals, name="Last Game", marker_color="#ff6f00")
    fig.add_bar(x=stats, y=p_vals, name="Model Prediction", marker_color="#00bcd4", opacity=.7)
    fig.update_layout(title=title, barmode="group",
                      paper_bgcolor="#000", plot_bgcolor="#000", font_color="#F5F5F5")
    st.plotly_chart(fig, width="stretch")

# -------------------------------------------------
# MAIN
# -------------------------------------------------
st.title("📊 Hot Shot Props — Player AI Model")
st.caption("Next-game predictions, recent form, and player insights.")
if st.button("🏠 Back to Home"):
    st.switch_page("Home.py")

nba_players = players.get_active_players()
player_list = sorted([p["full_name"] for p in nba_players])
player = st.selectbox("Select a Player", ["Select a player..."] + player_list)

if player == "Select a player...":
    st.info("👈 Choose a player from the dropdown to view predictions.")
    st.stop()

pid = next(p["id"] for p in nba_players if p["full_name"] == player)

# header
st.markdown("<div class='player-header'>", unsafe_allow_html=True)
photo_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
st.image(photo_url, width=120)
st.markdown("</div>", unsafe_allow_html=True)

# data prep
cur = enrich(get_games(pid, CURRENT_SEASON))
pre = enrich(get_games(pid, PRESEASON))
last = enrich(get_games(pid, LAST_SEASON))
data = pd.concat([cur, pre, last]).drop_duplicates(subset="Game_ID", keep="first")
if data.empty:
    st.error("No game data found.")
    st.stop()
data = data.sort_values("GAME_DATE").reset_index(drop=True)

# model
models = train_model(data)

# next game prediction
pred_next = {}
if models:
    dfp = prepare(data)
    if not dfp.empty:
        feats = list(next(iter(models.values()))[1])
        X = dfp.iloc[[-1]][feats]
        for s, (m, _) in models.items():
            val = float(m.predict(X)[0])
            pred_next[s] = round(val, 1)

if pred_next:
    st.subheader("🧠 AI Predicted Next Game Stats")
    metric_cards(pred_next)

# last game stats
if not data.empty:
    last_game = data.iloc[0]
    recent = {s: round(last_game[s], 1) for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]}
    st.subheader("🔥 Most Recent Game Performance")
    metric_cards(recent)
    if pred_next:
        bar_compare(recent, pred_next, "Comparison: Last Game vs AI Prediction")
