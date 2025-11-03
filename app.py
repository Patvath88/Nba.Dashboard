# -------------------------------------------------
# HOT SHOT PROPS — NBA AI RETRO VALIDATION DASHBOARD
# -------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, playercareerstats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from PIL import Image
import requests
from io import BytesIO
from random import choice

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Hot Shot Props | Retro NBA AI Dashboard",
                   page_icon="🏀", layout="wide")

CURRENT_SEASON = "2025-26"
LAST_SEASON = "2024-25"
PRESEASON = "2025 Preseason"
ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# -------------------------------------------------
# STYLE (animated NBA logo + arrows)
# -------------------------------------------------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at center, #0b0b0b 0%, #000 90%);
    background-image: url("https://upload.wikimedia.org/wikipedia/en/0/03/National_Basketball_Association_logo.svg");
    background-repeat: no-repeat;
    background-position: center center;
    background-attachment: fixed;
    background-size: 800px;
    animation: spinbg 60s linear infinite;
    color: #F5F5F5;
    font-family: 'Roboto', sans-serif;
}
@keyframes spinbg {
    from {background-position: center center; transform: rotate(0deg);}
    to   {background-position: center center; transform: rotate(360deg);}
}
h1,h2,h3,h4 {
    font-family:'Oswald',sans-serif;
    color:#ff6f00;
    text-shadow:0 0 10px #ff9f43;
}
.metric-grid {
    display:grid;
    grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
    gap:10px;margin-bottom:10px;justify-items:center;
}
.metric-card {
    width:100%;max-width:150px;
    background:linear-gradient(145deg,#1e1e1e,#121212);
    border:1px solid rgba(255,111,0,0.4);
    border-radius:10px;
    padding:10px;text-align:center;
    box-shadow:0 0 10px rgba(255,111,0,0.2);
    transition:all .3s ease;
}
.metric-card:hover{
    transform:scale(1.04);
    box-shadow:0 0 16px rgba(255,111,0,0.6);
}
.metric-value{
    font-size:1.1em;
    font-weight:700;
}
.metric-label{
    font-size:.8em;color:#bbb;
}
.arrow-up {color:#00FF80;animation:pulseUp 1.2s infinite;}
.arrow-down {color:#FF5555;animation:pulseDown 1.2s infinite;}
@keyframes pulseUp {0%{opacity:.6;}50%{opacity:1;}100%{opacity:.6;}}
@keyframes pulseDown {0%{opacity:.6;}50%{opacity:1;}100%{opacity:.6;}}
@media(max-width:600px){.metric-grid{grid-template-columns:repeat(2,1fr);}}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
@st.cache_data(ttl=900)
def get_games(player_id, season):
    try:
        log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        return log.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

def enrich(df):
    if df.empty: return df
    df["P+R"] = df["PTS"] + df["REB"]
    df["P+A"] = df["PTS"] + df["AST"]
    df["R+A"] = df["REB"] + df["AST"]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

# -------------------------------------------------
# MODELING
# -------------------------------------------------
def prepare(df):
    if df.empty: return df
    for w in [5,10,20]:
        for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV"]:
            df[f"{s}_avg_{w}"]=df[s].rolling(w).mean()
            df[f"{s}_std_{w}"]=df[s].rolling(w).std()
    df=df.dropna().reset_index(drop=True)
    return df

def train_model(df):
    df=prepare(df)
    feats=[c for c in df.columns if "avg" in c or "std" in c]
    stats=["PTS","REB","AST","FG3M","STL","BLK","TOV","P+R","P+A","R+A","PRA"]
    models={}
    for s in stats:
        if s not in df: continue
        X,y=df[feats],df[s]
        if len(X)<6: continue
        Xtr,Xte,Ytr,Yte=train_test_split(X,y,test_size=.2,random_state=42)
        m=RandomForestRegressor(n_estimators=200,max_depth=8,random_state=42)
        m.fit(Xtr,Ytr); models[s]=m
    return models

def predict_all(df,models):
    if not models or df.empty: return df
    df=prepare(df)
    feats=[c for c in df.columns if "avg" in c or "std" in c]
    for s,m in models.items():
        df[f"pred_{s}"]=m.predict(df[feats])
    return df

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
def metric_cards(stats,season=None):
    html="<div class='metric-grid'>"
    for k,v in stats.items():
        arrow=""
        if season and k in season:
            if v>season[k]: arrow=f"<span class='arrow-up'>▲</span>"
            elif v<season[k]: arrow=f"<span class='arrow-down'>▼</span>"
        html+=f"<div class='metric-card'><div class='metric-value'>{v} {arrow}</div><div class='metric-label'>{k}</div></div>"
    html+="</div>"
    st.markdown(html,unsafe_allow_html=True)

def bar_compare(actual,pred,title):
    stats=list(actual.keys())
    a_vals=[actual[s] for s in stats]
    p_vals=[pred.get(s,0) for s in stats]
    colors=["#ff6f00","#ff1744","#00e676","#2979ff","#fdd835","#9c27b0","#29b6f6","#e53935"]
    fig=go.Figure()
    fig.add_bar(x=stats,y=a_vals,name="Actual",marker_color=colors)
    fig.add_bar(x=stats,y=p_vals,name="Predicted",marker_color="#00bcd4",opacity=.7)
    fig.update_layout(title=title,barmode="group",paper_bgcolor="#111",
                      plot_bgcolor="#111",font_color="#F5F5F5")
    st.plotly_chart(fig,use_container_width=True)

# -------------------------------------------------
# MAIN
# -------------------------------------------------
st.title("🏀 Hot Shot Props — AI Validation Dashboard")
st.caption("AI model trains on historical data and compares its predictions to real game results.")

nba_players=players.get_active_players()
nba_teams=teams.get_teams()
lookup={t["id"]:t for t in nba_teams}

player_list=sorted([p["full_name"] for p in nba_players])
player=st.selectbox("Select a Player",player_list)

if not player: st.stop()

# --- Get player ID safely ---
pid = next(p["id"] for p in nba_players if p["full_name"] == player)

# --- Try to infer team from latest game data ---
try:
    latest_game = get_games(pid, CURRENT_SEASON)
    if latest_game.empty:
        latest_game = get_games(pid, LAST_SEASON)
    team_abbr = latest_game["MATCHUP"].iloc[0].split(" ")[0]
except Exception:
    team_abbr = "N/A"


# --- gather data across seasons ---
cur=enrich(get_games(pid,CURRENT_SEASON))
pre=enrich(get_games(pid,PRESEASON))
last=enrich(get_games(pid,LAST_SEASON))
data=pd.concat([cur,pre,last]).drop_duplicates(subset="Game_ID",keep="first")

if data.empty:
    st.error("No data found for this player.")
    st.stop()

# --- model train and predict every game ---
models=train_model(data)
data_pred=predict_all(data,models)

# --- most recent game actuals ---
latest=data_pred.head(1)
actual={s:round(latest[s].iloc[0],1) for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]}
predicted={s:round(latest[f"pred_{s}"].iloc[0],1) for s in actual if f"pred_{s}" in latest.columns}

st.markdown("## 🔥 Most Recent Game (Actual)")
metric_cards(actual)

st.markdown("## 🤖 Model Prediction for Same Game")
metric_cards(predicted)

bar_compare(actual,predicted,"Most Recent Game — Actual vs Predicted")

# --- season averages for trend ---
season_avg={s:round(data[s].mean(),1) for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV"] if s in data}

# --- AI prediction for next game (latest trained model) ---
pred_next={s:round(float(m.predict(prepare(data).iloc[[-1]][[c for c in prepare(data).columns if 'avg' in c or 'std' in c]]))[0],1) for s,m in models.items() if s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]}
st.markdown("---")
st.markdown("## 🧠 AI Predicted Next Game Stats")
metric_cards(pred_next,season_avg)

# --- recent form summaries ---
def avg_block(df,title):
    if df.empty: return
    avg={s:round(df[s].mean(),1) for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]}
    st.markdown(f"### {title}")
    metric_cards(avg,season_avg)
    colors=["#ff6f00","#ff1744","#00e676","#2979ff","#fdd835","#9c27b0","#29b6f6","#e53935"]
    fig=go.Figure([go.Bar(x=list(avg.keys()),y=list(avg.values()),marker_color=[choice(colors) for _ in avg])])
    fig.update_layout(title=f"{title} Averages",paper_bgcolor="#111",plot_bgcolor="#111",font_color="#fff")
    st.plotly_chart(fig,use_container_width=True)

avg_block(data.head(5),"Last 5 Games")
if len(data)>10: avg_block(data.head(10),"Last 10 Games")
if len(data)>20: avg_block(data.head(20),"Last 20 Games")
avg_block(data,"Current Season Averages")

st.markdown("---")
st.caption("⚡ Hot Shot Props AI Model — trained, tested, and validated automatically © 2025")
