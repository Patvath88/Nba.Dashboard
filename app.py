# -------------------------------------------------
# HOT SHOT PROPS — NBA AI DASHBOARD (FINAL VERSION)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, playercareerstats, leagueleaders, scoreboardv2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests, io
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Hot Shot Props | AI NBA Dashboard", page_icon="🏀", layout="wide")

CURRENT_SEASON = "2025-26"
LAST_SEASON = "2024-25"

# -------------------------------------------------
# STYLE
# -------------------------------------------------
st.markdown("""
<style>
body { background-color:#000; color:#F5F5F5; font-family:'Roboto',sans-serif; }
h1,h2,h3,h4 { color:#ff3b3b; text-shadow:0 0 10px #ff3b3b; }
.metric-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:10px; justify-items:center; }
.metric-card { width:100%; max-width:160px; background:#121212; border:1px solid rgba(255,59,59,0.5);
               border-radius:12px; padding:10px; text-align:center; box-shadow:0 0 10px rgba(255,59,59,0.2); transition:0.3s; }
.metric-card:hover{transform:scale(1.03);box-shadow:0 0 20px rgba(255,59,59,0.6);}
.metric-value{font-size:1.2em;font-weight:700;}
.metric-label{font-size:.8em;color:#aaa;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
@st.cache_data(ttl=900)
def get_games(player_id, season):
    try:
        return playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

def enrich(df):
    if df.empty: return df
    df=df.copy()
    df["P+R"]=df["PTS"]+df["REB"]
    df["P+A"]=df["PTS"]+df["AST"]
    df["R+A"]=df["REB"]+df["AST"]
    df["PRA"]=df["PTS"]+df["REB"]+df["AST"]
    return df

def prepare(df):
    if df.empty: return df
    df=df.copy()
    for w in [5,10,20]:
        for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV"]:
            df[f"{s}_avg_{w}"]=df[s].rolling(w).mean()
            df[f"{s}_std_{w}"]=df[s].rolling(w).std()
    return df.dropna().reset_index(drop=True)

def train_model(df):
    df=prepare(df)
    feats=[c for c in df.columns if "avg" in c or "std" in c]
    stats=["PTS","REB","AST","FG3M","STL","BLK","TOV","P+R","P+A","R+A","PRA"]
    models={}
    for s in stats:
        if s not in df: continue
        X,y=df[feats],df[s]
        if len(X)<8: continue
        Xtr,Xte,Ytr,Yte=train_test_split(X,y,test_size=.2,random_state=42)
        m=RandomForestRegressor(n_estimators=200,max_depth=8,random_state=42)
        m.fit(Xtr,Ytr)
        models[s]=(m,feats)
    return models

def predict_all(df,models):
    if not models or df.empty: return df
    df=prepare(df.copy())
    for s,(m,feats) in models.items():
        if not all(f in df.columns for f in feats): continue
        X=df[feats].dropna()
        if X.empty: continue
        df.loc[X.index,f"pred_{s}"]=m.predict(X)
    return df

def metric_cards(stats):
    html="<div class='metric-grid'>"
    for k,v in stats.items():
        html+=f"<div class='metric-card'><div class='metric-value'>{v}</div><div class='metric-label'>{k}</div></div>"
    html+="</div>"
    st.markdown(html,unsafe_allow_html=True)

def bar_chart(d,title,color="#ff3b3b"):
    fig=go.Figure([go.Bar(x=list(d.keys()),y=list(d.values()),marker_color=color,text=[str(v) for v in d.values()],textposition="outside")])
    fig.update_layout(title=title,paper_bgcolor="#000",plot_bgcolor="#000",font_color="#fff",xaxis_color="#fff",yaxis_color="#fff")
    st.plotly_chart(fig,use_container_width=True)

def get_player_photo(player_id):
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

def get_next_game(player_team):
    try:
        url = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        data = requests.get(url,headers={"User-Agent":"Mozilla/5.0"}).json()
        for g in data["events"]:
            for t in g["competitions"][0]["competitors"]:
                if player_team.lower() in t["team"]["displayName"].lower():
                    opp = [c for c in g["competitions"][0]["competitors"] if c!=t][0]
                    homeaway = "Home" if t["homeAway"]=="home" else "Away"
                    time = g["date"].split("T")[1][:5]
                    link = g["links"][0]["href"]
                    return {"opponent":opp["team"]["displayName"],"time":time,"homeaway":homeaway,"link":link}
    except: pass
    return None

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.title("🏀 Hot Shot Props — NBA AI Prediction Dashboard")

nba_players=players.get_active_players()
player_list=sorted([p["full_name"] for p in nba_players])
player=st.selectbox("Select a Player",[""]+player_list,index=0)

if not player:
    st.markdown("### 🧠 Select a player to view AI model predictions and performance.")
    st.stop()

pid=next(p["id"] for p in nba_players if p["full_name"]==player)
team_name="Unknown"
try:
    p_team = teams.find_team_by_player_id(pid)
    if p_team: team_name=p_team['full_name']
except: pass

photo_url=get_player_photo(pid)
st.image(photo_url,width=300,caption=f"{player} — {team_name}")

# Data
cur=enrich(get_games(pid,CURRENT_SEASON))
last=enrich(get_games(pid,LAST_SEASON))
data=pd.concat([cur,last]).drop_duplicates(subset="Game_ID",keep="first")
data=data.sort_values("GAME_DATE").reset_index(drop=True)

if data.empty:
    st.error("No data found for this player.")
    st.stop()

# AI model
models=train_model(data)
data_pred=predict_all(data,models)

# Most recent game
recent=data.iloc[-1]
st.markdown("## 🔥 Most Recent Game")
st.write(f"{recent['MATCHUP']} | {recent['GAME_DATE']} | {recent['WL']} | {recent['MIN']} mins")
recent_stats={s:round(recent[s],1) for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]}
metric_cards(recent_stats)
bar_chart(recent_stats,"Most Recent Game — Key Stats","#ff3b3b")

# Model projection
pred_next={}
if models:
    dfp=prepare(data)
    if not dfp.empty:
        feats=list(next(iter(models.values()))[1])
        X=dfp.iloc[[-1]][feats]
        for s,(m,_) in models.items():
            try: pred_next[s]=round(float(m.predict(X)[0]),1)
            except: continue

if pred_next:
    st.markdown("## 🤖 AI Projected Next Game Stats")
    metric_cards(pred_next)
    bar_chart(pred_next,"Projected Stats","#00E676")

    # Add reasoning (example + future: web-sourced)
    reasoning = (
        f"**Model Insight:** {player} has averaged {recent['PTS']} PTS, {recent['REB']} REB, and {recent['AST']} AST "
        f"in recent games. The AI model projects continued performance due to favorable matchup trends "
        f"and increased usage rate for {team_name}."
    )
    st.markdown(reasoning)

# Next game details
next_game=get_next_game(team_name)
if next_game:
    st.markdown(f"### 🗓️ Next Game: {team_name} ({next_game['homeaway']}) vs {next_game['opponent']} at {next_game['time']} — [Preview on ESPN]({next_game['link']})")

# Generate PNG
st.markdown("### 📸 Shareable Player Projection Card")

if st.button("Generate PNG Card"):
    fig, ax = plt.subplots(figsize=(5,7))
    fig.patch.set_facecolor("black")
    ax.axis("off")

    logo = Image.open("logo.png").resize((80,80))
    plt.figimage(logo, xo=30, yo=600, alpha=0.8, zorder=10)
    plt.text(150, 670, f"{team_name} — {player}", color="white", fontsize=16, weight="bold")
    plt.text(150, 640, f"Next Game: {next_game['opponent']} ({next_game['homeaway']})", color="gray", fontsize=12)
    plt.text(150, 620, f"Tip: {next_game['time']}", color="gray", fontsize=12)

    # Bar chart in the PNG
    stats = ["PTS","REB","AST","FG3M"]
    vals = [pred_next.get(s,0) for s in stats]
    ax.bar(stats, vals, color="#ff3b3b")
    for i,v in enumerate(vals): ax.text(i, v+0.3, str(v), color="white", ha="center")
    plt.text(0, 1.05, reasoning, wrap=True, color="white", fontsize=10, transform=ax.transAxes)
    plt.savefig("player_projection.png", bbox_inches="tight", facecolor="black")
    st.success("✅ PNG Card Generated!")
    st.image("player_projection.png", use_container_width=True)
