# -------------------------------------------------
# HOT SHOT PROPS — NBA AI DASHBOARD (TEAM LOGO + RECENT FORM PATCH)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, playercareerstats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Hot Shot Props | AI NBA Dashboard",
                   page_icon="🏀", layout="wide")

CURRENT_SEASON = "2025-26"
LAST_SEASON = "2024-25"
PRESEASON = "2025 Preseason"

# -------------------------------------------------
# STYLE — black background + translucent team logo
# -------------------------------------------------
st.markdown("""
<style>
body { background-color:#000; color:#F5F5F5; font-family:'Roboto',sans-serif; }
h1,h2,h3,h4 { font-family:'Oswald',sans-serif; color:#ff6f00; text-shadow:0 0 10px #ff9f43; }
.metric-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
               gap:10px; margin-bottom:10px; justify-items:center; }
.metric-card { width:100%; max-width:150px; background:linear-gradient(145deg,#1e1e1e,#121212);
               border:1px solid rgba(255,111,0,0.4); border-radius:10px; padding:10px;
               text-align:center; box-shadow:0 0 10px rgba(255,111,0,0.2); transition:all .3s ease; }
.metric-card:hover{transform:scale(1.04);box-shadow:0 0 16px rgba(255,111,0,0.6);}
.metric-value{font-size:1.1em;font-weight:700;}
.metric-label{font-size:.8em;color:#bbb;}
.conf-low {color:#ff4444;} .conf-med {color:#ffaa00;} .conf-high {color:#00ff80;}
.team-bg {position:fixed;top:0;left:0;width:100%;height:100%;
          background-repeat:no-repeat;background-position:center;
          background-size:500px;opacity:0.06;z-index:-1;}
@media(max-width:600px){.metric-grid{grid-template-columns:repeat(2,1fr);}}
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

def bar_chart(d,title,color="#29b6f6"):
    fig=go.Figure([go.Bar(x=list(d.keys()),y=list(d.values()),marker_color=color,text=[str(v) for v in d.values()],textposition="outside")])
    fig.update_layout(title=title,paper_bgcolor="#000",plot_bgcolor="#000",font_color="#fff")
    st.plotly_chart(fig,use_container_width=True)

def get_player_photo(player_name):
    name = player_name.replace(" ","_").lower()
    url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{name}.png"
    return url

def get_team_logo(team_name):
    tname=team_name.replace(" ","-").lower()
    return f"https://cdn.ssref.net/req/202305101/images/team_logos/{tname}.png"

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.title("🏀 Hot Shot Props — AI NBA Prediction Dashboard")

nba_players=players.get_active_players()
player_list=sorted([p["full_name"] for p in nba_players])
player=st.selectbox("Select a Player",[""]+player_list,index=0)

if not player:
    st.markdown("### 🧠 Select a player to view AI model predictions and performance.")
    st.stop()

pid=next(p["id"] for p in nba_players if p["full_name"]==player)
team_name = "Unknown"
try:
    p_team = teams.find_team_by_player_id(pid)
    if p_team: team_name = p_team['full_name']
except Exception:
    pass

# Background team logo
team_logo_url = get_team_logo(team_name)
st.markdown(f"<div class='team-bg' style='background-image:url({team_logo_url});'></div>", unsafe_allow_html=True)

# Player header
photo_url = get_player_photo(player)
st.image(photo_url,width=180,caption=f"{player} — {team_name}")

# Gather data
cur=enrich(get_games(pid,CURRENT_SEASON))
pre=enrich(get_games(pid,PRESEASON))
last=enrich(get_games(pid,LAST_SEASON))
data=pd.concat([cur,pre,last]).drop_duplicates(subset="Game_ID",keep="first")
if data.empty:
    st.error("No data found for this player.")
    st.stop()
data=data.sort_values("GAME_DATE").reset_index(drop=True)

# Train + predict
models=train_model(data)
data_pred=predict_all(data,models)

# Backtest last game
bt_rows=[]
for i in range(8,len(data)):
    past=data.iloc[:i]
    test=data.iloc[[i]]
    m=train_model(past)
    preds=predict_all(test,m)
    if preds.empty: continue
    preds["Game_ID"]=test["Game_ID"].values[0]
    bt_rows.append(preds)
bt=pd.concat(bt_rows) if bt_rows else pd.DataFrame()

stats=["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]
if not bt.empty:
    latest=bt.iloc[-1]
    actual={s:round(latest[s],1) for s in stats if s in bt.columns}
    pred={s:round(latest.get(f"pred_{s}",np.nan),1) for s in stats}

    st.markdown("## 🔥 Most Recent Game (Actual)")
    metric_cards(actual)

    st.markdown("## 🤖 Model Validation — Actual vs Predicted (Last Game)")
    metric_cards(pred)
    bar_chart({k:pred[k]-actual.get(k,0) for k in pred if k in actual},"Model Deviation (Predicted - Actual)",color="#ff6f00")

# --- Next Game Predictions ---
pred_next={}
if models:
    dfp=prepare(data)
    if not dfp.empty:
        feats=list(next(iter(models.values()))[1])
        X=dfp.iloc[[-1]][feats]
        for s,(m,_) in models.items():
            try:
                pred_next[s]=round(float(m.predict(X)[0]),1)
            except Exception:
                continue

if pred_next:
    st.markdown("---")
    st.markdown("## 🧠 AI Predicted Next Game Stats")
    metric_cards(pred_next)

# --- RECENT FORM METRICS ---
st.markdown("---")
st.markdown("## 📊 Recent Form & Averages")

def avg_stats(df): return {s:round(df[s].mean(),1) for s in stats if s in df.columns}

sections = [
    ("Last 5 Games", data.head(5)),
    ("Last 10 Games", data.head(10)),
    ("Last 20 Games", data.head(20)),
    ("Current Season", cur),
    ("Last Season", last),
    ("Career Totals", playercareerstats.PlayerCareerStats(player_id=pid).get_data_frames()[0])
]

for title, df in sections:
    if df.empty: continue
    st.markdown(f"### {title}")
    averages=avg_stats(df)
    metric_cards(averages)
    bar_chart({k:averages[k] for k in ["PTS","REB","AST","FG3M"] if k in averages},f"{title} — Key Averages")

st.markdown("---")
st.caption("⚡ Hot Shot Props AI Model — trained, validated, and explained © 2025")

