# -------------------------------------------------
# HOT SHOT PROPS — PLAYER AI DASHBOARD
# -------------------------------------------------
import streamlit as st, pandas as pd, numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Hot Shot Props | Player AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
body{background:#000;color:#EEE;font-family:'Roboto',sans-serif;}
h1,h2,h3{color:#FF6F00;text-shadow:0 0 10px #FF8C00;}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin:10px 0;}
.metric{background:#1A1A1A;border:1px solid #FF6F00;border-radius:10px;text-align:center;padding:10px;}
.metric:hover{box-shadow:0 0 12px #FF6F00;}
.player-header{display:flex;align-items:center;gap:20px;margin-bottom:20px;}
.player-img{width:140px;height:140px;border-radius:50%;border:3px solid #FF6F00;object-fit:cover;}
</style>
""", unsafe_allow_html=True)

CURRENT="2025-26"; LAST="2024-25"; PRE="2025 Preseason"

@st.cache_data(ttl=600)
def get_games(pid,season):
    try:return playergamelog.PlayerGameLog(player_id=pid,season=season).get_data_frames()[0]
    except:return pd.DataFrame()

def enrich(df):
    if df.empty:return df
    df["PRA"]=df["PTS"]+df["REB"]+df["AST"]
    df["P+R"]=df["PTS"]+df["REB"]; df["P+A"]=df["PTS"]+df["AST"]; df["R+A"]=df["REB"]+df["AST"]
    return df

def prepare(df):
    for w in [5,10,20]:
        for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV"]:
            df[f"{s}_avg_{w}"]=df[s].rolling(w).mean()
    return df.dropna()

def model(df):
    df = prepare(df)
    feats = [c for c in df if "avg" in c]
    models = {}

    for s in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "PRA", "P+R", "P+A", "R+A"]:
        if s not in df:
            continue

        # Drop NaNs and align X/y properly
        X = df[feats].dropna()
        y = df.loc[X.index, s].dropna()

        # Make sure both have the same length
        min_len = min(len(X), len(y))
        if min_len < 8:
            continue
        X, y = X.iloc[-min_len:], y.iloc[-min_len:]

        Xtr, Xte, Ytr, Yte = train_test_split(X, y, test_size=0.2, random_state=42)
        m = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        m.fit(Xtr, Ytr)
        models[s] = m

    return models, feats


def metric_cards(stats):
    html="<div class='metric-grid'>"
    for k,v in stats.items():
        html+=f"<div class='metric'><div style='font-size:1.3em;font-weight:bold'>{v}</div><div>{k}</div></div>"
    html+="</div>"; st.markdown(html,unsafe_allow_html=True)

if st.button("🏠 Back to Home"): st.switch_page("Home.py")

players_list=players.get_active_players()
names=sorted([p["full_name"] for p in players_list])
query=st.experimental_get_query_params().get("player",[None])[0]
player=st.selectbox("Select a Player",["Select a player..."]+names,index=names.index(query)+1 if query in names else 0)
if player=="Select a player...": st.stop()

pid=next(p["id"] for p in players_list if p["full_name"]==player)
photo=f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
st.markdown(f"<div class='player-header'><img src='{photo}' class='player-img'><h2>{player}</h2></div>",unsafe_allow_html=True)

df=pd.concat([enrich(get_games(pid,CURRENT)),enrich(get_games(pid,PRE)),enrich(get_games(pid,LAST))])
if df.empty: st.error("No data found."); st.stop()
df=df.sort_values("GAME_DATE",ascending=False).reset_index(drop=True)

models,feats=model(df)
if models:
    latest=prepare(df).iloc[[-1]][feats]
    preds={s:round(float(m.predict(latest)[0]),1) for s,m in models.items()}
    st.subheader("🧠 AI Predicted Next Game Stats"); metric_cards(preds)

last=df.iloc[0]; recent={s:round(last[s],1) for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]}
st.subheader("🔥 Most Recent Game"); metric_cards(recent)

for label,n in [("📅 Last 5 Games",5),("📅 Last 10 Games",10),("📅 Last 20 Games",20)]:
    sub=df.head(n); avg={s:round(sub[s].mean(),1) for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]}
    st.markdown(f"### {label}"); metric_cards(avg)

this=df.head(30); lastS=enrich(get_games(pid,LAST))
if not this.empty: st.markdown("### 📊 This Season Avg"); metric_cards({s:round(this[s].mean(),1) for s in ["PTS","REB","AST","PRA"]})
if not lastS.empty: st.markdown("### 📊 Last Season Avg"); metric_cards({s:round(lastS[s].mean(),1) for s in ["PTS","REB","AST","PRA"]})
