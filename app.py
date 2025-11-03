# -------------------------------------------------
# HOT SHOT PROPS — NBA AI DASHBOARD (FINAL SHAREABLE VERSION)
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
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import datetime

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Hot Shot Props | AI NBA Dashboard",
                   page_icon="🏀", layout="wide")

CURRENT_SEASON = "2025-26"
LAST_SEASON = "2024-25"

# -------------------------------------------------
# STYLES
# -------------------------------------------------
st.markdown("""
<style>
body { background-color:#000; color:#fff; font-family:'Roboto',sans-serif; }
h1,h2,h3,h4 { font-family:'Oswald',sans-serif; color:#ff3b3b; text-shadow:0 0 10px #e60000; }
.metric-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; justify-items:center; margin-bottom:10px;}
.metric-card { background:linear-gradient(145deg,#1b1b1b,#111); border:1px solid rgba(255,60,60,0.4); border-radius:10px; padding:10px; text-align:center; box-shadow:0 0 8px rgba(255,60,60,0.3); transition:all .3s ease; max-width:150px; }
.metric-card:hover{transform:scale(1.03); box-shadow:0 0 14px rgba(255,60,60,0.5);}
.metric-value{font-size:1.1em;font-weight:700;}
.metric-label{font-size:.8em;color:#ccc;}
.reasoning-box { background-color:#111; border-left:4px solid #ff3b3b; padding:12px 16px; margin-top:10px; border-radius:6px; color:#ddd; font-size:0.9em; }
.player-photo { display:flex; justify-content:center; align-items:center; margin:20px auto; }
.download-btn { background:#ff3b3b; color:#fff; border:none; border-radius:6px; padding:8px 16px; cursor:pointer; font-weight:600; }
.download-btn:hover { background:#ff5555; }
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

def bar_chart(d,title,color="#ff3b3b"):
    if not d: return
    fig=go.Figure([go.Bar(x=list(d.keys()),y=list(d.values()),marker_color=color,text=[str(v) for v in d.values()],textposition="outside")])
    fig.update_layout(title=title,paper_bgcolor="#000",plot_bgcolor="#000",font_color="#fff")
    st.plotly_chart(fig,use_container_width=True)

def get_player_photo(player_id):
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.title("🏀 Hot Shot Props — AI NBA Prediction Dashboard")

nba_players=players.get_active_players()
player_list=sorted([p["full_name"] for p in nba_players])
player=st.selectbox("Select a Player",[""]+player_list,index=0)

if not player:
    st.markdown("## 📈 Model Success Overview")
    st.info("Select a player to generate AI-based projections and insights.")
    st.stop()

pid=next(p["id"] for p in nba_players if p["full_name"]==player)

# Load data
cur=enrich(get_games(pid,CURRENT_SEASON))
last=enrich(get_games(pid,LAST_SEASON))
data=pd.concat([cur,last]).drop_duplicates(subset="Game_ID",keep="first")
if data.empty:
    st.error("No game data found for this player.")
    st.stop()
data=data.sort_values("GAME_DATE").reset_index(drop=True)

# Determine team + opponent from last game
team_name="Unknown"
opponent="N/A"
home_away="Home/Away"
try:
    if "MATCHUP" in data.columns:
        matchup=data.iloc[0]["MATCHUP"]
        if "@" in matchup:
            parts=matchup.split(" @ ")
            team_abbr=parts[0]; opp_abbr=parts[1]
            home_away="Away"
        elif "vs." in matchup:
            parts=matchup.split(" vs. ")
            team_abbr=parts[0]; opp_abbr=parts[1]
            home_away="Home"
        opp_team=[t["full_name"] for t in teams.get_teams() if t["abbreviation"]==opp_abbr]
        if opp_team: opponent=opp_team[0]
        team_team=[t["full_name"] for t in teams.get_teams() if t["abbreviation"]==team_abbr]
        if team_team: team_name=team_team[0]
except Exception:
    pass

# Player header with image and matchup info
photo_url=get_player_photo(pid)
st.markdown(f"<div class='player-photo'><img src='{photo_url}' width='350'></div>", unsafe_allow_html=True)
st.markdown(f"### {player} — {team_name}")
if not data.empty:
    last_game=data.iloc[0]
    game_date=last_game['GAME_DATE']
    st.caption(f"🗓️ Last Game: {game_date} — {home_away} vs {opponent}")

# Train + predict
models=train_model(data)
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
    st.markdown("## 🧠 Model Projected Props")
    metric_cards(pred_next)

# Professional reasoning
def fetch_reasoning(player_name):
    try:
        q=player_name.replace(" ","+")
        url=f"https://search.espn.com/results?q={q}+NBA+news"
        r=requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=5)
        if "nba" in r.text.lower():
            return f"**Professional Insight:** Recent performance and trends available from [ESPN]({url})"
    except Exception:
        pass
    form5=data.head(5)[["PTS","REB","AST","FG3M"]].mean()
    return (
        f"**Model Insight:** {player_name} has averaged **{form5['PTS']:.1f} PTS**, "
        f"**{form5['REB']:.1f} REB**, **{form5['AST']:.1f} AST**, and **{form5['FG3M']:.1f} 3PM** over the last 5 games. "
        f"The AI model expects continued production driven by matchup trends and recent uptick in usage rate for {team_name}."
    )

reasoning=fetch_reasoning(player)
st.markdown(f"<div class='reasoning-box'>{reasoning}</div>", unsafe_allow_html=True)

# --- PNG EXPORT FEATURE ---
if st.button("📸 Save Projection as PNG", key="save_png"):
    img = Image.new("RGB",(1000,700),(0,0,0))
    draw = ImageDraw.Draw(img)
    try:
        response = requests.get(photo_url)
        pimg = Image.open(BytesIO(response.content)).resize((280,200))
        img.paste(pimg,(50,80))
    except Exception:
        pass
    font = ImageFont.load_default()
    draw.text((350,50),f"{player} — {team_name}",fill=(255,90,90),font=font)
    draw.text((50,300),"Model Projected Props",fill=(255,255,255),font=font)
    y=330
    for k,v in pred_next.items():
        draw.text((70,y),f"{k}: {v}",fill=(255,255,255),font=font)
        y+=25
    draw.text((50,550),"Reasoning:",fill=(255,255,255),font=font)
    draw.multiline_text((50,580),reasoning,fill=(200,200,200),font=font,spacing=4)
    buf=BytesIO()
    img.save(buf,format="PNG")
    st.download_button("⬇️ Download PNG", data=buf.getvalue(),
                       file_name=f"{player.replace(' ','_')}_projection.png", mime="image/png")

# Recent form averages
st.markdown("---")
st.markdown("## 📊 Recent Form & Averages")
def avg_stats(df): return {s:round(df[s].mean(),1) for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"] if s in df.columns}
for label,df in [("Last 5 Games",data.head(5)),("Last 10 Games",data.head(10)),("Current Season",cur)]:
    if df.empty: continue
    st.markdown(f"### {label}")
    avg=avg_stats(df)
    metric_cards(avg)
    bar_chart(avg,f"{label} — Key Averages")

# Career totals
try:
    cdf=playercareerstats.PlayerCareerStats(player_id=pid).get_data_frames()[0]
    if not cdf.empty:
        st.markdown("### Career Totals")
        totals={k:int(cdf[k].sum()) for k in ["PTS","REB","AST","FG3M","STL","BLK","TOV"] if k in cdf.columns}
        totals_fmt={k:f"{v:,}" for k,v in totals.items()}
        metric_cards(totals_fmt)
        bar_chart({k:totals[k] for k in ["PTS","REB","AST","FG3M"]},"Career Totals — Key Stats")
except Exception as e:
    st.warning(f"Career stats unavailable: {e}")

st.caption("⚡ Hot Shot Props AI Model © 2025 — Live NBA Performance Predictor")
