# -------------------------------------------------
# HOT SHOT PROPS — NBA AI FORM & MODEL DASHBOARD
# -------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, playercareerstats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from PIL import Image
import requests
from io import BytesIO

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Hot Shot Props | NBA AI Dashboard",
                   page_icon="🔥", layout="wide")

CURRENT_SEASON = "2025-26"
LAST_SEASON = "2024-25"
ASSETS_DIR = "assets"
PLAYER_PHOTO_DIR = os.path.join(ASSETS_DIR, "player_photos")
TEAM_LOGO_DIR = os.path.join(ASSETS_DIR, "team_logos")
os.makedirs(PLAYER_PHOTO_DIR, exist_ok=True)
os.makedirs(TEAM_LOGO_DIR, exist_ok=True)

# -------------------------------------------------
# STYLE
# -------------------------------------------------
st.markdown("""
<style>
body {background:#0b0b0b;color:#F5F5F5;font-family:'Roboto',sans-serif;}
h1,h2,h3,h4 {font-family:'Oswald',sans-serif;color:#E50914;}
.metric-grid {
    display:grid;
    grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
    gap:10px;margin-bottom:10px;justify-items:center;
}
.metric-card {
    width:100%;max-width:150px;background:linear-gradient(145deg,#1b1b1b,#121212);
    border:1px solid #2a2a2a;border-radius:10px;padding:10px;text-align:center;
    transition:all .3s ease;box-shadow:0 0 6px rgba(229,9,20,.3);
}
.metric-card:hover{transform:scale(1.03);box-shadow:0 0 12px rgba(229,9,20,.6);}
.metric-value{font-size:1.1em;font-weight:700;}
.metric-label{font-size:.8em;color:#bbb;}
@media(max-width:600px){.metric-grid{grid-template-columns:repeat(2,1fr);}}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
@st.cache_data(ttl=3600)
def get_player_photo(player_name, player_id=None):
    safe = player_name.replace(" ", "_").lower()
    path = os.path.join(PLAYER_PHOTO_DIR, f"{safe}.png")
    if os.path.exists(path): return path
    try:
        url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            Image.open(BytesIO(r.content)).save(path)
            return path
    except Exception: pass
    return None

@st.cache_data(ttl=3600)
def get_team_logo(team_abbr):
    safe = team_abbr.lower()
    path = os.path.join(TEAM_LOGO_DIR, f"{safe}.png")
    if os.path.exists(path): return path
    try:
        url = f"https://loodibee.com/wp-content/uploads/nba-{safe}-logo.png"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            Image.open(BytesIO(r.content)).save(path)
            return path
    except Exception: pass
    return None

@st.cache_data(ttl=900)
def get_games(player_id, season):
    try:
        log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        return log.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def get_career(player_id):
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        return career.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

def enrich_stats(df):
    if df.empty: return df
    df["P+R"] = df["PTS"] + df["REB"]
    df["P+A"] = df["PTS"] + df["AST"]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

# -------------------------------------------------
# MODEL ENGINE (multi-stat predictor)
# -------------------------------------------------
@st.cache_data(ttl=900)
def prepare_features(df):
    if df.empty: return df
    df = df.copy()
    for w in [5,10,20]:
        for s in ["PTS","REB","AST","FG3M"]:
            df[f"{s}_avg_{w}"] = df[s].rolling(w).mean()
            df[f"{s}_std_{w}"] = df[s].rolling(w).std()
    df = df.dropna().reset_index(drop=True)
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

def train_player_model(df):
    if df.empty: return None, {}
    df = prepare_features(df)
    target_stats = ["PTS","REB","AST","FG3M","PRA"]
    models, scores = {}, {}
    features = [c for c in df.columns if "avg_" in c or "std_" in c]
    for stat in target_stats:
        X, y = df[features], df[stat]
        if len(X) < 8: continue
        Xtr,Xte,Ytr,Yte = train_test_split(X,y,test_size=0.2,random_state=42)
        model = RandomForestRegressor(n_estimators=300,max_depth=8,random_state=42)
        model.fit(Xtr,Ytr)
        preds = model.predict(Xte)
        mae = round(mean_absolute_error(Yte,preds),2)
        r2 = round(r2_score(Yte,preds),2)
        models[stat]=model
        scores[stat]={"MAE":mae,"R2":r2}
    return models,scores

def predict_next_game(df, models):
    if not models or df.empty: return {}
    df = prepare_features(df)
    latest = df.iloc[-1:]
    features = [c for c in df.columns if "avg_" in c or "std_" in c]
    preds = {}
    for stat,model in models.items():
        preds[stat] = round(float(model.predict(latest[features])[0]),1)
    preds["PRA"]=round(preds.get("PTS",0)+preds.get("REB",0)+preds.get("AST",0),1)
    return preds

# -------------------------------------------------
# METRIC CARDS + VISUALS
# -------------------------------------------------
def render_metric_cards(avg_dict, key_suffix=""):
    html = "<div class='metric-grid'>"
    for stat,val in avg_dict.items():
        color = "#00FF80" if isinstance(val,(int,float)) and val>0 else "#FF5555"
        html += f"""
        <div class='metric-card' style='border:1px solid {color};'>
            <div class='metric-value' style='color:{color};'>{val}</div>
            <div class='metric-label'>{stat}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_recent_game_section(df):
    if df.empty:
        st.info("No recent game data."); return
    last = df.iloc[0]
    metrics = {"PTS":last["PTS"],"REB":last["REB"],"AST":last["AST"],"3PM":last["FG3M"]}
    st.markdown("### Most Recent Game")
    render_metric_cards({k:round(v,1) for k,v in metrics.items()}, key_suffix="recent")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(metrics.keys()), y=list(metrics.values()),
                         marker_color=["#E50914","#00E676","#29B6F6","#FFD700"]))
    fig.update_layout(title="Performance Breakdown",paper_bgcolor="#0d0d0d",
                      plot_bgcolor="#0d0d0d",font_color="#F5F5F5",
                      margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True, key="recent_chart")

def render_model_section(models, preds, scores):
    st.markdown("### AI Model Predictions")
    render_metric_cards(preds, key_suffix="preds")
    st.markdown("### Model Performance Metrics")
    render_metric_cards({f"{k} MAE":v["MAE"] for k,v in scores.items()}, key_suffix="mae")
    render_metric_cards({f"{k} R²":v["R2"] for k,v in scores.items()}, key_suffix="r2")

def render_expander(title, df):
    if df.empty:
        st.warning(f"No data for {title}"); return
    avg = {s:round(df[s].mean(),1) for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","MIN"] if s in df.columns}
    avg.update({"P+R":round((df["PTS"]+df["REB"]).mean(),1),
                "P+A":round((df["PTS"]+df["AST"]).mean(),1),
                "PRA":round((df["PTS"]+df["REB"]+df["AST"]).mean(),1)})
    render_metric_cards(avg, key_suffix=title)
    metric_choice = st.selectbox(f"Select metric ({title})",
                                 ["PTS","REB","AST","FG3M","PRA"], key=f"sel_{title}")
    if "GAME_DATE" in df.columns:
        x, y = df["GAME_DATE"].iloc[::-1], df[metric_choice].iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x,y=y,name=metric_choice,marker_color="#E50914",opacity=.6))
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines+markers",
                                 line=dict(color="#29B6F6",width=2)))
        fig.update_layout(title=f"{metric_choice} Trend — {title}",
                          paper_bgcolor="#0d0d0d",plot_bgcolor="#0d0d0d",
                          font_color="#F5F5F5",margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{title}_{metric_choice}")

# -------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------
st.title("Hot Shot Props — NBA AI Dashboard")
st.caption("AI-driven player form analysis and model performance tracker.")

nba_players = players.get_active_players()
nba_teams = teams.get_teams()
team_lookup = {t["id"]:t for t in nba_teams}
team_map = {}
for p in nba_players:
    tid = p.get("team_id")
    abbr = team_lookup[tid]["abbreviation"] if tid in team_lookup else "FA"
    team_map.setdefault(abbr,[]).append(p["full_name"])
team_options=[]
for t,plist in sorted(team_map.items()):
    team_options.append(f"=== {t} ===")
    team_options.extend(sorted(plist))

selected_player = st.selectbox("Search or Browse by Team ↓", options=team_options,
                               index=None, placeholder="Select player")
if not selected_player or selected_player.startswith("==="):
    st.stop()

pinfo = next((p for p in nba_players if p["full_name"]==selected_player), None)
player_id = pinfo["id"]
team_abbr = team_lookup[pinfo["team_id"]]["abbreviation"] if pinfo.get("team_id") in team_lookup else "FA"
photo = get_player_photo(selected_player, player_id)
logo = get_team_logo(team_abbr)

games_current = enrich_stats(get_games(player_id, CURRENT_SEASON))
if games_current.empty:
    games_current = enrich_stats(get_games(player_id, LAST_SEASON))
games_last = enrich_stats(get_games(player_id, LAST_SEASON))
career_df = get_career(player_id)

col1,col2 = st.columns([1,3])
with col1:
    if photo: st.image(photo, use_container_width=True)
with col2:
    if logo: st.image(logo, width=100)
    st.markdown(f"## {selected_player} ({team_abbr})")

# --- Train + Predict ---
models, scores = train_player_model(games_current)
preds = predict_next_game(games_current, models)

st.markdown("---")
render_recent_game_section(games_current)
st.markdown("---")
render_model_section(models, preds, scores)
st.markdown("---")

with st.expander("Last 5 Games", expanded=False):
    render_expander("last5", games_current.head(5))
with st.expander("Last 10 Games", expanded=False):
    render_expander("last10", games_current.head(10))
with st.expander("Last 20 Games", expanded=False):
    df20 = games_current.copy()
    if len(df20)<20 and not games_last.empty:
        df20 = pd.concat([df20, games_last.head(20-len(df20))])
    render_expander("last20", df20)
with st.expander("Season Averages", expanded=False):
    if not games_current.empty:
        season_avg = games_current.mean(numeric_only=True)
        render_metric_cards({
            "PTS":round(season_avg["PTS"],1),
            "REB":round(season_avg["REB"],1),
            "AST":round(season_avg["AST"],1),
            "3PM":round(season_avg["FG3M"],1),
            "STL":round(season_avg["STL"],1),
            "BLK":round(season_avg["BLK"],1),
            "TOV":round(season_avg["TOV"],1),
            "MIN":round(season_avg["MIN"],1),
            "PRA":round(season_avg["PTS"]+season_avg["REB"]+season_avg["AST"],1)
        }, key_suffix="season")
with st.expander("Career Averages", expanded=False):
    if not career_df.empty:
        career_avg = career_df.groupby("SEASON_ID")[["PTS","REB","AST","STL","BLK","TOV","FG3M","MIN"]].mean().mean()
        render_metric_cards({
            "PTS":round(career_avg["PTS"],1),
            "REB":round(career_avg["REB"],1),
            "AST":round(career_avg["AST"],1),
            "3PM":round(career_avg["FG3M"],1),
            "STL":round(career_avg["STL"],1),
            "BLK":round(career_avg["BLK"],1),
            "TOV":round(career_avg["TOV"],1),
            "MIN":round(career_avg["MIN"],1),
        }, key_suffix="career")

st.markdown("---")
st.caption("⚡ Powered by NBA API and Hot Shot Props AI Engine © 2025")
