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
PRESEASON = "2025 Preseason"
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
    df["R+A"] = df["REB"] + df["AST"]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

# -------------------------------------------------
# MODEL ENGINE
# -------------------------------------------------
@st.cache_data(ttl=900)
def prepare_features(df):
    if df.empty: return df
    df = df.copy()
    for w in [5,10,20]:
        for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV"]:
            df[f"{s}_avg_{w}"] = df[s].rolling(w).mean()
            df[f"{s}_std_{w}"] = df[s].rolling(w).std()
    df = df.dropna().reset_index(drop=True)
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["P+R"] = df["PTS"] + df["REB"]
    df["P+A"] = df["PTS"] + df["AST"]
    df["R+A"] = df["REB"] + df["AST"]
    return df

def train_player_model(df):
    if df.empty: return None, {}
    df = prepare_features(df)
    target_stats = ["PTS","REB","AST","FG3M","STL","BLK","TOV","P+R","P+A","R+A","PRA"]
    models, scores = {}, {}
    features = [c for c in df.columns if "avg_" in c or "std_" in c]
    for stat in target_stats:
        if stat not in df.columns: continue
        X, y = df[features], df[stat]
        if len(X) < 5: continue
        Xtr,Xte,Ytr,Yte = train_test_split(X,y,test_size=0.2,random_state=42)
        model = RandomForestRegressor(n_estimators=250,max_depth=8,random_state=42)
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
    return preds

# -------------------------------------------------
# METRIC RENDER
# -------------------------------------------------
def render_metric_cards(avg_dict):
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

def render_avg_bar(df, title):
    if df.empty: return
    avg = {s:round(df[s].mean(),1) for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV"] if s in df.columns}
    avg["PRA"] = round(df["PTS"].mean() + df["REB"].mean() + df["AST"].mean(),1)
    st.markdown(f"### {title}")
    render_metric_cards(avg)
    fig = go.Figure([go.Bar(x=list(avg.keys()), y=list(avg.values()), marker_color="#E50914")])
    fig.update_layout(title=f"{title} — Averages",paper_bgcolor="#0d0d0d",plot_bgcolor="#0d0d0d",font_color="#F5F5F5")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# MAIN DASHBOARD
# -------------------------------------------------
st.title("Hot Shot Props — NBA AI Dashboard")
st.caption("AI-driven player form and projection model with extended historical learning.")

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

selected_player = st.selectbox("Search or Browse by Team ↓", team_options,
                               index=None, placeholder="Select player")
if not selected_player or selected_player.startswith("==="):
    st.stop()

pinfo = next((p for p in nba_players if p["full_name"]==selected_player), None)
player_id = pinfo["id"]
team_abbr = team_lookup[pinfo["team_id"]]["abbreviation"] if pinfo.get("team_id") in team_lookup else "FA"
photo = get_player_photo(selected_player, player_id)
logo = get_team_logo(team_abbr)

# --- Combine seasons intelligently ---
games_current = enrich_stats(get_games(player_id, CURRENT_SEASON))
if len(games_current) < 8:
    games_pre = enrich_stats(get_games(player_id, PRESEASON))
    games_last = enrich_stats(get_games(player_id, LAST_SEASON))
    combined = pd.concat([games_current, games_pre, games_last]).drop_duplicates(subset="Game_ID", keep="first")
    games_current = combined

career_df = get_career(player_id)

col1,col2 = st.columns([1,3])
with col1:
    if photo: st.image(photo, use_container_width=True)
with col2:
    if logo: st.image(logo, width=100)
    st.markdown(f"## {selected_player} ({team_abbr})")

# --- MODEL TRAINING & PREDICTION ---
models, scores = train_player_model(games_current)
preds = predict_next_game(games_current, models)

st.markdown("---")
with st.expander("🤖 AI Model Projections — Next Game", expanded=True):
    if not preds:
        st.error("Not enough data to generate predictions yet.")
    else:
        render_metric_cards(preds)

st.markdown("---")

# ---- RECENT FORM SECTIONS (no expanders) ----
if not games_current.empty:
    recent = games_current.head(1)
    render_avg_bar(recent, "🔥 Most Recent Game")

    if len(games_current) >= 5:
        render_avg_bar(games_current.head(5), "📅 Last 5 Games")
    if len(games_current) >= 10:
        render_avg_bar(games_current.head(10), "📅 Last 10 Games")
    if len(games_current) >= 20:
        render_avg_bar(games_current.head(20), "📅 Last 20 Games")

    # Season Averages
    render_avg_bar(games_current, "📊 Current Season Averages")

# ---- Career Totals ----
if not career_df.empty:
    totals = career_df[["PTS","REB","AST","STL","BLK","TOV","FG3M","MIN"]].sum()
    st.markdown("🏀 **Career Totals**")
    render_metric_cards({
        "PTS":int(totals["PTS"]),"REB":int(totals["REB"]),"AST":int(totals["AST"]),
        "3PM":int(totals["FG3M"]),"STL":int(totals["STL"]),"BLK":int(totals["BLK"]),
        "TOV":int(totals["TOV"]),"MIN":int(totals["MIN"])
    })

st.markdown("---")
st.caption("⚡ Powered by NBA API and Hot Shot Props AI Engine © 2025")
