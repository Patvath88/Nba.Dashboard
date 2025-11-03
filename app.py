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

# -------------------------------------------------
# MAIN DASHBOARD
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
st.subheader("📊 Recent Game vs. Model Prediction")

if games_current.empty:
    st.info("No recent game data available.")
else:
    last = games_current.iloc[0]
    actual = {
        "PTS": round(last["PTS"], 1),
        "REB": round(last["REB"], 1),
        "AST": round(last["AST"], 1),
        "3PM": round(last["FG3M"], 1),
        "PRA": round(last["PTS"] + last["REB"] + last["AST"], 1)
    }

    st.markdown("### 🔥 Most Recent Game (Actual)")
    render_metric_cards(actual, key_suffix="actual")

    st.markdown("### 🧠 Model Predictions (Next Game)")
    render_metric_cards(preds, key_suffix="preds")

    stats = list(actual.keys())
    actual_vals = [actual[s] for s in stats]
    pred_vals = [preds.get(s, 0) for s in stats]

    # Comparison chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=stats, y=actual_vals, name="Actual", marker_color="#E50914", opacity=0.7))
    fig.add_trace(go.Bar(x=stats, y=pred_vals, name="Predicted", marker_color="#29B6F6", opacity=0.7))
    fig.update_layout(barmode="group", title="Actual vs Predicted — Key Stats",
                      paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                      font_color="#F5F5F5", legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True, key="compare_chart")

    # --- Accuracy over time ---
    if len(games_current) > 10 and "PTS" in games_current:
        trend_df = prepare_features(games_current.copy())
        trend_preds = []
        features = [c for c in trend_df.columns if "avg_" in c or "std_" in c]
        for i in range(5, len(trend_df)):
            x = trend_df.iloc[i:i+1][features]
            preds_loop = {k: models[k].predict(x)[0] for k in models.keys()}
            preds_loop["GAME_DATE"] = trend_df.iloc[i]["GAME_DATE"]
            preds_loop["PTS_ACTUAL"] = trend_df.iloc[i]["PTS"]
            trend_preds.append(preds_loop)
        trend = pd.DataFrame(trend_preds)
        if not trend.empty:
            trend["Error"] = (trend["PTS_ACTUAL"] - trend["PTS"]).abs()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=trend["GAME_DATE"], y=trend["Error"], mode="lines+markers",
                                      line=dict(color="#FFD700", width=3)))
            fig2.update_layout(title="Model Error Over Last 10 Games (PTS)",
                               paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                               font_color="#F5F5F5", yaxis_title="Absolute Error")
            st.plotly_chart(fig2, use_container_width=True, key="trend_chart")

st.markdown("---")
st.caption("⚡ Powered by NBA API and Hot Shot Props AI Engine © 2025")
