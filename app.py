# -------------------------------------------------
# HOT SHOT PROPS — NBA AI PLAYER MODEL DASHBOARD (FINAL)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from PIL import Image
import requests
from io import BytesIO

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Hot Shot Props | NBA Player AI Dashboard",
                   page_icon="🏀", layout="wide")

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
.player-header {
    display:flex;align-items:center;gap:20px;margin-bottom:20px;
}
.player-img {
    width:120px;height:120px;border-radius:50%;border:3px solid #ff6f00;
    object-fit:cover;box-shadow:0 0 12px #ff6f00;
}
.team-logo {
    width:80px;height:80px;object-fit:contain;
}
.arrow-up {color:#00FF80;}
.arrow-down {color:#FF5555;}
@media(max-width:600px){.metric-grid{grid-template-columns:repeat(2,1fr);}}
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
    df = df.copy()
    for w in [5,10,20]:
        for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV"]:
            df.loc[:, f"{s}_avg_{w}"] = df[s].rolling(w).mean()
            df.loc[:, f"{s}_std_{w}"] = df[s].rolling(w).std()
    df = df.dropna().reset_index(drop=True)
    return df

def train_model(df):
    df = prepare(df)
    feats = [c for c in df.columns if "avg" in c or "std" in c]
    stats = ["PTS","REB","AST","FG3M","STL","BLK","TOV","P+R","P+A","R+A","PRA"]
    models = {}
    for s in stats:
        if s not in df or df[feats].isna().all().any(): continue
        X, y = df[feats], df[s]
        if len(X) < 8: continue
        Xtr, Xte, Ytr, Yte = train_test_split(X, y, test_size=.2, random_state=42)
        m = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        m.fit(Xtr, Ytr)
        models[s] = (m, feats)
    return models

def metric_cards(stats, season=None):
    html = "<div class='metric-grid'>"
    for k, v in stats.items():
        arrow = ""
        if season and k in season:
            if v > season[k]: arrow = f"<span class='arrow-up'>▲</span>"
            elif v < season[k]: arrow = f"<span class='arrow-down'>▼</span>"
        html += f"<div class='metric-card'><div class='metric-value'>{v} {arrow}</div><div class='metric-label'>{k}</div></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def bar_chart(df, title):
    stats = ["PTS","REB","AST","FG3M"]
    avg = [df[s].mean() for s in stats]
    fig = go.Figure([go.Bar(x=stats, y=avg,
                            marker_color=["#ff6f00","#00e676","#2979ff","#fdd835"])])
    fig.update_layout(title=title, paper_bgcolor="#000", plot_bgcolor="#000", font_color="#fff")
    st.plotly_chart(fig, width="stretch")

def bar_compare(actual, pred, title):
    stats = list(actual.keys())
    a_vals = [actual[s] for s in stats]
    p_vals = [pred.get(s, 0) for s in stats]
    fig = go.Figure()
    fig.add_bar(x=stats, y=a_vals, name="Last Game", marker_color="#ff6f00")
    fig.add_bar(x=stats, y=p_vals, name="Model Prediction (Next Game)", marker_color="#00bcd4", opacity=.7)
    fig.update_layout(title=title, barmode="group", paper_bgcolor="#000", plot_bgcolor="#000", font_color="#F5F5F5")
    st.plotly_chart(fig, width="stretch")

def get_player_photo(player_id):
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

def get_team_logo(team_name):
    try:
        team = next(t for t in teams.get_teams() if t["full_name"] == team_name)
        return f"https://cdn.nba.com/logos/nba/{team['id']}/primary/L/logo.svg"
    except StopIteration:
        return None

# -------------------------------------------------
# MAIN
# -------------------------------------------------
st.title("🏀 Hot Shot Props — Player AI Dashboard")
st.caption("AI model predictions, recent performance, and player insights")

nba_players = players.get_active_players()
player_list = sorted([p["full_name"] for p in nba_players])
player = st.selectbox("Select a Player", player_list)

if not player:
    st.stop()
pid = next(p["id"] for p in nba_players if p["full_name"] == player)

# header visuals
photo_url = get_player_photo(pid)
team_name = "Unknown Team"
team_logo = None
try:
    g_log = get_games(pid, CURRENT_SEASON)
    if not g_log.empty:
        matchup = g_log["MATCHUP"].iloc[0]
        team_name = matchup.split(" ")[0]
        team_logo = get_team_logo(team_name)
except Exception:
    pass

st.markdown("<div class='player-header'>", unsafe_allow_html=True)
st.image(photo_url, caption=player, width=120, use_container_width=False)
if team_logo:
    st.image(team_logo, width=80)
st.markdown("</div>", unsafe_allow_html=True)

# gather data
cur = enrich(get_games(pid, CURRENT_SEASON))
pre = enrich(get_games(pid, PRESEASON))
last = enrich(get_games(pid, LAST_SEASON))
data = pd.concat([cur, pre, last]).drop_duplicates(subset="Game_ID", keep="first")
if data.empty:
    st.error("No data found for this player.")
    st.stop()
data = data.sort_values("GAME_DATE").reset_index(drop=True)

# train model
models = train_model(data)

# --- AI Predictions for Next Game ---
pred_next = {}
if models:
    dfp = prepare(data)
    if not dfp.empty:
        feats = list(next(iter(models.values()))[1])
        if all(f in dfp.columns for f in feats):
            X = dfp.iloc[[-1]][feats]
            for s, (m, _) in models.items():
                try:
                    val = float(m.predict(X)[0])
                    pred_next[s] = round(val, 1)
                except Exception:
                    continue

if pred_next:
    st.subheader("🧠 AI Predicted Next Game Stats")
    season_avg = {s: round(data[s].mean(), 1)
                  for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV"] if s in data}
    metric_cards(pred_next, season_avg)

# --- Most Recent Game Stats ---
if not data.empty:
    last_game = data.iloc[0]
    recent = {s: round(last_game[s], 1) for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"] if s in data.columns}
    st.subheader("🔥 Most Recent Game Performance")
    metric_cards(recent)
    if pred_next:
        bar_compare(recent, pred_next, "Comparison: Last Game vs AI Next Game Prediction")

# --- Historical Performance Sections ---
def show_form_section(title, df):
    st.markdown(f"### {title}")
    if df.empty: return
    stats = ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]
    form = {s: round(df[s].mean(), 1) for s in stats if s in df}
    metric_cards(form)
    bar_chart(df, f"{title} — Averages")

if len(data) >= 5: show_form_section("Last 5 Games", data.head(5))
if len(data) >= 10: show_form_section("Last 10 Games", data.head(10))
if len(data) >= 20: show_form_section("Last 20 Games", data.head(20))
show_form_section("This Season Averages", cur)
show_form_section("Last Season Averages", last)
show_form_section("Career Totals", data)

st.markdown("---")
st.caption("⚡ Hot Shot Props AI Player Dashboard © 2025")
