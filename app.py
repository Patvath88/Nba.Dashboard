# -------------------------------------------------
# HOT SHOT PROPS — NBA AI PROP DASHBOARD
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
from sklearn.metrics import mean_absolute_error
from PIL import Image
import requests
from io import BytesIO

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Hot Shot Props | NBA AI Dashboard", page_icon="🔥", layout="wide")

CURRENT_SEASON = "2025-26"
LAST_SEASON = "2024-25"
ASSETS_DIR = "assets"
PLAYER_PHOTO_DIR = os.path.join(ASSETS_DIR, "player_photos")
TEAM_LOGO_DIR = os.path.join(ASSETS_DIR, "team_logos")
os.makedirs(PLAYER_PHOTO_DIR, exist_ok=True)
os.makedirs(TEAM_LOGO_DIR, exist_ok=True)
ODDS_API_KEY = "e11d4159145383afd3a188f99489969e"

# -------------------------------------------------
# STYLE
# -------------------------------------------------
st.markdown("""
<style>
body {background-color:#0b0b0b;color:#F5F5F5;font-family:'Roboto',sans-serif;}
h1,h2,h3,h4 {font-family:'Oswald',sans-serif;color:#E50914;}
.scroll-row {display:flex;overflow-x:auto;gap:10px;padding-bottom:6px;scrollbar-width:thin;}
.scroll-row::-webkit-scrollbar {height:6px;}
.scroll-row::-webkit-scrollbar-thumb {background:#333;border-radius:4px;}
.metric-card {
    flex:0 0 auto;width:110px;background:linear-gradient(145deg,#1b1b1b,#121212);
    border:1px solid #2a2a2a;border-radius:10px;padding:8px;text-align:center;
    transition:all 0.3s ease;box-shadow:0 0 6px rgba(229,9,20,0.3);
}
.metric-card:hover {transform:scale(1.03);box-shadow:0 0 12px rgba(229,9,20,0.6);}
.metric-value {font-size:1.1em;font-weight:700;}
.metric-label {font-size:0.75em;color:#bbb;}
.value-bet-card {
    background:linear-gradient(160deg,#1c1c1c,#151515);
    border:1px solid #333;border-radius:18px;padding:14px;text-align:center;
    transition:transform 0.25s ease,box-shadow 0.25s ease;
}
.value-bet-card:hover {transform:scale(1.05);box-shadow:0 0 20px rgba(0,255,128,0.6);}
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
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=900)
def get_career(player_id):
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        return career.get_data_frames()[0]
    except Exception: return pd.DataFrame()

def enrich_stats(df):
    if df.empty: return df
    df["P+R"] = df["PTS"] + df["REB"]
    df["P+A"] = df["PTS"] + df["AST"]
    df["R+A"] = df["REB"] + df["AST"]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

# -------------------------------------------------
# ODDSAPI + ML ENGINE
# -------------------------------------------------
@st.cache_data(ttl=600)
def get_odds_data():
    """Fetch NBA player prop odds from OddsAPI (10-min cache)."""
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "regions": "us",
        "markets": "player_points,player_rebounds,player_assists,player_threes_made,player_points_rebounds_assists",
        "oddsFormat": "american",
        "include": "all",
        "apiKey": ODDS_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return data if isinstance(data, list) else []
        else:
            st.error(f"OddsAPI error {r.status_code}")
    except Exception as e:
        st.error(f"Odds fetch failed: {e}")
    return []

def extract_best_line(player_name, odds_json):
    best = None
    for game in odds_json:
        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                for outcome in market.get("outcomes", []):
                    name = outcome.get("description", "")
                    if player_name.lower() in name.lower():
                        line = outcome.get("point")
                        price = outcome.get("price")
                        if line is not None:
                            best = {"market": market.get("key"),
                                    "book": book["title"],
                                    "line": line, "price": price}
    return best

def prepare_features(df):
    if df.empty: return df
    df = df.copy()
    for w in [5,10,20]:
        for s in ["PTS","REB","AST","PRA"]:
            df[f"{s}_{w}"] = df[s].rolling(w).mean()
    return df.dropna().reset_index(drop=True)

def predict_prop_value(df, target_stat):
    if df.empty or target_stat not in df.columns: return None
    df = prepare_features(df)
    if len(df) < 8: return round(df[target_stat].mean(),1)
    feature_cols = [c for c in df.columns if any(x in c for x in ["PTS","REB","AST","PRA"]) and c != target_stat]
    X, y = df[feature_cols], df[target_stat]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    return round(model.predict([X.iloc[-1]])[0],1)

def compute_edge(model_pred, book_line):
    if not book_line: return 0, "N/A"
    diff = model_pred - book_line
    edge = (diff / book_line) * 100 if book_line != 0 else 0
    return round(edge,1), "Over" if diff > 0 else "Under"

# -------------------------------------------------
# VALUE BETS + WINS
# -------------------------------------------------
if "recent_wins" not in st.session_state:
    st.session_state["recent_wins"] = []

def get_recent_wins():
    df = pd.DataFrame(st.session_state["recent_wins"])
    return df[df["correct"]==True] if not df.empty else pd.DataFrame()

def render_value_bet_card(player_name, team_abbr, player_id, stat, model_pred, book_line, book, edge, direction):
    photo = get_player_photo(player_name, player_id)
    glow = "rgba(0,255,128,0.7)" if edge >= 10 else "rgba(255,165,0,0.7)"
    st.markdown(f"""
    <div class='value-bet-card' style='box-shadow:0 0 20px {glow};'>
        <img src='{photo}' width='120' style='border-radius:12px;'>
        <h4 style='margin-bottom:4px;color:#00FF80;'>{player_name}</h4>
        <p style='margin:2px 0;color:#aaa;'>Prop: <b>{stat}</b></p>
        <p style='margin:2px 0;color:#fff;'>Model: <b>{model_pred}</b> | Line: <b>{book_line}</b></p>
        <p style='margin:2px 0;color:#0f0;'>Edge: <b>{edge}%</b> ({direction})</p>
        <p style='margin:2px 0;color:#999;font-size:0.8em;'>Best Book: {book}</p>
    </div>
    """, unsafe_allow_html=True)

def render_hot_value_bets(player_id, player_name, team_abbr, odds_json, games_df):
    best_line = extract_best_line(player_name, odds_json)
    if not best_line:
        st.caption("No sportsbook data available for this player.")
        return
    stat_map = {"player_points":"PTS","player_rebounds":"REB","player_assists":"AST",
                "player_threes_made":"FG3M","player_points_rebounds_assists":"PRA"}
    target_stat = stat_map.get(best_line["market"],"PTS")
    model_pred = predict_prop_value(games_df, target_stat)
    if model_pred is None:
        st.warning("Not enough data for ML projection.")
        return
    edge, direction = compute_edge(model_pred, best_line["line"])
    st.session_state["latest_model"] = {f"{target_stat} Pred":model_pred,"Line":best_line["line"],"Edge%":edge}
    st.markdown("### 🤖 Model Projection")
    render_metric_cards(st.session_state["latest_model"], key_suffix="proj")
    if edge >= 10:
        st.markdown("## 🔥 Hot Value Bet")
        render_value_bet_card(player_name, team_abbr, player_id, target_stat, model_pred,
                              best_line["line"], best_line["book"], edge, direction)

def render_recent_wins_section():
    wins = get_recent_wins()
    st.markdown("### 🏆 Recent Model Wins")
    if wins.empty:
        st.caption("No wins recorded yet.")
        return
    cols = st.columns(2)
    for i, (_, row) in enumerate(wins.head(6).iterrows()):
        c = cols[i%2]
        color = "#00FF80" if row["direction"]=="Over" else "#FFA500"
        c.markdown(f"<div class='value-bet-card' style='border:1px solid {color};box-shadow:0 0 15px {color};'><b>{row['player']}</b><br>{row['stat']} | Pred {row['pred']} | Actual {row['actual']}<br><span style='color:{color};'>{row['direction']} ✓</span></div>", unsafe_allow_html=True)

# -------------------------------------------------
# METRIC ROWS + VISUALS
# -------------------------------------------------
def render_metric_cards(avg_dict, key_suffix=""):
    """Render compact metric cards in a responsive 4-column grid."""
    st.markdown("""
    <style>
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 10px;
        margin-bottom: 10px;
        justify-items: center;
    }
    .metric-card {
        width: 100%;
        max-width: 150px;
        background: linear-gradient(145deg, #1b1b1b, #121212);
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 0 6px rgba(229,9,20,0.3);
    }
    .metric-card:hover {
        transform: scale(1.03);
        box-shadow: 0 0 12px rgba(229,9,20,0.6);
    }
    .metric-value {
        font-size: 1.1em;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.8em;
        color: #bbb;
    }
    @media (max-width: 600px) {
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    </style>
    """, unsafe_allow_html=True)

    html = "<div class='metric-grid'>"
    for stat, val in avg_dict.items():
        # color coding for hot/cold
        color = "#00FF80" if isinstance(val, (int, float)) and val > 0 else "#FF5555"
        html += f"""
        <div class='metric-card' style='border:1px solid {color};'>
            <div class='metric-value' style='color:{color};'>{val}</div>
            <div class='metric-label'>{stat}</div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_recent_game_section(df, model_dict=None):
    if df.empty:
        st.info("No recent game data.")
        return
    last = df.iloc[0]
    metrics = {"PTS":last["PTS"],"REB":last["REB"],"AST":last["AST"],"3PM":last["FG3M"]}
    st.markdown("### 🕹️ Most Recent Game")
    render_metric_cards({k:round(v,1) for k,v in metrics.items()}, key_suffix="recent")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(metrics.keys()), y=list(metrics.values()),
                         marker_color=["#E50914","#00E676","#29B6F6","#FFD700"]))
    fig.update_layout(title="Performance Breakdown", paper_bgcolor="#0d0d0d",
                      plot_bgcolor="#0d0d0d", font_color="#F5F5F5",
                      margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, width="stretch", key="recent_chart")
    if model_dict:
        st.markdown("### 🤖 Model Projection")
        render_metric_cards(model_dict, key_suffix="model")

def render_expander(title, df):
    if df.empty:
        st.warning(f"No data for {title}")
        return
    avg = {s:round(df[s].mean(),1) for s in ["PTS","REB","AST","FG3M","STL","BLK","TOV","MIN"] if s in df.columns}
    avg.update({"P+R":round((df["PTS"]+df["REB"]).mean(),1),
                "P+A":round((df["PTS"]+df["AST"]).mean(),1),
                "PRA":round((df["PTS"]+df["REB"]+df["AST"]).mean(),1)})
    render_metric_cards(avg, key_suffix=title)
    metric_choice = st.selectbox(f"Select metric ({title})", ["PTS","REB","AST","FG3M","PRA"], key=f"sel_{title}")
    if "GAME_DATE" in df.columns:
        x, y = df["GAME_DATE"].iloc[::-1], df[metric_choice].iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=y, name=metric_choice, marker_color="#E50914", opacity=0.6))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers",
                                 line=dict(color="#29B6F6", width=2)))
        fig.update_layout(title=f"{metric_choice} Trend — {title}", paper_bgcolor="#0d0d0d",
                          plot_bgcolor="#0d0d0d", font_color="#F5F5F5", margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, width="stretch", key=f"chart_{title}_{metric_choice}")

# -------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------
st.title("🏀 Hot Shot Props — NBA AI Dashboard")
st.caption("Live player projections, sportsbook odds, and AI-driven value edges.")

nba_players = players.get_active_players()
nba_teams = teams.get_teams()
team_lookup = {t["id"]:t for t in nba_teams}
team_map = {}
for p in nba_players:
    tid = p.get("team_id")
    abbr = team_lookup[tid]["abbreviation"] if tid in team_lookup else "FA"
    team_map.setdefault(abbr,[]).append(p["full_name"])
team_options = []
for t,plist in sorted(team_map.items()):
    team_options.append(f"=== {t} ===")
    team_options.extend(sorted(plist))

selected_player = st.selectbox("Search or Browse by Team ↓", options=team_options, index=None, placeholder="Select player")
if not selected_player or selected_player.startswith("==="): st.stop()
pinfo = next((p for p in nba_players if p["full_name"]==selected_player), None)
player_id = pinfo["id"]
team_abbr = team_lookup[pinfo["team_id"]]["abbreviation"] if pinfo.get("team_id") in team_lookup else "FA"
photo = get_player_photo(selected_player, player_id)
logo = get_team_logo(team_abbr)

games_current = enrich_stats(get_games(player_id, CURRENT_SEASON))
if games_current.empty: games_current = enrich_stats(get_games(player_id, LAST_SEASON))
games_last = enrich_stats(get_games(player_id, LAST_SEASON))
career_df = get_career(player_id)
odds_json = get_odds_data()

col1, col2 = st.columns([1,3])
with col1:
    if photo: st.image(photo, width="stretch")
with col2:
    if logo: st.image(logo, width=100)
    st.markdown(f"## {selected_player} ({team_abbr})")

st.markdown("---")
render_hot_value_bets(player_id, selected_player, team_abbr, odds_json, games_current)
st.markdown("---")
render_recent_wins_section()
st.markdown("---")
render_recent_game_section(games_current, st.session_state.get("latest_model"))

with st.expander("📅 Last 5 Games", expanded=False): render_expander("last5", games_current.head(5))
with st.expander("📅 Last 10 Games", expanded=False): render_expander("last10", games_current.head(10))
with st.expander("
