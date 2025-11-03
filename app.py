# -------------------------------------------------
# HOT SHOT PROPS – NBA AI PROP DASHBOARD (PART 1)
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
import json

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
# STYLES
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color:#0b0b0b;
    color:#F5F5F5;
    font-family:'Roboto',sans-serif;
}
h1,h2,h3,h4 {
    font-family:'Oswald',sans-serif;
    color:#E50914;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 10px;
    margin-bottom: 10px;
}
.metric-card {
    background: linear-gradient(145deg, #1b1b1b, #121212);
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 8px;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 0 6px rgba(229,9,20,0.3);
}
.metric-card:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 0 15px rgba(229,9,20,0.6);
}
.metric-value {
    font-size: 1.3em;
    color: #fff;
    font-weight: bold;
}
.metric-label {
    font-size: 0.8em;
    color: #bbb;
}
.value-bet-card {
    background: linear-gradient(160deg, #1c1c1c, #151515);
    border: 1px solid #333;
    border-radius: 18px;
    padding: 14px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    text-align:center;
}
.value-bet-card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(0,255,128,0.6);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
@st.cache_data(ttl=3600)
def get_player_photo(player_name, player_id=None):
    safe = player_name.replace(" ", "_").lower()
    local_path = os.path.join(PLAYER_PHOTO_DIR, f"{safe}.png")
    if os.path.exists(local_path):
        return local_path
    try:
        if player_id:
            url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content))
                img.save(local_path)
                return local_path
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def get_team_logo(team_abbr):
    safe = team_abbr.lower()
    path = os.path.join(TEAM_LOGO_DIR, f"{safe}.png")
    if os.path.exists(path):
        return path
    try:
        url = f"https://loodibee.com/wp-content/uploads/nba-{safe}-logo.png"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content))
            img.save(path)
            return path
    except Exception:
        pass
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
    if df.empty:
        return df
    df["P+R"] = df["PTS"] + df["REB"]
    df["P+A"] = df["PTS"] + df["AST"]
    df["R+A"] = df["REB"] + df["AST"]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df
# -------------------------------------------------
# PART 2: ODDS FETCH + ML PREDICTION ENGINE
# -------------------------------------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

@st.cache_data(ttl=600)
def get_odds_data():
    """Fetch all NBA player prop odds from OddsAPI (cached 10 min)."""
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "regions": "us",
        "markets": "player_points,player_rebounds,player_assists,player_threes,player_points_rebounds_assists",
        "oddsFormat": "american",
        "apiKey": ODDS_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"OddsAPI error: {r.status_code}")
    except Exception as e:
        st.error(f"Odds fetch failed: {e}")
    return []

def extract_best_line(player_name, odds_json):
    """Return best line & sportsbook for this player (all markets)."""
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
                            if best is None or abs(price) > abs(best["price"]):
                                best = {
                                    "market": market.get("key"),
                                    "book": book["title"],
                                    "line": line,
                                    "price": price,
                                }
    return best

def prepare_features(df):
    """Create rolling-average feature set for ML model."""
    if df.empty: return df
    df = df.copy()
    df["PTS_5"] = df["PTS"].rolling(5).mean()
    df["REB_5"] = df["REB"].rolling(5).mean()
    df["AST_5"] = df["AST"].rolling(5).mean()
    df["PRA_5"] = df["PRA"].rolling(5).mean()
    df["PTS_10"] = df["PTS"].rolling(10).mean()
    df["REB_10"] = df["REB"].rolling(10).mean()
    df["AST_10"] = df["AST"].rolling(10).mean()
    df["PRA_10"] = df["PRA"].rolling(10).mean()
    df["PTS_20"] = df["PTS"].rolling(20).mean()
    df["REB_20"] = df["REB"].rolling(20).mean()
    df["AST_20"] = df["AST"].rolling(20).mean()
    df["PRA_20"] = df["PRA"].rolling(20).mean()
    df = df.dropna().reset_index(drop=True)
    return df

def predict_prop_value(df, target_stat):
    """
    Train Random Forest on player’s past 20 games
    and return projected value for the next game.
    """
    if df.empty or target_stat not in df.columns:
        return None

    df = prepare_features(df)
    if len(df) < 8:
        return df[target_stat].mean()  # not enough samples

    feature_cols = [c for c in df.columns if any(x in c for x in ["PTS","REB","AST","PRA","MIN"]) and c != target_stat]
    X = df[feature_cols]
    y = df[target_stat]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    proj = model.predict([X.iloc[-1]])[0]
    return round(proj, 1)

def compute_edge(model_pred, book_line):
    """Return edge % and value direction."""
    if not book_line:
        return 0, "N/A"
    diff = model_pred - book_line
    edge = (diff / book_line) * 100 if book_line != 0 else 0
    direction = "Over" if diff > 0 else "Under"
    return round(edge, 1), direction
# -------------------------------------------------
# PART 3: VALUE BETS + RECENT WINS
# -------------------------------------------------
from random import shuffle

# In-memory store for model results
if "recent_wins" not in st.session_state:
    st.session_state["recent_wins"] = []

def record_model_result(player_name, stat, model_pred, actual_val, direction, correct):
    """Store each model result in session memory."""
    st.session_state["recent_wins"].append({
        "player": player_name,
        "stat": stat,
        "pred": round(model_pred, 1),
        "actual": round(actual_val, 1),
        "direction": direction,
        "correct": correct,
        "time": datetime.now().strftime("%Y-%m-%d")
    })
    # Keep only last 20
    st.session_state["recent_wins"] = st.session_state["recent_wins"][-20:]

def get_recent_wins():
    """Return most recent correct model calls."""
    df = pd.DataFrame(st.session_state["recent_wins"])
    if df.empty:
        return pd.DataFrame()
    return df[df["correct"] == True].sort_values("time", ascending=False)

def render_value_bet_card(player_name, team_abbr, player_id, stat, model_pred, book_line, book, edge, direction):
    """Render glowing 3D-style card for high-value props."""
    photo = get_player_photo(player_name, player_id)
    logo = get_team_logo(team_abbr)
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
    """Compute & display only the top-edge prop for this player."""
    best_line = extract_best_line(player_name, odds_json)
    if not best_line:
        st.info("No active sportsbook lines found for this player.")
        return

    # Determine stat name
    stat_map = {
        "player_points": "PTS",
        "player_rebounds": "REB",
        "player_assists": "AST",
        "player_threes": "FG3M",
        "player_points_rebounds_assists": "PRA"
    }
    target_stat = stat_map.get(best_line["market"], "PTS")

    model_pred = predict_prop_value(games_df, target_stat)
    if model_pred is None:
        st.warning("Not enough data to train model.")
        return

    edge, direction = compute_edge(model_pred, best_line["line"])

    if edge >= 10:
        st.markdown("## 🔥 Hot Value Bet")
        render_value_bet_card(
            player_name, team_abbr, player_id,
            target_stat, model_pred, best_line["line"],
            best_line["book"], edge, direction
        )
    else:
        st.caption("No high-value props detected at this time.")

def render_recent_wins_section():
    """Show list of recent successful model predictions."""
    wins = get_recent_wins()
    st.markdown("### 🏆 Model’s Recent Wins")
    if wins.empty:
        st.info("No wins recorded yet.")
        return
    cols = st.columns(2)
    for i, (_, row) in enumerate(wins.head(10).iterrows()):
        c = cols[i % 2]
        with c:
            color = "#00FF80" if row["direction"] == "Over" else "#FFA500"
            c.markdown(f"""
            <div class='value-bet-card' style='border:1px solid {color};box-shadow:0 0 15px {color};'>
                <h5 style='margin:0;color:{color};'>{row["player"]} — {row["stat"]}</h5>
                <p style='margin:2px 0;color:#ccc;font-size:0.9em;'>
                Pred: <b>{row["pred"]}</b> | Actual: <b>{row["actual"]}</b> ({row["direction"]})
                </p>
                <p style='margin:2px 0;color:#777;font-size:0.75em;'>{row["time"]}</p>
            </div>
            """, unsafe_allow_html=True)
# -------------------------------------------------
# PART 4: PLAYER DASHBOARD + VISUALS
# -------------------------------------------------
def render_metric_cards(avg_dict):
    """Compact 2–3-row metric layout."""
    st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
    for stat, val in avg_dict.items():
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{stat}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_recent_game_section(df):
    """Show most recent game stats with a mini bar chart."""
    if df.empty: 
        st.info("No recent game data.")
        return
    last = df.iloc[0]
    metrics = {
        "PTS": last["PTS"], "REB": last["REB"],
        "AST": last["AST"], "FG3M": last["FG3M"]
    }
    st.markdown("### 🕹️ Most Recent Game")
    render_metric_cards({k: round(v,1) for k,v in metrics.items()})
    # Bar chart under cards
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(metrics.keys()), y=list(metrics.values()),
        marker_color=["#E50914","#00E676","#29B6F6","#FFD700"]
    ))
    fig.update_layout(
        title="Performance Breakdown",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        font_color="#F5F5F5", margin=dict(l=10,r=10,t=30,b=10),
        yaxis_title="Stat Value"
    )
    st.plotly_chart(fig, width="stretch", key="recent_game_chart")

def render_expander(title, df):
    """Reusable expander with metric cards and dropdown chart."""
    if df.empty:
        st.warning(f"No data for {title}")
        return

    avg = {
        "PTS": round(df["PTS"].mean(),1), "REB": round(df["REB"].mean(),1),
        "AST": round(df["AST"].mean(),1), "FG3M": round(df["FG3M"].mean(),1),
        "STL": round(df["STL"].mean(),1), "BLK": round(df["BLK"].mean(),1),
        "TOV": round(df["TOV"].mean(),1), "MIN": round(df["MIN"].mean(),1),
        "P+R": round((df["PTS"]+df["REB"]).mean(),1),
        "P+A": round((df["PTS"]+df["AST"]).mean(),1),
        "R+A": round((df["REB"]+df["AST"]).mean(),1),
        "PRA": round((df["PTS"]+df["REB"]+df["AST"]).mean(),1)
    }

    render_metric_cards(avg)

    metric_choice = st.selectbox(
        f"📊 Choose a metric to visualize ({title})",
        ["PTS","REB","AST","FG3M","STL","BLK","TOV","MIN","PRA"],
        key=f"metric_{title}"
    )

    if "GAME_DATE" in df.columns:
        x = df["GAME_DATE"].iloc[::-1]
        y = df[metric_choice].iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=y, name=metric_choice,
                             marker_color="#E50914", opacity=0.6))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers",
                                 line=dict(color="#29B6F6", width=2)))
        fig.update_layout(
            title=f"{metric_choice} Trend — {title}",
            paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
            font_color="#F5F5F5", margin=dict(l=10,r=10,t=40,b=10),
            legend=dict(orientation="h", yanchor="bottom")
        )
        st.plotly_chart(fig, width="stretch", key=f"chart_{title}_{metric_choice}")
# -------------------------------------------------
# PART 5: MAIN LAYOUT & ASSEMBLY
# -------------------------------------------------
st.title("🏀 Hot Shot Props — NBA AI Dashboard")
st.caption("Real-time player projections, sportsbook odds, and AI-driven value edges.")

# Load NBA players and teams
nba_players = players.get_active_players()
nba_teams = teams.get_teams()
team_lookup = {t["id"]: t for t in nba_teams}

# Build team/player dropdown
team_map = {}
for p in nba_players:
    tid = p.get("team_id")
    abbr = team_lookup[tid]["abbreviation"] if tid in team_lookup else "FA"
    team_map.setdefault(abbr, []).append(p["full_name"])

team_options = []
for t, plist in sorted(team_map.items()):
    team_options.append(f"=== {t} ===")
    team_options.extend(sorted(plist))

selected_player = st.selectbox(
    "Search or Browse by Team ↓",
    options=team_options,
    index=None,
    placeholder="Select an NBA player"
)
if selected_player is None or selected_player == "" or selected_player.startswith("==="):
    st.stop()

# Player info
pinfo = next((p for p in nba_players if p["full_name"] == selected_player), None)
if not pinfo:
    st.error("Player not found.")
    st.stop()

player_id = pinfo["id"]
team_abbr = "FA"
if pinfo.get("team_id") in team_lookup:
    team_abbr = team_lookup[pinfo["team_id"]]["abbreviation"]

photo = get_player_photo(selected_player, player_id)
logo = get_team_logo(team_abbr)

# Load data
games_current = enrich_stats(get_games(player_id, CURRENT_SEASON))
if games_current.empty:
    games_current = enrich_stats(get_games(player_id, LAST_SEASON))
games_last = enrich_stats(get_games(player_id, LAST_SEASON))
career_df = get_career(player_id)
odds_json = get_odds_data()

# Top profile
col1, col2 = st.columns([1,3])
with col1:
    if photo: st.image(photo, width="stretch")
with col2:
    if logo: st.image(logo, width=100)
    st.markdown(f"## {selected_player} ({team_abbr})")

# ----- HOT VALUE BET -----
st.markdown("---")
render_hot_value_bets(player_id, selected_player, team_abbr, odds_json, games_current)

# ----- RECENT WINS -----
st.markdown("---")
render_recent_wins_section()

# ----- PLAYER DASHBOARD -----
st.markdown("---")
render_recent_game_section(games_current)

with st.expander("📅 Last 5 Games", expanded=False):
    render_expander("last5", games_current.head(5))
with st.expander("📅 Last 10 Games", expanded=False):
    render_expander("last10", games_current.head(10))
with st.expander("📅 Last 20 Games", expanded=False):
    df20 = games_current.copy()
    if len(df20) < 20:
        need = 20 - len(df20)
        df20 = pd.concat([df20, games_last.head(need)], ignore_index=True)
    render_expander("last20", df20)
if not games_current.empty:
    with st.expander("📊 Current Season Averages", expanded=False):
        render_expander("currentSeason", games_current)
if not games_last.empty:
    with st.expander("🕰️ Last Season Averages", expanded=False):
        render_expander("lastSeason", games_last)
if not career_df.empty:
    career_avg = career_df.groupby("PLAYER_ID").agg({
        "PTS":"mean","REB":"mean","AST":"mean","FG3M":"mean",
        "STL":"mean","BLK":"mean","TOV":"mean","MIN":"mean"
    }).reset_index()
    career_avg = enrich_stats(career_avg)
    with st.expander("🏆 Career Averages", expanded=False):
        render_expander("career", career_avg)

# Footer
st.markdown(f"""
---
<div style='text-align:center;color:#777;font-size:13px;margin-top:20px;'>
Hot Shot Props © {datetime.now().year} | Powered by NBA API + OddsAPI + Streamlit
</div>
""", unsafe_allow_html=True)
