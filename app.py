import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

st.set_page_config(page_title="NBA Player Predictor (Codespaces Safe)", layout="wide")
st.title("🏀 NBA Player Next Game Predictor (BallDontLie API)")

# ----------------------------
# 1. Safe API Helpers
# ----------------------------
BASE_URL = "https://www.balldontlie.io/api/v1"

def safe_get(url, params=None):
    """Always returns JSON or an empty dict instead of crashing."""
    try:
        res = requests.get(url, params=params, timeout=20)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"⚠️ API request failed: {e}")
        return {"data": []}

@st.cache_data(ttl=3600)
def get_player_id(name):
    data = safe_get(f"{BASE_URL}/players", {"search": name})
    if data["data"]:
        p = data["data"][0]
        return p["id"], f"{p['first_name']} {p['last_name']}"
    return None, None

@st.cache_data(ttl=600)
def get_game_logs(pid):
    games = []
    page = 1
    while True:
        data = safe_get(f"{BASE_URL}/stats", {"player_ids[]": pid, "per_page": 100, "page": page})
        if not data["data"]:
            break
        games.extend(data["data"])
        if not data["meta"]["next_page"]:
            break
        page += 1
    if not games:
        return pd.DataFrame()
    df = pd.json_normalize(games)
    df["game.date"] = pd.to_datetime(df["game.date"])
    df = df.sort_values("game.date")
    df["pts"] = df["pts"].astype(float)
    df["reb"] = df["reb"].astype(float)
    df["ast"] = df["ast"].astype(float)
    df["fg3m"] = df["fg3m"].astype(float)
    df["min_played"] = df["min"].apply(lambda x: int(x.split(":")[0]) if isinstance(x, str) else 0)
    return df

# ----------------------------
# 2. Regression
# ----------------------------
def lin_reg(X, y, next_f):
    X = np.c_[np.ones(len(X)), X]
    coeffs = np.linalg.pinv(X.T @ X) @ X.T @ y
    return float(np.dot([1] + next_f, coeffs))

def prep_features(df):
    for stat in ["pts","reb","ast","fg3m"]:
        df[f"{stat}_avg5"] = df[stat].rolling(5,1).mean()
    feats = ["min_played","pts_avg5","reb_avg5","ast_avg5","fg3m_avg5"]
    return df.dropna(subset=feats), feats

# ----------------------------
# 3. Streamlit UI
# ----------------------------
name = st.text_input("Enter player name (e.g. LeBron James):", "LeBron James")

if name:
    pid, pname = get_player_id(name)
    if pid:
        df = get_game_logs(pid)
        if not df.empty:
            df, feats = prep_features(df)
            st.subheader(f"📊 {pname} Recent Games")
            st.dataframe(df[["game.date","pts","reb","ast","fg3m","min_played"]].tail(10))

            st.sidebar.header("Next Game Inputs")
            mins = st.sidebar.slider("Projected Minutes",25,45,35)
            opp = st.sidebar.text_input("Opponent Team (for display only):","Warriors")

            # Opponent-adjustment placeholder (you can hook real defensive data later)
            adj = {"pts":1.0,"reb":1.0,"ast":1.0,"fg3m":1.0}

            last = df.iloc[-1]
            base = [mins,last["pts_avg5"],last["reb_avg5"],last["ast_avg5"],last["fg3m_avg5"]]

            preds={}
            for stat in ["pts","reb","ast","fg3m"]:
                X,y=df[feats].values,df[stat].values
                preds[stat]=lin_reg(X,y,base)*adj[stat]

            st.subheader(f"🎯 Predicted Next Game Stats vs {opp}")
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Points",f"{preds['pts']:.1f}")
            c2.metric("Rebounds",f"{preds['reb']:.1f}")
            c3.metric("Assists",f"{preds['ast']:.1f}")
            c4.metric("3PT Made",f"{preds['fg3m']:.1f}")

            st.markdown("### 📈 Recent Trends (Last 10 Games)")
            c1.plotly_chart(px.line(df.tail(10),x="game.date",y="pts",title="Points"),use_container_width=True)
            c2.plotly_chart(px.line(df.tail(10),x="game.date",y="reb",title="Rebounds"),use_container_width=True)
        else:
            st.warning("No game data found for this player.")
    else:
        st.warning("Player not found.")
