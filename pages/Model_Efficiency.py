import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

st.set_page_config(page_title="🧠 Model Efficiency Dashboard", layout="wide")

st.markdown("""
<style>
body {background-color:black; color:white;}
.metric-card {
    border: 1px solid #00FFFF;
    border-radius: 15px;
    padding: 18px;
    background-color: #111;
    box-shadow: 0 0 15px #00FFFF55;
    text-align: center;
    margin-bottom: 20px;
}
.player-card {
    border: 1px solid #E50914;
    border-radius: 15px;
    padding: 12px;
    background-color: #1a1a1a;
    box-shadow: 0 0 12px #E5091444;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Model Efficiency Command Deck")

# ---------------------- LOAD ----------------------
path = "saved_projections.csv"
try:
    data = pd.read_csv(path)
except FileNotFoundError:
    st.warning("No saved projections found.")
    st.stop()
if data.empty:
    st.info("No data to analyze.")
    st.stop()

nba_players = players.get_active_players()
player_map = {p["full_name"]: p["id"] for p in nba_players}

@st.cache_data(ttl=120)
def get_gamelog(pid):
    try:
        gl = playergamelog.PlayerGameLog(player_id=pid, season="2025-26").get_data_frames()[0]
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        return gl
    except Exception:
        return pd.DataFrame()

def evaluate_prediction(row):
    pid = player_map.get(row["player"])
    if not pid:
        return None
    gl = get_gamelog(pid)
    if gl.empty:
        return None
    game_row = gl[gl["GAME_DATE"].astype(str).str.contains(str(row["game_date"]))]
    if game_row.empty:
        return None
    game = game_row.iloc[0]
    result = {"player": row["player"], "game_date": row["game_date"]}
    for stat in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "PRA"]:
        pred = row.get(stat, 0)
        act = game.get(stat, 0)
        if pred:
            result[stat + "_acc"] = max(0, 1 - abs(pred - act) / pred) * 100
            result[stat + "_err"] = act - pred
    return result

st.info("Analyzing model performance...")

evaluations = [res for _, row in data.iterrows() if (res := evaluate_prediction(row))]
if not evaluations:
    st.warning("No games with completed results yet.")
    st.stop()

df_eval = pd.DataFrame(evaluations)
stat_cols = [c for c in df_eval.columns if c.endswith("_acc")]
overall_acc = np.nanmean(df_eval[stat_cols].values.flatten()).round(2)
total_games = len(df_eval)
per_stat_acc = df_eval[stat_cols].mean().sort_values(ascending=False)
df_eval["avg_acc"] = df_eval[stat_cols].mean(axis=1)

# --- METRIC CARDS ---
col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='metric-card'><h3>🎯 Overall Accuracy</h3><h1 style='color:#00FFFF'>{overall_acc}%</h1></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><h3>📊 Games Evaluated</h3><h1 style='color:#00FFFF'>{total_games}</h1
