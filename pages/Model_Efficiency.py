import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import time

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="🧠 Model Efficiency Dashboard", layout="wide")
st.title("📊 AI Model Efficiency & Backtesting Overview")

# ---------------------- LOAD SAVED PROJECTIONS ----------------------
path = "saved_projections.csv"
try:
    data = pd.read_csv(path)
except FileNotFoundError:
    st.warning("No saved projections found. Generate some predictions first.")
    st.stop()

if data.empty:
    st.info("No saved projection data available.")
    st.stop()

nba_players = players.get_active_players()
player_map = {p["full_name"]: p["id"] for p in nba_players}

# ---------------------- HELPERS ----------------------
@st.cache_data(ttl=120)
def get_gamelog(pid):
    """Return full season game log."""
    try:
        gl = playergamelog.PlayerGameLog(player_id=pid, season="2025-26").get_data_frames()[0]
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        return gl
    except Exception:
        return pd.DataFrame()

def evaluate_prediction(row):
    """Compare AI prediction vs actual."""
    pid = player_map.get(row["player"])
    if not pid:
        return None
    gl = get_gamelog(pid)
    if gl.empty:
        return None
    # Match by date
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

# ---------------------- EVALUATE ALL ----------------------
st.info("Analyzing backtested projections… this may take a few seconds.")
evaluations = []
for _, row in data.iterrows():
    res = evaluate_prediction(row)
    if res:
        evaluations.append(res)

if not evaluations:
    st.warning("No completed games found to evaluate yet.")
    st.stop()

df_eval = pd.DataFrame(evaluations)

# ---------------------- SUMMARY STATS ----------------------
stat_cols = [c for c in df_eval.columns if c.endswith("_acc")]
all_acc_values = df_eval[stat_cols].values.flatten()
overall_mean = np.nanmean(all_acc_values).round(2)
total_games = len(df_eval)

col1, col2, col3 = st.columns(3)
col1.metric("🎯 Overall Model Accuracy", f"{overall_mean}%")
col2.metric("📊 Games Evaluated", total_games)
col3.metric("🧩 Stats per Game", len(stat_cols))

# ---------------------- PER-PLAYER PERFORMANCE ----------------------
st.markdown("### 🏀 Player Accuracy Summary")
player_acc = df_eval.groupby("player")[stat_cols].mean().round(1).reset_index()
player_acc["Overall"] = player_acc[stat_cols].mean(axis=1).round(1)
player_acc = player_acc.sort_values("Overall", ascending=False)
st.dataframe(player_acc, use_container_width=True, hide_index=True)

# ---------------------- PER-STAT ACCURACY ----------------------
st.markdown("### 📈 Average Accuracy by Statistic")
stat_means = df_eval[stat_cols].mean().sort_values(ascending=False)
fig_stat = go.Figure()
fig_stat.add_trace(go.Bar(x=[s.replace("_acc", "") for s in stat_means.index],
                          y=stat_means.values,
                          marker_color="#00FFFF"))
fig_stat.update_layout(
    title="Average Accuracy by Stat",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    height=350,
    margin=dict(l=30, r=30, t=40, b=30)
)
st.plotly_chart(fig_stat, use_container_width=True, key="per_stat_accuracy")

# ---------------------- ACCURACY OVER TIME ----------------------
st.markdown("### ⏳ Accuracy Trend Over Time")
df_eval["avg_acc"] = df_eval[stat_cols].mean(axis=1)
df_eval["game_date"] = pd.to_datetime(df_eval["game_date"])
trend = df_eval.groupby("game_date")["avg_acc"].mean().reset_index()

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=trend["game_date"], y=trend["avg_acc"], mode="lines+markers",
    line=dict(width=2, color="#E50914"), marker=dict(size=8)
))
fig_trend.update_layout(
    title="Average Model Accuracy Over Time",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    height=350,
    margin=dict(l=30, r=30, t=40, b=30)
)
st.plotly_chart(fig_trend, use_container_width=True, key="accuracy_trend")

# ---------------------- ERROR DISTRIBUTION ----------------------
st.markdown("### 📉 Prediction Error Distribution")
err_cols = [c for c in df_eval.columns if c.endswith("_err")]
errors = df_eval[err_cols].values.flatten()
fig_err = go.Figure()
fig_err.add_trace(go.Histogram(x=errors, nbinsx=40, marker_color="#E50914"))
fig_err.update_layout(
    title="Distribution of Prediction Errors (Actual - Predicted)",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    height=350,
    margin=dict(l=30, r=30, t=40, b=30)
)
st.plotly_chart(fig_err, use_container_width=True, key="error_distribution")

# ---------------------- TOP/BOTTOM PLAYERS ----------------------
st.markdown("### 🏆 Model Predictability Rankings")
top_players = player_acc.head(5)
bottom_players = player_acc.tail(5)
col_top, col_bottom = st.columns(2)

with col_top:
    st.markdown("#### 🔝 Most Predictable Players")
    st.table(top_players[["player", "Overall"]])

with col_bottom:
    st.markdown("#### ⚠️ Least Predictable Players")
    st.table(bottom_players[["player", "Overall"]])

st.markdown("---")
st.success("✅ Model efficiency report generated successfully.")
