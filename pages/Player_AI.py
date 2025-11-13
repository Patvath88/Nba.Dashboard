
# pages/Player_AI.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Player AI Dashboard", page_icon="📊")

st.title("📊 Player AI Dashboard")
st.markdown("""
Welcome to the **AI-powered NBA Player Analytics Dashboard**.  
Select a player below to view real stats and AI-predicted performance metrics.
""")

# ---------- SAMPLE DATA ----------
# You can replace this with real player data later
players = ["LeBron James", "Stephen Curry", "Jayson Tatum", "Giannis Antetokounmpo"]
data = {
    "LeBron James": {"PTS": 25.3, "AST": 7.8, "REB": 8.1},
    "Stephen Curry": {"PTS": 29.7, "AST": 6.4, "REB": 4.3},
    "Jayson Tatum": {"PTS": 27.1, "AST": 4.6, "REB": 8.8},
    "Giannis Antetokounmpo": {"PTS": 31.1, "AST": 5.9, "REB": 11.6},
}

# ---------- PLAYER SELECTION ----------
selected_player = st.selectbox("Select a player:", players)

# ---------- DISPLAY CURRENT STATS ----------
player_stats = data[selected_player]
st.subheader(f"📈 Current Season Stats for {selected_player}")
st.metric("Points Per Game", player_stats["PTS"])
st.metric("Assists Per Game", player_stats["AST"])
st.metric("Rebounds Per Game", player_stats["REB"])

# ---------- SIMPLE AI PREDICTION MODEL ----------
# (Simulate AI predictions using noise — replace with ML model later)
st.divider()
st.subheader("🤖 AI Performance Prediction")

np.random.seed(42)
predictions = {
    stat: round(val + np.random.normal(0, 1), 1)
    for stat, val in player_stats.items()
}

# Show predicted stats
cols = st.columns(3)
for i, stat in enumerate(predictions):
    with cols[i]:
        st.metric(f"Predicted {stat}", predictions[stat])

# ---------- CHART VISUALIZATION ----------
st.divider()
st.subheader("📊 Comparison: Actual vs Predicted Stats")

actual_vals = list(player_stats.values())
predicted_vals = list(predictions.values())

fig, ax = plt.subplots()
x = np.arange(len(player_stats))
width = 0.35
ax.bar(x - width/2, actual_vals, width, label="Actual")
ax.bar(x + width/2, predicted_vals, width, label="Predicted")
ax.set_xticks(x)
ax.set_xticklabels(player_stats.keys())
ax.legend()
st.pyplot(fig)

st.success("✅ AI prediction generated successfully!")
