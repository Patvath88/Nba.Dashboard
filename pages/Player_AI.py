# pages/Player_AI.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression

# ---------------------------
# PAGE CONFIGURATION
# ---------------------------
st.set_page_config(page_title="Player AI Dashboard", page_icon="📊")

st.title("📊 Player AI Dashboard — Real NBA Data")
st.markdown("""
This dashboard uses **real NBA data** (via [balldontlie.io](https://www.balldontlie.io))  
and a simple AI model to predict next-game performance.
""")

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

@st.cache_data(show_spinner=False)
def get_player_id(player_name: str):
    """Fetch player ID from balldontlie API."""
    url = f"https://www.balldontlie.io/api/v1/players?search={player_name}"
    r = requests.get(url)
    data = r.json()
    if data["data"]:
        return data["data"][0]["id"]
    else:
        return None


@st.cache_data(show_spinner=False)
def get_player_game_stats(player_id: int):
    """Fetch last 20 games for player."""
    url = f"https://www.balldontlie.io/api/v1/stats?player_ids[]={player_id}&per_page=20"
    r = requests.get(url)
    games = r.json()["data"]
    if not games:
        return pd.DataFrame()
    
    df = pd.DataFrame([
        {
            "Game": i + 1,
            "PTS": g["pts"],
            "REB": g["reb"],
            "AST": g["ast"],
            "STL": g["stl"],
            "BLK": g["blk"]
        }
        for i, g in enumerate(games)
    ])
    df = df.iloc[::-1].reset_index(drop=True)  # chronological order
    return df


def predict_next_game(df: pd.DataFrame):
    """Use simple regression per stat to predict next game performance."""
    predictions = {}
    for stat in ["PTS", "REB", "AST", "STL", "BLK"]:
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[stat].values
        model = LinearRegression()
        model.fit(X, y)
        next_game = model.predict([[len(df)]])[0]
        predictions[stat] = round(next_game, 1)
    return predictions


# ---------------------------
# USER INPUT
# ---------------------------

players = [
    "LeBron James", "Stephen Curry", "Jayson Tatum", "Giannis Antetokounmpo",
    "Luka Doncic", "Kevin Durant", "Nikola Jokic", "Shai Gilgeous-Alexander"
]

selected_player = st.selectbox("Select a player:", players)

# ---------------------------
# DATA RETRIEVAL
# ---------------------------

player_id = get_player_id(selected_player)

if player_id is None:
    st.error("❌ Player not found. Try a different name.")
    st.stop()

stats_df = get_player_game_stats(player_id)

if stats_df.empty:
    st.warning("No recent stats found for this player.")
    st.stop()

# ---------------------------
# DISPLAY STATS SUMMARY
# ---------------------------

st.subheader(f"📈 Recent Performance for {selected_player}")
st.dataframe(stats_df.tail(10), use_container_width=True)

mean_stats = stats_df.mean().round(1)
cols = st.columns(5)
for i, stat in enumerate(["PTS", "REB", "AST", "STL", "BLK"]):
    with cols[i]:
        st.metric(stat, mean_stats[stat])

# ---------------------------
# AI PREDICTION
# ---------------------------

st.divider()
st.subheader("🤖 AI Predicted Next Game Performance")

predicted_stats = predict_next_game(stats_df)

cols = st.columns(5)
for i, stat in enumerate(predicted_stats):
    with cols[i]:
        st.metric(f"Predicted {stat}", predicted_stats[stat])

# ---------------------------
# VISUALIZATION
# ---------------------------

st.divider()
st.subheader("📊 Actual vs Predicted Comparison")

fig, ax = plt.subplots()
actual = [mean_stats[s] for s in predicted_stats]
predicted = [predicted_stats[s] for s in predicted_stats]

x = np.arange(len(predicted_stats))
width = 0.35
ax.bar(x - width/2, actual, width, label="Actual (avg last 10 games)")
ax.bar(x + width/2, predicted, width, label="Predicted Next Game")
ax.set_xticks(x)
ax.set_xticklabels(predicted_stats.keys())
ax.legend()
ax.set_ylabel("Stat Value")
st.pyplot(fig)

st.success(f"✅ AI prediction for {selected_player} generated successfully!")

