import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(page_title="NBA Player Prediction Dashboard", layout="wide")

st.title("🏀 NBA Player Next Game Predictor")

# -------------------------------
# 1. Get Player Data from API
# -------------------------------
@st.cache_data(ttl=3600)
def get_player_id(player_name):
    url = f"https://www.balldontlie.io/api/v1/players?search={player_name}"
    res = requests.get(url)
    data = res.json()
    if data["data"]:
        return data["data"][0]["id"], data["data"][0]["first_name"] + " " + data["data"][0]["last_name"]
    return None, None

@st.cache_data(ttl=300)
def get_player_stats(player_id):
    stats = []
    page = 1
    while True:
        url = f"https://www.balldontlie.io/api/v1/stats?player_ids[]={player_id}&per_page=100&page={page}"
        res = requests.get(url)
        data = res.json()
        stats.extend(data["data"])
        if not data["meta"]["next_page"]:
            break
        page += 1
    return pd.json_normalize(stats)

# -------------------------------
# 2. User Input
# -------------------------------
player_name = st.text_input("Enter player name (e.g. Luka Doncic):", "Luka Doncic")

if player_name:
    player_id, full_name = get_player_id(player_name)
    if player_id:
        df = get_player_stats(player_id)

        if not df.empty:
            df["pts"] = df["pts"].astype(float)
            df["reb"] = df["reb"].astype(float)
            df["ast"] = df["ast"].astype(float)
            df["min_played"] = df["min"].apply(lambda x: int(x.split(":")[0]) if isinstance(x, str) else 0)
            df = df.sort_values("game.date")

            # -------------------------------
            # 3. Simple Regression Model
            # -------------------------------
            X = df[["min_played"]]
            y = df["pts"]
            model = LinearRegression().fit(X, y)

            next_game_mins = st.slider("Projected minutes next game:", 25, 45, 35)
            predicted_points = model.predict([[next_game_mins]])[0]

            st.subheader(f"Predicted Points for Next Game: {predicted_points:.2f}")

            # -------------------------------
            # 4. Chart Visualization
            # -------------------------------
            fig = px.line(df.tail(10), x="game.date", y="pts", title=f"{full_name} - Last 10 Games (Points)")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df[["game.date", "pts", "reb", "ast", "min_played"]].tail(10))
        else:
            st.warning("No game data found for this player.")
    else:
        st.warning("Player not found.")
