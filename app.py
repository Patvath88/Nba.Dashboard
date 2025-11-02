import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px


st.set_page_config(page_title="NBA Player Prediction Dashboard", layout="wide")

st.title("🏀 NBA Player Next Game Predictor (Full Regression Model)")

# -------------------------------
# Helper Functions
# -------------------------------
@st.cache_data(ttl=3600)
def get_player_id(player_name):
    url = f"https://www.balldontlie.io/api/v1/players?search={player_name}"
    res = requests.get(url)
    data = res.json()
    if data["data"]:
        player = data["data"][0]
        return player["id"], player["first_name"] + " " + player["last_name"]
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
    df = pd.json_normalize(stats)
    return df

# -------------------------------
# Data Preparation
# -------------------------------
def prepare_features(df):
    df = df.copy()
    df["min_played"] = df["min"].apply(lambda x: int(x.split(":")[0]) if isinstance(x, str) else 0)
    df["home_game"] = df["game.home_team_id"] == df["team.id"]
    df["home_game"] = df["home_game"].astype(int)
    df["pts"] = df["pts"].astype(float)
    df["reb"] = df["reb"].astype(float)
    df["ast"] = df["ast"].astype(float)
    df["fg3m"] = df["fg3m"].astype(float)

    # Rolling averages (last 5 games)
    for stat in ["pts", "reb", "ast", "fg3m"]:
        df[f"{stat}_avg5"] = df[stat].rolling(5, min_periods=1).mean()

    # Feature columns
    features = ["min_played", "home_game", "pts_avg5", "reb_avg5", "ast_avg5", "fg3m_avg5"]
    df = df.dropna(subset=features)
    return df, features

# -------------------------------
# Regression Model Training
# -------------------------------
def train_and_predict(df, features, next_game_features):
    models = {}
    predictions = {}

    for target in ["pts", "reb", "ast", "fg3m"]:
        X = df[features]
        y = df[target]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression().fit(X_scaled, y)
        models[target] = model
        next_game_scaled = scaler.transform([next_game_features])
        predictions[target] = model.predict(next_game_scaled)[0]

    return predictions

# -------------------------------
# Streamlit UI
# -------------------------------
player_name = st.text_input("Enter player name (e.g. Luka Doncic):", "Luka Doncic")

if player_name:
    player_id, full_name = get_player_id(player_name)
    if player_id:
        df = get_player_stats(player_id)

        if not df.empty:
            df, features = prepare_features(df)
            df = df.sort_values("game.date")

            st.subheader(f"📈 {full_name} Recent Performance")
            st.dataframe(df[["game.date", "pts", "reb", "ast", "fg3m", "min_played"]].tail(10))

            # User-adjustable inputs for next game
            st.sidebar.header("Next Game Factors")
            next_mins = st.sidebar.slider("Projected Minutes", 25, 45, 35)
            home_game = 1 if st.sidebar.checkbox("Home Game?", value=True) else 0

            last_row = df.iloc[-1]
            next_game_features = [
                next_mins,
                home_game,
                last_row["pts_avg5"],
                last_row["reb_avg5"],
                last_row["ast_avg5"],
                last_row["fg3m_avg5"],
            ]

            preds = train_and_predict(df, features, next_game_features)

            st.subheader("🎯 Predicted Next Game Stats")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Points", f"{preds['pts']:.1f}")
            col2.metric("Rebounds", f"{preds['reb']:.1f}")
            col3.metric("Assists", f"{preds['ast']:.1f}")
            col4.metric("3PT Made", f"{preds['fg3m']:.1f}")

            # Charts
            st.markdown("### Trend Charts (Last 10 Games)")
            col1, col2 = st.columns(2)
            fig_pts = px.line(df.tail(10), x="game.date", y="pts", title="Points")
            fig_reb = px.line(df.tail(10), x="game.date", y="reb", title="Rebounds")
            col1.plotly_chart(fig_pts, use_container_width=True)
            col2.plotly_chart(fig_reb, use_container_width=True)

            col3, col4 = st.columns(2)
            fig_ast = px.line(df.tail(10), x="game.date", y="ast", title="Assists")
            fig_3pt = px.line(df.tail(10), x="game.date", y="fg3m", title="3PT Made")
            col3.plotly_chart(fig_ast, use_container_width=True)
            col4.plotly_chart(fig_3pt, use_container_width=True)

        else:
            st.warning("No game data found for this player.")
    else:
        st.warning("Player not found.")
