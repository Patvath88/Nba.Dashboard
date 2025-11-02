# app.py — Hot Shot Props | NBA Player Projections (Fast Version)
# Uses balldontlie.io instead of stats.nba.com (instant, no rate limits)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import altair as alt

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="NBA Player Projections — Hot Shot Props", page_icon="🏀", layout="wide")

# ---------------------- API Helpers ----------------------
BASE_URL = "https://api.balldontlie.io/v1"

@st.cache_data(ttl=60*30)
def get_players():
    players = []
    page = 1
    while True:
        res = requests.get(f"{BASE_URL}/players", params={"per_page": 100, "page": page})
        if res.status_code != 200:
            break
        data = res.json()["data"]
        if not data:
            break
        players.extend(data)
        page += 1
    return pd.DataFrame(players)

@st.cache_data(ttl=60*30)
def get_player_stats(player_id: int, last_n=20):
    today = dt.date.today()
    start = today - dt.timedelta(days=90)
    url = f"{BASE_URL}/stats"
    res = requests.get(url, params={"player_ids[]": player_id, "per_page": 100, "start_date": start, "end_date": today})
    if res.status_code != 200:
        return pd.DataFrame()
    df = pd.json_normalize(res.json()["data"])
    if df.empty:
        return pd.DataFrame()
    df["game_date"] = pd.to_datetime(df["game.date"])
    df = df.sort_values("game_date", ascending=False).head(last_n)
    df["PTS"] = df["pts"]; df["REB"] = df["reb"]; df["AST"] = df["ast"]; df["3PM"] = df["fg3m"]
    return df[["game_date","PTS","REB","AST","3PM"]]

# ---------------------- Model ----------------------
def james_stein(mean, league_mean, n, var):
    var = max(var, 1e-6)
    k = var / (var + max(n,1))
    return (1 - k)*mean + k*league_mean

def ensemble_projection(df):
    if df.empty:
        return {"PTS":0,"REB":0,"AST":0,"3PM":0}

    out = {}
    league_avg = df[["PTS","REB","AST","3PM"]].mean().to_dict()
    for stat in ["PTS","REB","AST","3PM"]:
        vals = pd.to_numeric(df[stat], errors="coerce").dropna()
        if vals.empty:
            out[stat] = 0
            continue
        # Recency windows
        samples = []
        for w, wgt in [(5,1.0),(10,0.8),(20,0.6)]:
            sub = vals.head(w)
            if not sub.empty:
                samples.append((sub.mean(), wgt/(sub.var(ddof=1)+1)))
        num = sum(m*w for m,w in samples)
        den = sum(w for _,w in samples)
        base = num/den if den>0 else 0
        var = vals.var(ddof=1)
        out[stat] = round(james_stein(base, league_avg[stat], len(vals), var), 2)
    return out

# ---------------------- Sidebar ----------------------
st.sidebar.title("🏀 NBA Dashboard (Fast Mode)")
players_df = get_players()

query = st.sidebar.text_input("Search player", placeholder="e.g., Jayson Tatum")
filtered = players_df[players_df["first_name"].str.contains(query, case=False, na=False) | 
                      players_df["last_name"].str.contains(query, case=False, na=False)] if query else players_df

if filtered.empty:
    st.warning("No players found.")
    st.stop()

player = st.sidebar.selectbox("Select Player", sorted(filtered["first_name"] + " " + filtered["last_name"]))
player_id = int(filtered.loc[(filtered["first_name"] + " " + filtered["last_name"]) == player, "id"].iloc[0])

# ---------------------- Main ----------------------
st.header(player)
with st.spinner("Fetching data..."):
    logs = get_player_stats(player_id)

if logs.empty:
    st.error("No recent games found for this player.")
    st.stop()

proj = ensemble_projection(logs)
st.subheader("Next Game Projection (Regression-Proof Model)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Points", proj["PTS"])
col2.metric("Rebounds", proj["REB"])
col3.metric("Assists", proj["AST"])
col4.metric("3PM", proj["3PM"])

# ---------------------- Chart ----------------------
melted = logs.melt("game_date", value_vars=["PTS","REB","AST","3PM"], var_name="Stat", value_name="Value")
chart = (
    alt.Chart(melted)
    .mark_line(point=True)
    .encode(
        x="game_date:T",
        y="Value:Q",
        color="Stat:N",
        tooltip=["game_date:T","Stat","Value:Q"]
    )
    .properties(height=300)
)
st.altair_chart(chart, use_container_width=True)

st.caption("Fast API version — built for Hot Shot Props ⚡")
