# app.py — Hot Shot Props | NBA Player Projections (Fast + Opponent Adjusted + Fixed)
# Fixes KeyError: 'first_name' by normalizing player data

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import altair as alt

# ---------------------- Config ----------------------
st.set_page_config(page_title="NBA Player Projections — Hot Shot Props", page_icon="🏀", layout="wide")
BASE_URL = "https://api.balldontlie.io/v1"

# ---------------------- API Helpers ----------------------
@st.cache_data(ttl=60*60)
def get_players():
    """Fetch full player list from Balldontlie and normalize column names."""
    players = []
    page = 1
    while True:
        res = requests.get(f"{BASE_URL}/players", params={"per_page": 100, "page": page})
        if res.status_code != 200:
            break
        data = res.json().get("data", [])
        if not data:
            break
        players.extend(data)
        page += 1
        if page > 30:  # safety break
            break
    df = pd.json_normalize(players)
    # Ensure required columns exist
    if "first_name" not in df.columns or "last_name" not in df.columns:
        df["first_name"] = df.get("first_name", "")
        df["last_name"] = df.get("last_name", "")
    df["full_name"] = df["first_name"].astype(str) + " " + df["last_name"].astype(str)
    return df[["id", "first_name", "last_name", "full_name", "team.full_name"]].rename(columns={"team.full_name":"team"})

@st.cache_data(ttl=60*30)
def get_player_stats(player_id: int, last_n=20):
    today = dt.date.today()
    start = today - dt.timedelta(days=90)
    url = f"{BASE_URL}/stats"
    res = requests.get(url, params={"player_ids[]": player_id, "per_page": 100, "start_date": start, "end_date": today})
    if res.status_code != 200:
        return pd.DataFrame()
    data = res.json().get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.json_normalize(data)
    df["game_date"] = pd.to_datetime(df["game.date"])
    df = df.sort_values("game_date", ascending=False).head(last_n)
    df["PTS"] = df["pts"]; df["REB"] = df["reb"]; df["AST"] = df["ast"]; df["3PM"] = df["fg3m"]
    df["opp_team"] = df["game.visitor_team.full_name"]
    df.loc[df["team.id"] == df["game.visitor_team.id"], "opp_team"] = df["game.home_team.full_name"]
    return df[["game_date","PTS","REB","AST","3PM","opp_team"]]

@st.cache_data(ttl=60*60)
def get_team_avg_allowed():
    """Approximate opponent defensive stats using recent 60 days of data."""
    today = dt.date.today()
    start = today - dt.timedelta(days=60)
    all_stats = []
    for page in range(1, 8):
        url = f"{BASE_URL}/stats"
        res = requests.get(url, params={"per_page": 100, "page": page, "start_date": start, "end_date": today})
        if res.status_code != 200:
            break
        data = res.json().get("data", [])
        if not data:
            break
        all_stats.extend(data)
    df = pd.json_normalize(all_stats)
    if df.empty:
        return pd.DataFrame()
    df["team_name"] = df["team.full_name"]
    grouped = df.groupby("team_name")[["pts","reb","ast","fg3m"]].mean().reset_index()
    grouped.rename(columns={"pts":"ALLOW_PTS","reb":"ALLOW_REB","ast":"ALLOW_AST","fg3m":"ALLOW_3PM"}, inplace=True)
    return grouped

# ---------------------- Model ----------------------
def james_stein(mean, league_mean, n, var):
    var = max(var, 1e-6)
    k = var / (var + max(n,1))
    return (1 - k)*mean + k*league_mean

def ensemble_projection(df, opp_name, team_allow_df):
    if df.empty:
        return {"PTS":0,"REB":0,"AST":0,"3PM":0}
    out = {}
    league_avg = df[["PTS","REB","AST","3PM"]].mean().to_dict()

    # Opponent adjustment
    opp_adj = {"PTS":1.0,"REB":1.0,"AST":1.0,"3PM":1.0}
    if not team_allow_df.empty and opp_name in team_allow_df["team_name"].values:
        opp = team_allow_df.loc[team_allow_df["team_name"] == opp_name].iloc[0]
        mean_vals = team_allow_df[["ALLOW_PTS","ALLOW_REB","ALLOW_AST","ALLOW_3PM"]].mean()
        for stat, key in [("PTS","ALLOW_PTS"),("REB","ALLOW_REB"),("AST","ALLOW_AST"),("3PM","ALLOW_3PM")]:
            opp_adj[stat] = min(1.15, max(0.85, opp[key]/mean_vals[key]))

    for stat in ["PTS","REB","AST","3PM"]:
        vals = pd.to_numeric(df[stat], errors="coerce").dropna()
        if vals.empty:
            out[stat] = 0
            continue
        samples = []
        for w, wgt in [(5,1.0),(10,0.8),(20,0.6)]:
            sub = vals.head(w)
            if not sub.empty:
                samples.append((sub.mean(), wgt/(sub.var(ddof=1)+1)))
        num = sum(m*w for m,w in samples)
        den = sum(w for _,w in samples)
        base = num/den if den>0 else 0
        var = vals.var(ddof=1)
        adj = james_stein(base, league_avg[stat], len(vals), var) * opp_adj[stat]
        out[stat] = round(adj, 2)
    return out

# ---------------------- Sidebar ----------------------
st.sidebar.title("🏀 NBA Dashboard (Fast + Opponent Adjusted)")
players_df = get_players()

query = st.sidebar.text_input("Search player", placeholder="e.g., Luka Doncic")

if query:
    mask = players_df["full_name"].str.contains(query, case=False, na=False)
    filtered = players_df[mask]
else:
    filtered = players_df

if filtered.empty:
    st.warning("No players found.")
    st.stop()

player = st.sidebar.selectbox("Select Player", sorted(filtered["full_name"].tolist()))
player_id = int(filtered.loc[filtered["full_name"] == player, "id"].iloc[0])

# ---------------------- Main ----------------------
st.header(player)
with st.spinner("Fetching player and opponent data..."):
    logs = get_player_stats(player_id)
    team_allow = get_team_avg_allowed()

if logs.empty:
    st.error("No recent games found for this player.")
    st.stop()

last_opp = logs["opp_team"].iloc[0] if "opp_team" in logs.columns else "Unknown"
proj = ensemble_projection(logs, last_opp, team_allow)

st.subheader(f"Next Game Projection vs {last_opp} (Regression-Proof Model)")
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
    .encode(x="game_date:T", y="Value:Q", color="Stat:N", tooltip=["game_date:T","Stat","Value:Q"])
    .properties(height=300)
)
st.altair_chart(chart, use_container_width=True)

st.caption("Built for Hot Shot Props ⚡ Fast, regression-proof, and opponent-aware.")
