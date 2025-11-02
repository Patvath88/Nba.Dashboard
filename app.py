import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

st.set_page_config(page_title="NBA Player Predictor", layout="wide")
st.title("🏀 NBA Player Next Game Predictor (nba_api Version)")

# -----------------------------
# Data helpers
# -----------------------------
@st.cache_data(ttl=3600)
def get_player_id(name):
    p = players.find_players_by_full_name(name)
    if p:
        return p[0]["id"], p[0]["full_name"]
    return None, None

@st.cache_data(ttl=900)
def get_logs(pid):
    df = playergamelog.PlayerGameLog(player_id=pid, season="2024-25").get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE")
    df[["PTS","REB","AST","FG3M","MIN"]] = df[["PTS","REB","AST","FG3M","MIN"]].apply(pd.to_numeric)
    return df

@st.cache_data(ttl=900)
def get_team_defense():
    df = leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
        season="2024-25",
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    return df

# -----------------------------
# Regression (NumPy)
# -----------------------------
def lin_reg(X, y, next_f):
    X = np.c_[np.ones(len(X)), X]
    coeffs = np.linalg.pinv(X.T @ X) @ X.T @ y
    return float(np.dot([1] + next_f, coeffs))

def prep_features(df):
    df["PTS_AVG5"] = df["PTS"].rolling(5,1).mean()
    df["REB_AVG5"] = df["REB"].rolling(5,1).mean()
    df["AST_AVG5"] = df["AST"].rolling(5,1).mean()
    df["FG3M_AVG5"] = df["FG3M"].rolling(5,1).mean()
    return df, ["MIN","PTS_AVG5","REB_AVG5","AST_AVG5","FG3M_AVG5"]

# -----------------------------
# UI
# -----------------------------
name = st.text_input("Enter player name:", "LeBron James")
if name:
    pid, pname = get_player_id(name)
    if pid:
        df = get_logs(pid)
        if not df.empty:
            df, feats = prep_features(df)
            st.subheader(f"{pname} Recent Games")
            st.dataframe(df[["GAME_DATE","PTS","REB","AST","FG3M","MIN"]].tail(10))

            st.sidebar.header("Next Game Inputs")
            mins = st.sidebar.slider("Projected Minutes", 25, 45, 35)
            opp = st.sidebar.text_input("Opponent Team (e.g. Warriors):", "Warriors")

            teams = get_team_defense()
            match = teams[teams["TEAM_NAME"].str.contains(opp, case=False)]
            if not match.empty:
                t = match.iloc[0]
                factors = {
                    "PTS": t["PTS"] / teams["PTS"].mean(),
                    "REB": t["REB"] / teams["REB"].mean(),
                    "AST": t["AST"] / teams["AST"].mean(),
                    "FG3M": t["FG3M"] / teams["FG3M"].mean(),
                }
            else:
                factors = {k:1 for k in ["PTS","REB","AST","FG3M"]}

            last = df.iloc[-1]
            base = [mins,last["PTS_AVG5"],last["REB_AVG5"],last["AST_AVG5"],last["FG3M_AVG5"]]

            preds = {}
            for stat in ["PTS","REB","AST","FG3M"]:
                X, y = df[feats].values, df[stat].values
                preds[stat] = lin_reg(X,y,base)*factors[stat]

            st.subheader("Predicted Next Game Stats")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Points",f"{preds['PTS']:.1f}")
            c2.metric("Rebounds",f"{preds['REB']:.1f}")
            c3.metric("Assists",f"{preds['AST']:.1f}")
            c4.metric("3PT Made",f"{preds['FG3M']:.1f}")

            st.markdown("### Trends (Last 10 Games)")
            c1.plotly_chart(px.line(df.tail(10),x="GAME_DATE",y="PTS",title="Points"),use_container_width=True)
            c2.plotly_chart(px.line(df.tail(10),x="GAME_DATE",y="REB",title="Rebounds"),use_container_width=True)
        else:
            st.warning("No game data found.")
    else:
        st.warning("Player not found.")
