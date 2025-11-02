# app.py — Hot Shot Props | NBA Player Projections (Fast + Opponent Adjusted + Fixed)
# Final version with safe get_players(), imports, and caching setup

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import altair as alt

# ---------------------- Config ----------------------
st.set_page_config(
    page_title="NBA Player Projections — Hot Shot Props",
    page_icon="🏀",
    layout="wide"
)

BASE_URL = "https://api.balldontlie.io/v1"

# ---------------------- API Helpers ----------------------
@st.cache_data(ttl=60*60)
def get_players():
    """Fetch player list from balldontlie and normalize fields safely."""
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
        if page > 30:
            break

    # Normalize nested JSON → flat DataFrame
    df = pd.json_normalize(players)

    # Ensure required columns always exist
    for col in ["id", "first_name", "last_name"]:
        if col not in df.columns:
            df[col] = ""

    # Handle team name safely (some players have null team)
    if "team.full_name" not in df.columns:
        df["team.full_name"] = ""
    df["team.full_name"] = df["team.full_name"].fillna("Free Agent")

    # Create full name for search
    df["full_name"] = (df["first_name"].astype(str) + " " + df["last_name"].astype(str)).str.strip()

    # Return a clean, uniform DataFrame
    return df.loc[:, ["id", "first_name", "last_name", "full_name", "team.full_name"]].rename(
        columns={"team.full_name": "team"}
    )
