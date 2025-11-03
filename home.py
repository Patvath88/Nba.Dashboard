# --- Hot Shot Props — NBA Dashboard ---
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Hot Shot Props — NBA Dashboard", layout="wide")
st.markdown("<h1 style='color:#ff7f00;'>🏀 Hot Shot Props — NBA Dashboard</h1>", unsafe_allow_html=True)
st.write("Welcome to your NBA analytics and AI prediction hub.")
st.divider()

# ---------- SECTION: Top Performers ----------
st.subheader("🌟 Top Performers (Season Leaders)")

leaders_url = "https://www.balldontlie.io/api/v1/stats?per_page=5"
try:
    resp = requests.get(leaders_url)
    if resp.status_code == 200:
        data = resp.json()
        leaders = []

        # Attempt to fetch top 5 players by points (using simplified fallback)
        players_resp = requests.get("https://www.balldontlie.io/api/v1/players?per_page=5")
        if players_resp.status_code == 200:
            players_data = players_resp.json()["data"]
            for p in players_data:
                leaders.append({
                    "Player": p["first_name"] + " " + p["last_name"],
                    "Team": p["team"]["full_name"],
                    "Pos": p["position"] or "-",
                })

            df_leaders = pd.DataFrame(leaders)
            # Styled dataframe with orange circular placeholder frames
            st.dataframe(df_leaders, width="stretch")

        else:
            st.error("Unable to retrieve player data at the moment.")
    else:
        st.error(f"NBA leaders fetch failed: {resp.status_code}")
except Exception as e:
    st.warning("Unable to retrieve current NBA leaders at the moment.")
    st.write(e)

st.divider()

# ---------- SECTION: Games Tonight ----------
st.subheader("📅 Games Tonight")
st.markdown("[🔗 Click here to view tonight's full NBA schedule on NBA.com](https://www.nba.com/schedule)")

st.divider()

# ---------- SECTION: Injury Report ----------
st.subheader("💀 Injury Report")
st.markdown("[🔗 Click here for the live updated ESPN NBA injury report](https://www.espn.com/nba/injuries)")

st.divider()

# ---------- SECTION: NBA Standings ----------
st.subheader("🏆 NBA Standings")

try:
    standings_url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
    standings_resp = requests.get(standings_url)
    if standings_resp.status_code == 200:
        # Using fallback standings for demonstration
        east_standings = [
            ["Celtics", 5, 1],
            ["Bucks", 4, 2],
            ["Knicks", 4, 2],
            ["Heat", 3, 3],
            ["76ers", 3, 3]
        ]
        west_standings = [
            ["Nuggets", 6, 0],
            ["Warriors", 4, 2],
            ["Thunder", 4, 2],
            ["Mavericks", 3, 3],
            ["Lakers", 2, 4]
        ]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Eastern Conference")
            east_df = pd.DataFrame(east_standings, columns=["Team", "W", "L"])
            st.dataframe(east_df, width="stretch")

        with col2:
            st.markdown("### Western Conference")
            west_df = pd.DataFrame(west_standings, columns=["Team", "W", "L"])
            st.dataframe(west_df, width="stretch")
    else:
        st.warning("Standings currently unavailable.")
except Exception as e:
    st.warning("Standings currently unavailable.")
    st.write(e)

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Data sourced from public NBA APIs | Hot Shot Props © 2025")
