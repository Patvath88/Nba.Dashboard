import streamlit as st
import requests
import datetime

st.set_page_config(page_title="Hot Shot Props — NBA Dashboard", layout="wide")

# ---------------------- STYLE ----------------------
st.markdown("""
<style>
body { background-color: #0a0a0a; color: white; }
.stButton>button {
    border-radius: 8px;
    border: 1px solid #E50914;
    color: white;
    background-color: #1e1e1e;
    padding: 6px 16px;
    font-weight: bold;
    box-shadow: 0px 0px 6px #E50914;
    transition: 0.2s;
}
.stButton>button:hover {
    background-color: #E50914;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- PAGE HEADER ----------------------
st.markdown("# 🏀 Hot Shot Props — NBA Dashboard")
st.markdown("Welcome to your NBA analytics and AI prediction hub.")

# ---------------------- NAVIGATION ----------------------
def go_to_player_page(player_name: str):
    st.query_params["player"] = player_name
    st.switch_page("pages/Player_AI.py")

# ---------------------- TOP PERFORMERS (NBA OFFICIAL) ----------------------
st.markdown("## 🌟 Top Performers (Season Leaders)")

@st.cache_data(ttl=1800)
def get_league_leaders():
    url = "https://stats.nba.com/stats/leagueleaders"
    params = {
        "LeagueID": "00",
        "PerMode": "PerGame",
        "Scope": "S",
        "Season": "2025-26",
        "SeasonType": "Regular Season",
        "StatCategory": "PTS"
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true"
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result_set = data.get("resultSet", data.get("resultSets", [{}]))[0]
        rows = result_set.get("rowSet", [])
        headers_list = result_set.get("headers", [])
        return rows, headers_list
    except Exception:
        return [], []

rows, headers_list = get_league_leaders()

if rows:
    top5 = rows[:5]
    cols = st.columns(5)
    for i, row in enumerate(top5):
        player_name = row[2]
        team_abbr = row[4]
        ppg = row[22]
        with cols[i]:
            st.markdown(f"### {player_name}")
            st.markdown(f"**PPG:** {ppg}  \n**Team:** {team_abbr}")
            if st.button("View Player", key=f"player_{i}"):
                go_to_player_page(player_name)
else:
    st.info("Unable to retrieve live leader stats at the moment. Please try again shortly.")

st.markdown("---")

# ---------------------- GAMES TONIGHT (BUTTON LINK) ----------------------
st.markdown("## 🗓️ Games Tonight")
st.markdown(
    "[Click here to view tonight’s full NBA schedule on NBA.com 🏀](https://www.nba.com/schedule)",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------------- INJURY REPORT (BUTTON LINK) ----------------------
st.markdown("## 💀 Injury Report")
st.markdown(
    "[Click here for the live updated ESPN NBA injury report 💉](https://www.espn.com/nba/injuries)",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------------- STANDINGS (NBA OFFICIAL) ----------------------
@st.cache_data(ttl=1800)
def get_standings():
    url = "https://stats.nba.com/stats/leaguestandingsv3"
    params = {
        "LeagueID": "00",
        "Season": "2025-26",
        "SeasonType": "Regular Season"
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true"
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data["resultSets"][0]["rowSet"]
        east = [t for t in results if t[5] == "East"]
        west = [t for t in results if t[5] == "West"]
        return east, west
    except Exception:
        return [], []

st.markdown("## 🏆 NBA Standings")

east, west = get_standings()
if east or west:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Eastern Conference")
        for t in east[:10]:
            st.markdown(f"{t[3]} — {t[12]}W-{t[13]}L")
    with c2:
        st.markdown("### Western Conference")
        for t in west[:10]:
            st.markdown(f"{t[3]} — {t[12]}W-{t[13]}L")
else:
    st.info("Standings currently unavailable.")
