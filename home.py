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

# ---------------------- NAVIGATION FIX ----------------------
# When user clicks a player button, go to Player_AI.py
def go_to_player_page(player_name: str):
    st.query_params["player"] = player_name
    st.switch_page("pages/Player_AI.py")

# ---------------------- TOP PERFORMERS ----------------------
# ---------------------- TOP PERFORMERS (Patched for NBA Stats headers) ----------------------
st.markdown("## 🌟 Top Performers (Season Leaders)")

@st.cache_data(ttl=1800)
def get_top_performers():
    """Fetch league leaders with realistic browser headers (bypass NBA stats block)."""
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
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "Connection": "keep-alive",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true"
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result_set = data.get("resultSet", data.get("resultSets", [{}]))[0]
        headers_list = result_set.get("headers", [])
        rows = result_set.get("rowSet", [])
        return rows, headers_list
    except Exception as e:
        st.error(f"NBA API Error: {e}")
        return [], []

rows, headers_list = get_top_performers()

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
                st.query_params["player"] = player_name
                st.switch_page("pages/Player_AI.py")
else:
    st.info("Could not retrieve live stats from NBA servers. Please try again shortly.")


# ---------------------- GAMES TONIGHT ----------------------
@st.cache_data(ttl=900)
def get_todays_games():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        resp = requests.get("https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json", timeout=10)
        data = resp.json()
        games = [g for g in data["leagueSchedule"]["gameDates"] if g["gameDate"] == today]
        if not games:
            return []
        games_list = []
        for g in games[0]["games"]:
            games_list.append(f"{g['awayTeam']['teamName']} @ {g['homeTeam']['teamName']} ({g['gameTimeUTC'][11:16]} UTC)")
        return games_list
    except Exception:
        return []

st.markdown("## 🗓️ Games Tonight")
games = get_todays_games()
if games:
    for g in games:
        st.markdown(f"- {g}")
else:
    st.info("No games tonight.")

st.markdown("---")

# ---------------------- INJURY REPORT ----------------------
@st.cache_data(ttl=900)
def get_injuries():
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news"
    try:
        data = requests.get(url, timeout=10).json()
        injuries = [a for a in data.get("articles", []) if "injury" in a.get("type", "").lower()]
        return injuries[:10]
    except Exception:
        return []

st.markdown("## 💀 Injury Report")
injuries = get_injuries()
if injuries:
    for inj in injuries:
        st.markdown(f"**{inj['headline']}** — {inj['description'][:100]}...")
else:
    st.info("No injury data currently available.")

st.markdown("---")

# ---------------------- STANDINGS (if already present, this keeps it) ----------------------
try:
    standings_url = "https://stats.nba.com/stats/leaguestandingsv3?LeagueID=00&Season=2025-26&SeasonType=Regular%20Season"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(standings_url, headers=headers, timeout=10).json()
    results = resp["resultSets"][0]["rowSet"]
    east = [t for t in results if t[5] == "East"]
    west = [t for t in results if t[5] == "West"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Eastern Conference")
        for t in east[:10]:
            st.markdown(f"{t[3]} — {t[12]}W-{t[13]}L")
    with c2:
        st.markdown("### Western Conference")
        for t in west[:10]:
            st.markdown(f"{t[3]} — {t[12]}W-{t[13]}L")
except Exception as e:
    st.info("Standings currently unavailable.")
