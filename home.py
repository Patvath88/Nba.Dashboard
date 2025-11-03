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

def go_to_player_page(player_name: str):
    st.query_params["player"] = player_name
    st.switch_page("pages/Player_AI.py")

# ---------------------- TOP PERFORMERS (BALLEDONTLIE FALLBACK) ----------------------
st.markdown("## 🌟 Top Performers (Season Leaders)")

@st.cache_data(ttl=1800)
def get_top_performers():
    try:
        # Using balldontlie.io for free reliable stats
        resp = requests.get("https://www.balldontlie.io/api/v1/season_averages?season=2025", timeout=10).json()
        data = resp.get("data", [])
        if not data:
            return []
        # Sort by PTS descending
        top = sorted(data, key=lambda x: x.get("pts", 0), reverse=True)[:5]
        return top
    except Exception:
        return []

players = get_top_performers()
if players:
    cols = st.columns(5)
    for i, p in enumerate(players):
        pid = p.get("player_id")
        player_name = f"{p.get('player', {}).get('first_name', '')} {p.get('player', {}).get('last_name', '')}".strip()
        with cols[i]:
            st.markdown(f"### {player_name}")
            st.markdown(f"**PTS:** {p.get('pts', 0)}  \n**REB:** {p.get('reb', 0)}  \n**AST:** {p.get('ast', 0)}")
            if st.button("View Player", key=f"player_{i}"):
                go_to_player_page(player_name)
else:
    st.info("Unable to retrieve live leader stats at the moment. Please try again shortly.")

st.markdown("---")

# ---------------------- GAMES TONIGHT (3-DAY LOOKAHEAD) ----------------------
@st.cache_data(ttl=900)
def get_upcoming_games():
    try:
        resp = requests.get("https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json", timeout=10)
        data = resp.json()
        today = datetime.datetime.now()
        upcoming = []
        for d in data["leagueSchedule"]["gameDates"]:
            date_obj = datetime.datetime.strptime(d["gameDate"], "%Y-%m-%d")
            if 0 <= (date_obj - today).days <= 3:
                for g in d["games"]:
                    home = g["homeTeam"]["teamName"]
                    away = g["awayTeam"]["teamName"]
                    time = g["gameTimeUTC"][11:16]
                    upcoming.append(f"{away} @ {home} ({time} UTC)")
        return upcoming
    except Exception:
        return []

st.markdown("## 🗓️ Games Tonight")
games = get_upcoming_games()
if games:
    for g in games:
        st.markdown(f"- {g}")
else:
    st.info("No scheduled games in the next 3 days.")

st.markdown("---")

# ---------------------- INJURY REPORT (ESPN ROSTER API) ----------------------
@st.cache_data(ttl=900)
def get_injuries():
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
        resp = requests.get(url, timeout=10).json()
        injuries = []
        for team in resp.get("sports", [])[0].get("leagues", [])[0].get("teams", []):
            for p in team.get("team", {}).get("injuries", []):
                name = p.get("athlete", {}).get("displayName", "")
                status = p.get("status", "")
                desc = p.get("description", "")
                injuries.append(f"**{name}** — {status} ({desc})")
        return injuries
    except Exception:
        return []

st.markdown("## 💀 Injury Report")
injuries = get_injuries()
if injuries:
    for i in injuries[:10]:
        st.markdown(i)
else:
    st.info("No injury data currently available.")

st.markdown("---")

# ---------------------- STANDINGS (BALLEDONTLIE TEAMS FALLBACK) ----------------------
@st.cache_data(ttl=1800)
def get_standings():
    try:
        resp = requests.get("https://www.balldontlie.io/api/v1/teams", timeout=10).json()
        data = resp.get("data", [])
        # Fake sort: East vs West split alphabetically for display
        east = sorted([t for t in data if t["conference"] == "East"], key=lambda x: x["full_name"])
        west = sorted([t for t in data if t["conference"] == "West"], key=lambda x: x["full_name"])
        return east, west
    except Exception:
        return [], []

st.markdown("## 🏆 NBA Standings (Demo Order)")
east, west = get_standings()
if east or west:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Eastern Conference")
        for t in east:
            st.markdown(f"{t['full_name']}")
    with c2:
        st.markdown("### Western Conference")
        for t in west:
            st.markdown(f"{t['full_name']}")
else:
    st.info("Standings currently unavailable.")
