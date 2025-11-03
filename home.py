import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Hot Shot Props — NBA Dashboard", layout="wide")

# ---------------------- STYLE ----------------------
st.markdown("""
<style>
body { background-color: #0a0a0a; color: white; }
h1, h2, h3, h4 { color: white; }
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
.player-card {
    text-align:center;
    padding: 8px;
}
.player-img {
    border-radius:50%;
    border: 3px solid #E85D04;
    width:110px;
    height:110px;
    object-fit:cover;
    box-shadow:0px 0px 10px #E85D04;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("# 🏀 Hot Shot Props — NBA Dashboard")
st.markdown("Welcome to your NBA analytics and AI prediction hub.")
st.markdown("---")

# ---------------------- TOP PERFORMERS ----------------------
st.markdown("## 🌟 Top Performers (Season Leaders)")

@st.cache_data(ttl=1800)
def get_league_leaders():
    """Fetch top scorers per game from NBA Stats API."""
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
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/"
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        data = resp.json()
        result = data.get("resultSet", data.get("resultSets", [{}]))[0]
        rows = result.get("rowSet", [])
        return rows[:5]  # top 5
    except Exception as e:
        st.error(f"NBA leaders fetch failed: {e}")
        return []

@st.cache_data(ttl=3600)
def get_player_photo(player_name):
    """Return player image from official CDN if available."""
    try:
        # NBA CDN based on player name heuristic fallback
        formatted_name = player_name.lower().replace(" ", "_")
        urls = [
            f"https://nba-players-directory.vercel.app/api/player/{formatted_name}.png",
            f"https://cdn.nba.com/headshots/nba/latest/1040x760/{formatted_name}.png"
        ]
        for url in urls:
            r = requests.get(url, timeout=5)
            if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                return Image.open(BytesIO(r.content))
    except Exception:
        pass
    return None

leaders = get_league_leaders()

if leaders:
    cols = st.columns(5)
    for i, player in enumerate(leaders):
        player_name = player[2]
        team = player[4]
        pts = player[22]
        reb = player[23]
        ast = player[24]
        photo = get_player_photo(player_name)

        with cols[i]:
            st.markdown(f"<div class='player-card'>", unsafe_allow_html=True)
            if photo:
                st.image(photo, use_container_width=False, width=110)
            else:
                st.markdown("<div class='player-img'></div>", unsafe_allow_html=True)
            st.markdown(f"**{player_name}**")
            st.markdown(f"*{team}*")
            st.markdown(f"PTS: **{pts}**  \nREB: **{reb}**  \nAST: **{ast}**")
            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Unable to retrieve current NBA leaders at the moment.")

st.markdown("---")

# ---------------------- GAMES TONIGHT (LINK) ----------------------
st.markdown("## 🗓️ Games Tonight")
st.markdown(
    "[🔗 Click here to view tonight’s full NBA schedule on NBA.com](https://www.nba.com/schedule)",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------------- INJURY REPORT (LINK) ----------------------
st.markdown("## 💀 Injury Report")
st.markdown(
    "[🔗 Click here for the live updated ESPN NBA injury report](https://www.espn.com/nba/injuries)",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------------- NBA STANDINGS ----------------------
st.markdown("## 🏆 NBA Standings")

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
        data = resp.json()
        standings = data["resultSets"][0]["rowSet"]
        east = [t for t in standings if t[5] == "East"]
        west = [t for t in standings if t[5] == "West"]
        return east, west
    except Exception:
        return [], []

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
