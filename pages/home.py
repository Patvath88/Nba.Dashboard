# -------------------------------------------------
# HOT SHOT PROPS — NBA HOME HUB (ESPN STYLE)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import datetime
from nba_api.stats.endpoints import leagueleaders, leaguestandingsv3, scoreboardv2
from nba_api.stats.static import teams, players
import requests

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Hot Shot Props | NBA Home Hub",
                   page_icon="🏠", layout="wide")

# -------------------------------------------------
# STYLE (lighter dark-gray ESPN look)
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color:#121212;
    color:#EAEAEA;
    font-family:'Roboto',sans-serif;
}
h1,h2,h3,h4 {
    font-family:'Oswald',sans-serif;
    color:#FF6F00;
    text-shadow:0 0 8px #FF9F43;
}
.section {
    background:#1C1C1C;
    border-radius:10px;
    padding:15px;
    margin-bottom:20px;
    box-shadow:0 0 12px rgba(255,111,0,0.1);
}
table {
    color:#FFF !important;
    font-size:0.9em;
}
th {
    background:#2A2A2A !important;
}
.btn {
    background:linear-gradient(90deg,#FF6F00,#FF9100);
    color:white;
    font-weight:bold;
    border:none;
    border-radius:8px;
    padding:10px 20px;
    transition:0.3s;
}
.btn:hover {
    transform:scale(1.03);
    box-shadow:0 0 10px #FF6F00;
}
.status-active {color:#00FF80;font-weight:bold;}
.status-questionable {color:#FFD700;font-weight:bold;}
.status-out {color:#FF5252;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
@st.cache_data(ttl=600)
def get_leaders():
    leaders = leagueleaders.LeagueLeaders(season="2025-26")
    df = leaders.get_data_frames()[0]
    return df

@st.cache_data(ttl=600)
def get_standings():
    standings = leaguestandingsv3.LeagueStandingsV3(season="2025-26")
    df = standings.get_data_frames()[0]
    return df

@st.cache_data(ttl=600)
def get_games_today():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    games = scoreboardv2.ScoreboardV2(game_date=today)
    return games.get_data_frames()

def get_injuries():
    try:
        url = "https://cdn.nba.com/static/json/injury/injury_2025.json"
        data = requests.get(url, timeout=10).json()
        df = pd.DataFrame(data["league"]["injuries"])
        return df
    except Exception:
        return pd.DataFrame()

# -------------------------------------------------
# HEADER / NAVIGATION
# -------------------------------------------------
st.title("🏠 Hot Shot Props — NBA Home Hub")
st.caption("Live season leaders, games, injuries, and standings")

col1, col2 = st.columns([1,1])
with col1:
    if st.button("📊 Go to Player AI Dashboard", type="primary"):
        st.switch_page("app.py")
with col2:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

# -------------------------------------------------
# SECTION 1: SEASON LEADERS
# -------------------------------------------------
st.markdown("## 🏀 Season Leaders")
df = get_leaders()

if not df.empty:
    categories = {
        "Points": "PTS",
        "Rebounds": "REB",
        "Assists": "AST",
        "3PM": "FG3M",
        "PRA": None
    }

    for cat, key in categories.items():
        with st.expander(f"🏆 {cat} Leaders (Top 10)", expanded=False):
            if key:
                leaders = df.nlargest(10, key)[["PLAYER","TEAM","PTS","REB","AST","FG3M"]]
            else:
                df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
                leaders = df.nlargest(10, "PRA")[["PLAYER","TEAM","PTS","REB","AST","PRA"]]
            st.dataframe(leaders, use_container_width=True)
else:
    st.info("Leader data not available.")

# -------------------------------------------------
# SECTION 2: GAMES TONIGHT
# -------------------------------------------------
st.markdown("## 📅 Games Tonight")
try:
    line_score, game_header, *_ = get_games_today()
    if not game_header.empty:
        for _, row in game_header.iterrows():
            st.markdown(
                f"<div class='section'><b>{row['VISITOR_TEAM_NAME']}</b> "
                f"@ <b>{row['HOME_TEAM_NAME']}</b> — "
                f"<i>{row['GAME_STATUS_TEXT']}</i></div>", unsafe_allow_html=True)
    else:
        st.info("No games scheduled tonight.")
except Exception:
    st.warning("Could not load today's games.")

# -------------------------------------------------
# SECTION 3: INJURY REPORT
# -------------------------------------------------
st.markdown("## 💀 Injury Report")
inj_df = get_injuries()
if not inj_df.empty:
    for _, row in inj_df.head(30).iterrows():
        status_class = "status-active"
        if "Out" in row["status"] or "Injured" in row["status"]:
            status_class = "status-out"
        elif "Questionable" in row["status"]:
            status_class = "status-questionable"

        st.markdown(
            f"<div class='section'><b>{row['player']}</b> — {row['team']}<br>"
            f"<span class='{status_class}'>{row['status']}</span> — "
            f"{row.get('description','')}</div>", unsafe_allow_html=True)
else:
    st.info("No injury data currently available.")

# -------------------------------------------------
# SECTION 4: STANDINGS
# -------------------------------------------------
st.markdown("## 🏆 NBA Standings")

stand = get_standings()
if not stand.empty:
    east = stand[stand["Conference"] == "East"]
    west = stand[stand["Conference"] == "West"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Eastern Conference")
        st.dataframe(east[["TeamCity","TeamName","WINS","LOSSES","WinPCT","Streak"]], use_container_width=True)
    with c2:
        st.markdown("### Western Conference")
        st.dataframe(west[["TeamCity","TeamName","WINS","LOSSES","WinPCT","Streak"]], use_container_width=True)
else:
    st.warning("Standings data not available.")

st.markdown("---")
st.caption("⚡ Hot Shot Props NBA Dashboard — Live Data © 2025")
