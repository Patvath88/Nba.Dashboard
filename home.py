# home.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
from nba_api.stats.endpoints import leagueleaders, leaguestandingsv3, scoreboardv2
from nba_api.stats.static import players, teams
from zoneinfo import ZoneInfo
import os

# ---------- CONFIG ----------
st.set_page_config(page_title="Hot Shot Props | NBA Home Hub",
                   page_icon="🏀", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
body {background:#121212;color:#EAEAEA;font-family:'Roboto',sans-serif;}
h1,h2,h3 {color:#FF6F00;text-shadow:0 0 8px #FF9F43;font-family:'Oswald',sans-serif;}
.section {background:#1C1C1C;border-radius:12px;padding:15px;margin-bottom:20px;
          box-shadow:0 0 12px rgba(255,111,0,0.1);}
.leader {display:flex;align-items:center;gap:18px;margin-bottom:18px;}
.leader img {width:110px;height:110px;border-radius:50%;border:3px solid #FF6F00;object-fit:cover;}
.status-active{color:#00FF80;font-weight:bold;}
.status-questionable{color:#FFD700;font-weight:bold;}
.status-out{color:#FF5252;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
@st.cache_data(ttl=600)
def get_leaders():
    df = leagueleaders.LeagueLeaders(season="2025-26").get_data_frames()[0]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

@st.cache_data(ttl=600)
def get_standings():
    return leaguestandingsv3.LeagueStandingsV3(season="2025-26").get_data_frames()[0]

@st.cache_data(ttl=600)
def get_games_today():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return scoreboardv2.ScoreboardV2(game_date=today).get_data_frames()

@st.cache_data(ttl=600)
def get_injuries():
    try:
        url="https://cdn.nba.com/static/json/injury/injury_2025.json"
        return pd.DataFrame(requests.get(url,timeout=10).json()["league"]["injuries"])
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def player_id_map():
    return {p["full_name"]:p["id"] for p in players.get_active_players()}

def player_photo(name):
    pid = player_id_map().get(name)
    if pid:
        return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
    return "https://cdn-icons-png.flaticon.com/512/847/847969.png"

def team_logo(abbr):
    return f"https://cdn.nba.com/logos/nba/{abbr}/primary/L/logo.svg"

# ---------- NEW HELPER: Tweets from X ----------
@st.cache_data(ttl=300)
def get_latest_nba_tweets(usernames: list, count: int = 3):
    """Fetch latest tweets from specified verified X users."""
    bearer_token = os.getenv("X_BEARER_TOKEN")
    if not bearer_token:
        st.warning("X API bearer token not provided; tweets will not load.")
        return []
    headers = {"Authorization": f"Bearer {bearer_token}"}
    tweets = []
    for user in usernames:
        # get user id
        resp = requests.get(f"https://api.twitter.com/2/users/by/username/{user}",
                            headers=headers, timeout=10)
        if resp.status_code != 200:
            continue
        user_id = resp.json().get("data", {}).get("id")
        if not user_id:
            continue
        t_resp = requests.get(
            f"https://api.twitter.com/2/users/{user_id}/tweets",
            params={"max_results": count, "tweet.fields": "created_at,author_id,text"},
            headers=headers, timeout=10
        )
        if t_resp.status_code != 200:
            continue
        for t in t_resp.json().get("data", []):
            tweets.append({
                "username": user,
                "created_at": t["created_at"],
                "text": t["text"]
            })
    tweets = sorted(tweets, key=lambda x: x["created_at"], reverse=True)[:count]
    return tweets

# ---------- NEW HELPER: Upcoming Games via ESPN (EST) ----------
@st.cache_data(ttl=600)
def get_games_from_espn(date: datetime.date):
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date.strftime('%Y%m%d')}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        games = []
        for event in data.get("events", []):
            comp = event["competitions"][0]
            competitors = comp["competitors"]
            home = next(c for c in competitors if c["homeAway"] == "home")
            away = next(c for c in competitors if c["homeAway"] == "away")

            utc_time = datetime.datetime.fromisoformat(comp["date"].replace("Z", "+00:00"))
            est_time = utc_time.astimezone(ZoneInfo("America/New_York"))
            time_str = est_time.strftime("%I:%M %p ET")

            broadcast = (comp.get("broadcasts")[0]["names"][0]
                         if comp.get("broadcasts") else "TBD")

            games.append({
                "home_team": home["team"]["displayName"],
                "home_logo": home["team"]["logo"],
                "away_team": away["team"]["displayName"],
                "away_logo": away["team"]["logo"],
                "time": time_str,
                "broadcast": broadcast
            })
        return games
    except Exception as e:
        st.error(f"Error fetching ESPN data: {e}")
        return []

# ---------- HEADER ----------
st.title("🏠 Hot Shot Props — NBA Home Hub")
st.caption("Live leaders, games, injuries & standings")

# ---------- LATEST NBA UPDATES FROM X ----------
st.markdown("## 🔔 Latest NBA Updates")
sources = ["NBA", "espnNBA", "wojespn"]
tweets = get_latest_nba_tweets(sources, count=3)
if not tweets:
    st.info("No recent tweets available from selected sources.")
else:
    for t in tweets:
        st.markdown(
            f"<div class='section'>"
            f"<b>@{t['username']}</b> — <i>{t['created_at']}</i><br>"
            f"{t['text']}"
            f"</div>",
            unsafe_allow_html=True
        )

# ---------- UPCOMING GAMES (ESPN Data) ----------
st.markdown("## 🗓️ Upcoming Games")
today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)
today_games = get_games_from_espn(today)
tomorrow_games = get_games_from_espn(tomorrow)

def render_games_section(title: str, games: list, date: datetime.date):
    st.markdown(f"### {title} ({date.strftime('%B %d, %Y')})")
    if not games:
        st.info("No scheduled games.")
        return
    for g in games:
        st.markdown(
            f"<div class='section'>"
            f"<img src='{g['away_logo']}' width='40'> "
            f"<b>{g['away_team']}</b> @ "
            f"<img src='{g['home_logo']}' width='40'> "
            f"<b>{g['home_team']}</b><br>"
            f"🕒 {g['time']} &nbsp;&nbsp; 📺 {g['broadcast']}"
            f"</div>",
            unsafe_allow_html=True
        )

render_games_section("🏀 Upcoming Games Tonight", today_games, today)
render_games_section("🌙 Upcoming Games Tomorrow", tomorrow_games, tomorrow)

# ---------- SEASON LEADERS ----------
st.markdown("## 🏀 Top Performers (Per Game Averages)")
df = get_leaders()
if not df.empty:
    df["PTS_Avg"] = (df["PTS"] / df["GP"]).round(1)
    df["REB_Avg"] = (df["REB"] / df["GP"]).round(1)
    df["AST_Avg"] = (df["AST"] / df["GP"]).round(1)
    df["FG3M_Avg"] = (df["FG3M"] / df["GP"]).round(1)
    df["BLK_Avg"] = (df["BLK"] / df["GP"]).round(1)
    df["STL_Avg"] = (df["STL"] / df["GP"]).round(1)
    df["TOV_Avg"] = (df["TOV"] / df["GP"]).round(1)

    categories = {
        "Points": "PTS_Avg",
        "Rebounds": "REB_Avg",
        "Assists": "AST_Avg",
        "3PT Field Goals Made": "FG3M_Avg",
        "Blocks": "BLK_Avg",
        "Steals": "STL_Avg",
        "Turnovers": "TOV_Avg"
    }

    for cat, key in categories.items():
        leader = df.loc[df[key].idxmax()]
        photo = player_photo(leader["PLAYER"])
        st.markdown(
            f"<div class='section leader'>"
            f"<img src='{photo}'>"
            f"<div><b>{leader['PLAYER']}</b><br>"
            f"{leader['TEAM']} — {cat}: <b>{leader[key]}</b></div></div>",
            unsafe_allow_html=True
        )
else:
    st.info("Leader data not available.")

# ---------- INJURY REPORT ----------
st.markdown("## 💀 Injury Report")
inj = get_injuries()
if not inj.empty:
    for _, r in inj.head(25).iterrows():
        scls = "status-active"
        if "Out" in r["status"]:
            scls = "status-out"
        elif "Questionable" in r["status"]:
            scls = "status-questionable"
        st.markdown(
            f"<div class='section'><b>{r['player']}</b> — {r['team']}<br>"
            f"<span class='{scls}'>{r['status']}</span> — {r.get('description','')}</div>",
            unsafe_allow_html=True
        )
else:
    st.warning("No injury data available.")

# ---------- STANDINGS ----------
st.markdown("## 🏆 NBA Standings")
stand = get_standings()
if not stand.empty:
    east = stand[stand["Conference"]=="East"]
    west = stand[stand["Conference"]=="West"]
    base = ["TeamCity","TeamName","WINS","LOSSES","WinPCT"]
    cols = [c for c in base if c in east.columns]
    if "Streak" in east.columns:
        cols.append("Streak")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Eastern Conference")
        st.dataframe(east[cols], width="stretch")
    with c2:
        st.markdown("### Western Conference")
        st.dataframe(west[cols], width="stretch")
else:
    st.warning("Standings unavailable.")

st.markdown("---")
st.caption("⚡ Hot Shot Props — Live Data © 2025")
