import streamlit as st
import requests
import time
import pandas as pd
from datetime import datetime, date
from nba_api.stats.endpoints import leagueleaders, leaguestandingsv3, scoreboardv2
from nba_api.stats.static import players, teams
from zoneinfo import ZoneInfo
import os
import feedparser  # For Google News fallback

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Hot Shot Props | NBA Home Hub", page_icon="🏀", layout="wide")

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
    today = datetime.now().strftime("%Y-%m-%d")
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
    pid=player_id_map().get(name)
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png" if pid else \
           "https://cdn-icons-png.flaticon.com/512/847/847969.png"

def team_logo(abbr):
    return f"https://cdn.nba.com/logos/nba/{abbr}/primary/L/logo.svg"


# ---------- HEADER ----------
st.title("🏠 Hot Shot Props — NBA Home Hub")
st.caption("Live leaders, games, injuries & standings")

# ---------- LATEST NBA NEWS (5 Image Cards, Clean Layout) ----------
import feedparser
from urllib.parse import quote

st.markdown("""
<h2 style="color:#FF6F00;text-shadow:0 0 8px #FF9F43;
           font-family:'Oswald',sans-serif;margin-top:30px;">
📰 Latest NBA News
</h2>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def fetch_latest_nba_news(limit=8):
    """Fetch top NBA news with working images."""
    feed_url = f"https://news.google.com/rss/search?q={quote('NBA basketball')}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)
    stories = []
    for entry in feed.entries:
        image = None
        # Prefer media_content image if available
        if "media_content" in entry and len(entry.media_content) > 0:
            image = entry.media_content[0].get("url")
        # Fallback: try embedded links
        elif hasattr(entry, "links"):
            for link in entry.links:
                if hasattr(link, "type") and "image" in link.type:
                    image = link.href
                    break
        # Only keep if there’s a valid image
        if image and image.startswith("http"):
            stories.append({
                "title": entry.title.strip(),
                "url": entry.link,
                "image": image
            })
        # Stop if enough valid image stories found
        if len(stories) >= limit:
            break
    return stories

news = fetch_latest_nba_news(limit=5)

if not news:
    st.info("🕵️ No NBA news found with images right now.")
else:
    st.markdown("""
    <style>
    .news-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 20px;
        margin-top: 10px;
    }
    .news-card {
        background: #1C1C1C;
        border-radius: 12px;
        width: calc(20% - 12px);
        min-width: 250px;
        box-shadow: 0 0 12px rgba(255,111,0,0.1);
        overflow: hidden;
        transition: all 0.25s ease-in-out;
    }
    .news-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 18px rgba(255,111,0,0.3);
    }
    .news-card img {
        width: 100%;
        height: 160px;
        object-fit: cover;
        display: block;
        border-bottom: 1px solid #333;
    }
    .news-card .title {
        color: #EAEAEA;
        font-family: 'Roboto', sans-serif;
        font-size: 0.9rem;
        padding: 10px 14px;
        line-height: 1.3em;
        height: 70px;
        overflow: hidden;
    }
    .news-card .title a {
        color: #FF9F43;
        text-decoration: none;
        font-weight: 500;
    }
    .news-card .title a:hover {
        text-decoration: underline;
    }
    @media (max-width: 1000px){
        .news-card { width: calc(50% - 12px); }
    }
    @media (max-width: 600px){
        .news-card { width: 100%; }
    }
    </style>
    """, unsafe_allow_html=True)

    html = "<div class='news-container'>"
    for item in news:
        html += f"""
        <div class='news-card'>
            <a href='{item['url']}' target='_blank'>
                <img src='{item['image']}' alt='NBA news image'/>
            </a>
            <div class='title'>
                <a href='{item['url']}' target='_blank'>{item['title']}</a>
            </div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)



# ---------- INJURY REPORT ----------
st.markdown("## 💀 Injury Report")
inj = get_injuries()
if not inj.empty:
    for _, r in inj.head(25).iterrows():
        scls = "status-active"
        if "Out" in r["status"]: scls = "status-out"
        elif "Questionable" in r["status"]: scls = "status-questionable"
        st.markdown(
            f"<div class='section'><b>{r['player']}</b> — {r['team']}<br>"
            f"<span class='{scls}'>{r['status']}</span> — {r.get('description','')}</div>",
            unsafe_allow_html=True)
else:
    st.warning("No injury data available.")

# ---------- STANDINGS ----------
st.markdown("## 🏆 NBA Standings")
stand = get_standings()
if not stand.empty:
    east = stand[stand["Conference"] == "East"]
    west = stand[stand["Conference"] == "West"]
    base = ["TeamCity", "TeamName", "WINS", "LOSSES", "WinPCT"]
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
