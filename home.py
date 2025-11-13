# -------------------------------------------------
# HOT SHOT PROPS — NBA HOME HUB (Full Restored + News Feed)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import datetime
import requests
from nba_api.stats.endpoints import leagueleaders, leaguestandingsv3, scoreboardv2
from nba_api.stats.static import players
from urllib.parse import quote
import feedparser

# ---------- PAGE CONFIG ----------
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

# ---------- HEADER ----------
st.title("🏠 Hot Shot Props — NBA Home Hub")
st.caption("Live leaders, news, injuries & standings")

# ---------- LATEST NBA NEWS (Clean Text-Only Headlines) ----------
import feedparser
from urllib.parse import quote

st.markdown("""
<h2 style="color:#FF6F00;text-shadow:0 0 8px #FF9F43;
           font-family:'Oswald',sans-serif;margin-top:30px;">
📰 Latest NBA News
</h2>
""", unsafe_allow_html=True)


@st.cache_data(ttl=900)
def fetch_latest_nba_news(limit=3):
    """Get 3 latest NBA news headlines and summaries."""
    feed_url = f"https://news.google.com/rss/search?q={quote('NBA basketball')}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)
    news_items = []
    for entry in feed.entries[:limit]:
        title = entry.title.strip()
        link = entry.link
        summary = getattr(entry, "summary", "")
        summary = summary.replace("<br>", " ").replace("\n", " ").strip()
        if len(summary) > 200:
            summary = summary[:200].rsplit(" ", 1)[0] + "..."
        news_items.append({"title": title, "link": link, "summary": summary})
    return news_items


news_items = fetch_latest_nba_news()

if not news_items:
    st.info("No NBA headlines available at the moment.")
else:
    st.markdown("""
    <style>
    .headline-card {
        background: #1C1C1C;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(255,111,0,0.1);
        transition: all 0.25s ease-in-out;
    }
    .headline-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 16px rgba(255,111,0,0.25);
    }
    .headline-title {
        font-family: 'Oswald', sans-serif;
        font-size: 1.25rem;
        color: #FF9F43;
        margin-bottom: 8px;
    }
    .headline-title a {
        color: #FF9F43;
        text-decoration: none;
    }
    .headline-title a:hover {
        color: #FFD480;
        text-decoration: underline;
    }
    .headline-summary {
        font-family: 'Roboto', sans-serif;
        font-size: 0.95rem;
        line-height: 1.5em;
        color: #EAEAEA;
    }
    </style>
    """, unsafe_allow_html=True)

    # Render each article as a clean clickable card
    for article in news_items:
        st.markdown(
            f"""
            <div class='headline-card'>
                <div class='headline-title'>
                    <a href="{article['link']}" target="_blank">{article['title']}</a>
                </div>
                <div class='headline-summary'>
                    {article['summary']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

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
