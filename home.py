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
import requests
from bs4 import BeautifulSoup

st.markdown("## 💀 Injury Report")
st.caption("Data sourced live from ESPN.com")

@st.cache_data(ttl=900)
def fetch_injury_report():
    """Scrape ESPN NBA injury report by team."""
    url = "https://www.espn.com/nba/injuries"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    tables = soup.find_all("table", class_="Table")
    if not tables:
        return pd.DataFrame()

    all_rows = []
    for table in tables:
        team_header = table.find_previous("h2")
        team_name = team_header.text.strip() if team_header else "Unknown Team"

        for row in table.find_all("tr")[1:]:
            cols = [c.text.strip() for c in row.find_all("td")]
            if len(cols) >= 4:
                player, pos, injury, status = cols[:4]
                all_rows.append({
                    "team": team_name,
                    "player": player,
                    "position": pos,
                    "injury": injury,
                    "status": status
                })

    return pd.DataFrame(all_rows)

inj_df = fetch_injury_report()

if inj_df.empty:
    st.warning("No current injury data available from ESPN.")
else:
    st.markdown("""
    <style>
    .inj-card {
        background: #1C1C1C;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 0 8px rgba(255,111,0,0.1);
        transition: all 0.25s ease-in-out;
    }
    .inj-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 14px rgba(255,111,0,0.25);
    }
    .inj-title {
        font-family: 'Oswald', sans-serif;
        color: #FF6F00;
        font-size: 1rem;
        margin-bottom: 4px;
    }
    .inj-text {
        color: #EAEAEA;
        font-family: 'Roboto', sans-serif;
        font-size: 0.9rem;
        line-height: 1.3em;
    }
    .inj-status {
        font-weight: bold;
    }
    .inj-out { color: #FF5252; }
    .inj-questionable { color: #FFD700; }
    .inj-active { color: #00FF80; }
    </style>
    """, unsafe_allow_html=True)

    # --- Team Filter ---
    teams = sorted(inj_df["team"].unique())
    selected_team = st.selectbox("Select a team to view injuries:", ["All Teams"] + teams)

    if selected_team != "All Teams":
        inj_df = inj_df[inj_df["team"] == selected_team]

    # --- Compact Scrollable Display ---
    container = st.container()
    with container:
        for _, row in inj_df.iterrows():
            status_class = "inj-active"
            if "Out" in row["status"]:
                status_class = "inj-out"
            elif "Questionable" in row["status"] or "Day-To-Day" in row["status"]:
                status_class = "inj-questionable"

            st.markdown(
                f"""
                <div class='inj-card'>
                    <div class='inj-title'>{row['player']} <span style='font-size:0.9em;'>({row['position']})</span></div>
                    <div class='inj-text'>
                        <b>Injury:</b> {row['injury']}<br>
                        <b>Status:</b> <span class='inj-status {status_class}'>{row['status']}</span><br>
                        <b>Team:</b> {row['team']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Add scrollbar for long lists
    st.markdown("""
    <style>
    .stContainer {
        overflow-y: auto !important;
        max-height: 700px;
    }
    </style>
    """, unsafe_allow_html=True)


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
