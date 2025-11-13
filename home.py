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

# ---------- SEASON LEADERS (3x2 Grid + Team Colors) ----------
st.markdown("""
<h2 style="color:#FF6F00;text-shadow:0 0 10px #FF9F43;
           font-family:'Oswald',sans-serif;text-align:center;">
🏀 Top Performers (Per Game Averages)
</h2>
""", unsafe_allow_html=True)

df = get_leaders()

if not df.empty:
    df["PTS_Avg"] = (df["PTS"] / df["GP"]).round(1)
    df["REB_Avg"] = (df["REB"] / df["GP"]).round(1)
    df["AST_Avg"] = (df["AST"] / df["GP"]).round(1)
    df["FG3M_Avg"] = (df["FG3M"] / df["GP"]).round(1)
    df["BLK_Avg"] = (df["BLK"] / df["GP"]).round(1)
    df["STL_Avg"] = (df["STL"] / df["GP"]).round(1)

    categories = {
        "Points": "PTS_Avg",
        "Rebounds": "REB_Avg",
        "Assists": "AST_Avg",
        "3PT Made": "FG3M_Avg",
        "Blocks": "BLK_Avg",
        "Steals": "STL_Avg"
    }

    team_colors = {
        "ATL": "#E03A3E", "BOS": "#007A33", "BKN": "#000000", "CHA": "#1D1160",
        "CHI": "#CE1141", "CLE": "#860038", "DAL": "#00538C", "DEN": "#0E2240",
        "DET": "#C8102E", "GSW": "#1D428A", "HOU": "#CE1141", "IND": "#FDBB30",
        "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9", "MIA": "#98002E",
        "MIL": "#00471B", "MIN": "#0C2340", "NOP": "#0C2340", "NYK": "#F58426",
        "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#1D1160",
        "POR": "#E03A3E", "SAC": "#5A2D81", "SAS": "#C4CED4", "TOR": "#CE1141",
        "UTA": "#002B5C", "WAS": "#002B5C"
    }

    st.markdown("""
    <style>
    .leader-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 25px;
        justify-items: center;
        margin: 30px auto;
        max-width: 1000px;
    }
    .leader-card {
        position: relative;
        background: radial-gradient(circle at top, #1a1a1a 0%, #0b0b0b 100%);
        border-radius: 18px;
        padding: 18px 10px;
        text-align: center;
        box-shadow: 0 0 15px rgba(255,111,0,0.25);
        transition: all 0.25s ease-in-out;
        overflow: hidden;
        width: 100%;
        max-width: 250px;
    }
    .leader-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 25px var(--team-color);
    }
    .leader-name {
        font-family: 'Oswald', sans-serif;
        font-size: 1.2rem;
        color: #FFFFFF;
        margin-bottom: 2px;
    }
    .leader-team {
        color: #FFB266;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .leader-photo {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        overflow: hidden;
        margin: 0 auto 10px;
        border: 3px solid var(--team-color);
        box-shadow: 0 0 15px var(--team-color);
        transition: transform 0.2s ease;
    }
    .leader-photo:hover {
        transform: scale(1.05);
    }
    .leader-photo img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 50%;
    }
    .leader-cat {
        font-family: 'Oswald', sans-serif;
        color: #FF9F43;
        font-size: 1rem;
        margin-top: 8px;
    }
    .leader-stat {
        font-family: 'Oswald', sans-serif;
        font-size: 2rem;
        color: var(--team-color);
        font-weight: bold;
        text-shadow: 0 0 10px var(--team-color);
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create grid layout correctly
    cards = []
    for cat, key in categories.items():
        leader = df.loc[df[key].idxmax()]
        photo = player_photo(leader["PLAYER"])
        team_abbr = leader["TEAM"]
        team_color = team_colors.get(team_abbr, "#FF6F00")

        card_html = f"""
        <div class='leader-card' style="--team-color: {team_color};">
            <div class='leader-name'>{leader["PLAYER"]}</div>
            <div class='leader-team'>{leader["TEAM"]}</div>
            <div class='leader-photo'>
                <img src='{photo}' alt='{leader["PLAYER"]}'>
            </div>
            <div class='leader-cat'>{cat}</div>
            <div class='leader-stat'>{leader[key]}</div>
        </div>
        """
        cards.append(card_html)

    grid_html = "<div class='leader-grid'>" + "".join(cards) + "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)

else:
    st.info("Leader data not available.")


# ---------- INJURY REPORT ----------
import requests
from bs4 import BeautifulSoup

st.markdown("## 💀 Injury Report")
st.caption("Live injury data — pulled directly from ESPN.com")

@st.cache_data(ttl=900)
def fetch_injury_report():
    """Scrape ESPN NBA injury report (team + player)."""
    url = "https://www.espn.com/nba/injuries"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    data = []
    # Each team section is wrapped in a div with class "Wrapper"
    sections = soup.find_all("section", class_="Card")
    for sec in sections:
        team_header = sec.find("h2")
        team_name = team_header.get_text(strip=True) if team_header else "Unknown Team"

        table = sec.find("table")
        if not table:
            continue
        rows = table.find_all("tr")[1:]
        for r_ in rows:
            cols = [c.get_text(strip=True) for c in r_.find_all("td")]
            if len(cols) >= 4:
                player, pos, injury, status = cols[:4]
                data.append({
                    "team": team_name,
                    "player": player,
                    "position": pos,
                    "injury": injury,
                    "status": status
                })
    return pd.DataFrame(data)


inj_df = fetch_injury_report()

if inj_df.empty:
    st.warning("No injury data currently available from ESPN.")
else:
    # --- Dropdown filter ---
    teams = sorted(inj_df["team"].unique())
    selected_team = st.selectbox("Select a team to view injuries:", ["All Teams"] + teams)

    if selected_team != "All Teams":
        inj_df = inj_df[inj_df["team"] == selected_team]

    # --- Compact styled table ---
    st.markdown("""
        <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th {
            background-color: #FF6F00;
            color: white;
            font-family: 'Oswald', sans-serif;
            padding: 10px;
            text-align: left;
            font-size: 1rem;
        }
        td {
            background-color: #1C1C1C;
            color: #EAEAEA;
            padding: 8px 10px;
            border-bottom: 1px solid #333;
            font-size: 0.9rem;
            font-family: 'Roboto', sans-serif;
        }
        tr:hover td {
            background-color: #2A2A2A;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display team name as section header
    if selected_team != "All Teams":
        st.markdown(f"### 🏀 {selected_team}")

    # Convert to HTML table (for styling control)
    html_table = inj_df.to_html(index=False, escape=False)
    st.markdown(html_table, unsafe_allow_html=True)



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
