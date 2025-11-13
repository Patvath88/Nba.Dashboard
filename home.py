# -------------------------------------------------
# HOT SHOT PROPS — NBA HOME HUB (Final Restored Edition)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import leagueleaders, leaguestandingsv3, scoreboardv2
from nba_api.stats.static import players
from urllib.parse import quote
import feedparser
import streamlit.components.v1 as components

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Hot Shot Props | NBA Home Hub",
                   page_icon="🏀", layout="wide")

# ---------- GLOBAL STYLE ----------
st.markdown("""
<style>
body {
    background-color: #000000 !important;
    color: #EAEAEA !important;
    font-family: 'Roboto', sans-serif;
}
h1, h2, h3 {
    color: #FF3B3B;
    text-shadow: 0 0 8px #0066FF;
    font-family: 'Oswald', sans-serif;
}
.section {
    background: #0A0A0A;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 20px;
    box-shadow: 0 0 20px rgba(0,102,255,0.2);
}
a {
    color: #FF3B3B;
}
a:hover {
    color: #66B3FF;
}
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

@st.cache_data(ttl=3600)
def player_id_map():
    return {p["full_name"]: p["id"] for p in players.get_active_players()}

def player_photo(name):
    pid = player_id_map().get(name)
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png" if pid else \
           "https://cdn-icons-png.flaticon.com/512/847/847969.png"

# ---------- HEADER ----------
st.title("🏠 Hot Shot Props — NBA Home Hub")
st.caption("Live news, games, leaders, injuries & standings")

# =========================================================
# 📰 LATEST NBA NEWS
# =========================================================
st.markdown("<h2>📰 Latest NBA News</h2>", unsafe_allow_html=True)

@st.cache_data(ttl=900)
def fetch_latest_nba_news(limit=3):
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
    st.info("No NBA headlines available right now.")
else:
    for article in news_items:
        st.markdown(
            f"""
            <div class='section'>
                <h3><a href="{article['link']}" target="_blank">{article['title']}</a></h3>
                <p>{article['summary']}</p>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# 🏟️ GAMES TONIGHT (Always Visible)
# =========================================================
st.markdown("<h2>🏟️ Games Tonight</h2>", unsafe_allow_html=True)

try:
    _, games, *_ = get_games_today()
    if not games.empty:
        for _, g in games.iterrows():
            st.markdown(
                f"""
                <div class='section'>
                    <b>{g['VISITOR_TEAM_NAME']}</b> @ <b>{g['HOME_TEAM_NAME']}</b><br>
                    <i>Tipoff:</i> {g['GAME_STATUS_TEXT']} (EST)
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No games scheduled tonight.")
except Exception:
    st.warning("Couldn't load today's schedule.")

# =========================================================
# 🏀 TOP PERFORMERS
# =========================================================
st.markdown("<h2>🏀 Top Performers (Per Game Averages)</h2>", unsafe_allow_html=True)
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
        "LAL": ("#552583", "#FDB927"), "GSW": ("#1D428A", "#FFC72C"),
        "BOS": ("#007A33", "#BA9653"), "DAL": ("#00538C", "#002B5E"),
        "MIA": ("#98002E", "#F9A01B"), "MIL": ("#00471B", "#EEE1C6"),
        "DEN": ("#0E2240", "#FEC524"), "NYK": ("#F58426", "#006BB6"),
        "PHI": ("#006BB6", "#ED174C"), "PHX": ("#1D1160", "#E56020")
    }

    html = """
    <style>
    .leader-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 25px;
        margin-top: 20px;
    }
    .leader-card {
        background: linear-gradient(180deg, #0B0B0B, #111);
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 0 30px rgba(255,0,0,0.2);
        transition: all 0.25s ease-in-out;
        width: 100%;
    }
    .leader-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 40px var(--team-primary);
    }
    .leader-photo img {
        width: 120px; height: 120px;
        border-radius: 50%;
        border: 3px solid var(--team-primary);
        box-shadow: 0 0 25px var(--team-secondary);
    }
    .leader-name {
        font-family: 'Oswald';
        font-size: 1.2rem;
        font-weight: bold;
        color: var(--team-primary);
        text-shadow: 0 0 6px var(--team-secondary);
    }
    .leader-team {
        font-size: 0.9rem;
        color: var(--team-secondary);
    }
    .leader-stat {
        font-size: 3rem;
        color: var(--team-primary);
        -webkit-text-stroke: 1px var(--team-secondary);
    }
    </style>
    <div class='leader-grid'>
    """

    for cat, key in categories.items():
        leader = df.loc[df[key].idxmax()]
        photo = player_photo(leader["PLAYER"])
        team_abbr = leader["TEAM"]
        primary, secondary = team_colors.get(team_abbr, ("#FF3B3B", "#0066FF"))
        html += f"""
        <div class='leader-card' style="--team-primary:{primary};--team-secondary:{secondary};">
            <div class='leader-name'>{leader["PLAYER"]}</div>
            <div class='leader-team'>{leader["TEAM"]}</div>
            <div class='leader-photo'><img src='{photo}'></div>
            <div class='leader-stat'>{leader[key]}</div>
            <div>{cat}</div>
        </div>
        """
    html += "</div>"
    components.html(html, height=800, scrolling=True)
else:
    st.warning("Leader data not available.")

# =========================================================
# 💀 INJURY REPORT
# =========================================================
st.markdown("<h2>💀 Injury Report</h2>", unsafe_allow_html=True)
@st.cache_data(ttl=900)
def fetch_injury_report():
    url = "https://www.espn.com/nba/injuries"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    data = []
    for sec in soup.find_all("section", class_="Card"):
        team_header = sec.find("h2")
        team_name = team_header.get_text(strip=True) if team_header else "Unknown"
        table = sec.find("table")
        if not table:
            continue
        for row in table.find_all("tr")[1:]:
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) >= 4:
                player, pos, injury, status = cols[:4]
                data.append({"team": team_name, "player": player, "position": pos, "injury": injury, "status": status})
    return pd.DataFrame(data)

inj_df = fetch_injury_report()
if not inj_df.empty:
    teams = sorted(inj_df["team"].unique())
    team = st.selectbox("Select a team to view injuries:", ["All Teams"] + teams)
    if team != "All Teams":
        inj_df = inj_df[inj_df["team"] == team]
    st.dataframe(inj_df, use_container_width=True)
else:
    st.warning("No injury data currently available from ESPN.")

# =========================================================
# 🏆 STANDINGS
# =========================================================
st.markdown("<h2>🏆 NBA Standings</h2>", unsafe_allow_html=True)
stand = get_standings()
if not stand.empty:
    east = stand[stand["Conference"] == "East"]
    west = stand[stand["Conference"] == "West"]
    cols = ["TeamCity", "TeamName", "WINS", "LOSSES", "WinPCT"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Eastern Conference")
        st.dataframe(east[cols], use_container_width=True)
    with c2:
        st.markdown("### Western Conference")
        st.dataframe(west[cols], use_container_width=True)
else:
    st.warning("Standings unavailable.")

st.markdown("---")
st.caption("⚡ Hot Shot Props — Live Data © 2025")
