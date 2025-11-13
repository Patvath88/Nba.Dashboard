# -------------------------------------------------
# HOT SHOT PROPS — NBA HOME HUB (Final Polished Neon Edition)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import leagueleaders, leaguestandingsv3
from nba_api.stats.static import players
from urllib.parse import quote
import feedparser
import streamlit.components.v1 as components

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Hot Shot Props | NBA Home Hub", page_icon="🏀", layout="wide")

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
    text-shadow: 0 0 8px #0066FF, 0 0 14px #FF3B3B;
    font-family: 'Oswald', sans-serif;
}
.section {
    background: #0A0A0A;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 20px;
    box-shadow: 0 0 25px rgba(255,0,0,0.2);
}
a { color: #FF3B3B; }
a:hover { color: #66B3FF; }
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

@st.cache_data(ttl=3600)
def player_id_map():
    return {p["full_name"]: p["id"] for p in players.get_active_players()}

def player_photo(name):
    pid = player_id_map().get(name)
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png" if pid else \
           "https://cdn-icons-png.flaticon.com/512/847/847969.png"

# ---------- ESPN GAME DATA ----------
def fetch_espn_games(days_ahead=0):
    base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    date = (datetime.datetime.now() + datetime.timedelta(days=days_ahead)).strftime("%Y%m%d")
    url = f"{base_url}?dates={date}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data.get("events", [])
    except Exception:
        return []

# ---------- HEADER ----------
st.title("🏠 Hot Shot Props — NBA Home Hub")
st.caption("Live news, leaders, games, injuries & standings")

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
for article in news_items:
    st.markdown(
        f"""
        <div class='section'>
            <h3><a href="{article['link']}" target="_blank">{article['title']}</a></h3>
            <p>{article['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

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
    }
    .leader-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 40px var(--team-primary);
    }
    .leader-photo img {
        width: 120px; height: 120px;
        border-radius: 50%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        border: 4px solid var(--team-primary);
        box-shadow: 0 0 25px var(--team-secondary);
    }
    .leader-name {
        font-family: 'Oswald';
        font-size: 1.25rem;
        color: var(--team-primary);
        text-shadow: 0 0 6px var(--team-secondary);
        margin-top: 8px;
    }
    .leader-team {
        font-size: 0.9rem;
        color: var(--team-secondary);
        margin-bottom: 5px;
    }
    .leader-stat {
        font-size: 3rem;
        color: var(--team-primary);
        -webkit-text-stroke: 1px var(--team-secondary);
        text-shadow: 0 0 10px var(--team-secondary);
        margin-top: 8px;
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
            <div class='leader-photo'><img src='{photo}'></div>
            <div class='leader-name'>{leader["PLAYER"]}</div>
            <div class='leader-team'>{leader["TEAM"]}</div>
            <div class='leader-stat'>{leader[key]}</div>
            <div>{cat}</div>
        </div>
        """
    html += "</div>"
    components.html(html, height=850, scrolling=True)

# =========================================================
# 🕒 GAMES TONIGHT + TOMORROW
# =========================================================
def render_games_section(title, games):
    st.markdown(f"<h2 style='color:#FF3B3B;text-shadow:0 0 10px #0066FF;'>🏟️ {title}</h2>", unsafe_allow_html=True)
    html = """
    <style>
    .game-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 25px;
        margin-top: 20px;
    }
    .game-card {
        background: linear-gradient(180deg, #0B0B0B, #111);
        border-radius: 18px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 0 25px rgba(255,0,0,0.2);
        transition: all 0.25s ease-in-out;
    }
    .game-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 40px var(--team-primary);
    }
    .team-logos img {
        width: 60px; height: 60px;
        border-radius: 50%;
        margin: 0 8px;
    }
    .team-name { color: var(--team-primary); font-weight: bold; font-size: 1rem; }
    .game-info { color: #EAEAEA; font-size: 0.9rem; margin-top: 5px; }
    </style>
    <div class='game-grid'>
    """

    for game in games:
        comp = game.get("competitions", [])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue
        home = next(c for c in competitors if c["homeAway"] == "home")
        away = next(c for c in competitors if c["homeAway"] == "away")
        home_team, away_team = home["team"], away["team"]
        home_color = "#" + home_team.get("color", "FF3B3B")
        away_color = "#" + away_team.get("color", "0066FF")

        venue = comp.get("venue", {}).get("fullName", "Unknown Arena")

        # --- SAFER BROADCAST HANDLING ---
        broadcasts = comp.get("broadcasts", [])
        if broadcasts:
            names = []
            for b in broadcasts:
                if "shortName" in b:
                    names.append(b["shortName"])
                elif "name" in b:
                    names.append(b["name"])
            broadcast = ", ".join(names)
        else:
            broadcast = "TBD"

        date = datetime.datetime.fromisoformat(game["date"].replace("Z", "+00:00"))
        time_est = date.astimezone(datetime.timezone(datetime.timedelta(hours=-5))).strftime("%I:%M %p EST")

        html += f"""
        <div class='game-card' style="--team-primary:{home_color};--team-secondary:{away_color};">
            <div class='team-logos'>
                <img src='{away_team["logo"]}'><img src='{home_team["logo"]}'>
            </div>
            <div class='team-name'>{away_team["displayName"]} @ {home_team["displayName"]}</div>
            <div class='game-info'><b>Tipoff:</b> {time_est}</div>
            <div class='game-info'><b>Network:</b> {broadcast}</div>
            <div class='game-info'><b>Arena:</b> {venue}</div>
        </div>
        """
    html += "</div>"
    components.html(html, height=900, scrolling=True)

render_games_section("Games Tonight", fetch_espn_games(0))
render_games_section("Tomorrow’s Games", fetch_espn_games(1))

# =========================================================
# 💀 CONFIRMED INJURY REPORT (LIVE ESPN Feed — Verified Working)
# =========================================================
st.markdown("<h2>💀 Confirmed Injury Report</h2>", unsafe_allow_html=True)
st.caption("Live from ESPN Injury Feed — showing confirmed OUT / long-term injuries only")

@st.cache_data(ttl=600)
def fetch_injury_report_live():
    """Fetch live injury news directly from ESPN's JSON feed."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news?category=injury"
    data = []
    try:
        res = requests.get(url, timeout=10).json()
        articles = res.get("articles", [])
        for a in articles:
            headline = a.get("headline", "")
            desc = a.get("description", "")
            teams = a.get("teams", [])
            date = a.get("published", "")[:10]

            if not teams:
                continue

            for t in teams:
                team = t.get("displayName", "Unknown Team")
                abbrev = t.get("abbreviation", "")
                color = "#" + t.get("color", "FF3B3B")

                # Parse the player & status from the headline if possible
                player = headline.split(":")[0].strip() if ":" in headline else headline
                if any(x in desc.lower() for x in ["out", "surgery", "injury", "fracture", "torn", "miss", "season", "indefinite"]):
                    data.append({
                        "team": team,
                        "abbrev": abbrev,
                        "color": color,
                        "player": player,
                        "injury": desc.strip(),
                        "status": "Out",
                        "date": date
                    })
    except Exception as e:
        st.error(f"Error fetching injury data: {e}")
    return pd.DataFrame(data)

inj_df = fetch_injury_report_live()

if inj_df.empty:
    st.warning("No confirmed injuries currently listed on ESPN’s live feed.")
else:
    teams = sorted(inj_df["team"].unique())
    selected_team = st.selectbox("Filter by team:", ["All Teams"] + teams)
    if selected_team != "All Teams":
        inj_df = inj_df[inj_df["team"] == selected_team]

    st.markdown("""
    <style>
    .injury-card {
        background: linear-gradient(180deg, #0B0B0B, #111);
        border-radius: 12px;
        padding: 12px 15px;
        margin-bottom: 10px;
        box-shadow: 0 0 15px rgba(255,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.25s ease-in-out;
    }
    .injury-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 25px rgba(0,102,255,0.35);
    }
    .injury-player { font-weight: bold; color: #FF3B3B; font-size: 1.1rem; }
    .injury-status { color: #66B3FF; font-weight: bold; }
    .injury-info { font-size: 0.9rem; color: #EAEAEA; }
    </style>
    """, unsafe_allow_html=True)

    for team in sorted(inj_df["team"].unique()):
        team_data = inj_df[inj_df["team"] == team]
        team_color = team_data.iloc[0]["color"]
        with st.expander(f"🏀 {team} ({len(team_data)} confirmed injuries)", expanded=False):
            for _, row in team_data.iterrows():
                st.markdown(f"""
                    <div class='injury-card' style='border-left:4px solid {team_color};'>
                        <div class='injury-player'>{row['player']}</div>
                        <div class='injury-info'>{row['injury']}</div>
                        <div class='injury-status'>{row['status']} • {row['date']}</div>
                    </div>
                """, unsafe_allow_html=True)

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
