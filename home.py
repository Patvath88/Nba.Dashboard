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
# 🩻 LATEST CONFIRMED INJURY NEWS (Filtered ESPN Feed)
# =========================================================
st.markdown("""
<h2 style="color:#FF3B3B;text-shadow:0 0 10px #FF3B3B;font-family:'Oswald',sans-serif;">
🩻 Latest Injury News
</h2>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def fetch_filtered_injury_news(limit=5):
    """Fetch only true injury-related news from ESPN feed."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news?category=injury"
    keywords = ["injury", "out", "ankle", "knee", "foot", "hand", "wrist", "elbow",
                "shoulder", "groin", "hamstring", "quad", "torn", "surgery",
                "illness", "back", "leg", "day-to-day", "questionable", "fracture"]
    news_items = []
    try:
        res = requests.get(url, timeout=10).json()
        articles = res.get("articles", [])
        for a in articles:
            headline = a.get("headline", "").strip()
            description = a.get("description", "").strip()
            full_text = f"{headline.lower()} {description.lower()}"
            if any(word in full_text for word in keywords):
                link = a.get("links", {}).get("web", {}).get("href", "")
                date = a.get("published", "")[:10]
                team = a.get("teams", [{}])[0].get("displayName", "Unknown Team")
                news_items.append({
                    "headline": headline,
                    "desc": description,
                    "link": link,
                    "date": date,
                    "team": team
                })
            if len(news_items) >= limit:
                break
    except Exception as e:
        st.error(f"Error fetching filtered injury news: {e}")
    return news_items

news_items = fetch_filtered_injury_news(limit=3)

if not news_items:
    st.warning("No recent confirmed NBA injury news found.")
else:
    st.markdown("""
    <style>
    .injury-news-card {
        background: linear-gradient(180deg, #0b0b0b, #121212);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 18px;
        border-left: 4px solid #FF3B3B;
        box-shadow: 0 0 15px rgba(255,59,59,0.25);
        transition: all 0.25s ease-in-out;
    }
    .injury-news-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 25px rgba(255,59,59,0.4);
    }
    .injury-title {
        font-family: 'Oswald', sans-serif;
        font-size: 1.2rem;
        color: #FF9F43;
        margin-bottom: 5px;
    }
    .injury-desc {
        font-family: 'Roboto', sans-serif;
        font-size: 0.95rem;
        color: #EAEAEA;
        line-height: 1.4em;
    }
    .injury-meta {
        font-size: 0.85rem;
        color: #66B3FF;
        margin-top: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

    for n in news_items:
        st.markdown(f"""
        <div class='injury-news-card'>
            <div class='injury-title'>
                <a href="{n['link']}" target="_blank" style="color:#FF9F43;text-decoration:none;">
                    {n['headline']}
                </a>
            </div>
            <div class='injury-desc'>{n['desc']}</div>
            <div class='injury-meta'>{n['team']} — {n['date']}</div>
        </div>
        """, unsafe_allow_html=True)



# =========================================================
# 🏆 NBA STANDINGS — Enhanced Leaderboard + Bracket (Fixed)
# =========================================================
st.markdown("""
<h2 style="color:#FF6F00;text-shadow:0 0 10px #FF9F43;font-family:'Oswald',sans-serif;">
🏆 NBA Standings
</h2>
""", unsafe_allow_html=True)

stand = get_standings()

if not stand.empty:
    # Normalize the column names for safety
    stand.columns = [c.replace(" ", "") for c in stand.columns]
    # Determine abbreviation column (try multiple options)
    team_col = next((c for c in stand.columns if c.lower() in ["teamtricode", "teamabbreviation", "teamslug"]), None)

    east = stand[stand["Conference"] == "East"].sort_values("PlayoffRank")
    west = stand[stand["Conference"] == "West"].sort_values("PlayoffRank")

    def render_leaderboard(df, conference_name):
        html = f"""
        <div style='background:linear-gradient(180deg,#0b0b0b,#121212);
                    border-radius:12px;padding:20px;margin-bottom:20px;
                    box-shadow:0 0 20px rgba(255,111,0,0.25);'>
            <h3 style='color:#FF9F43;text-shadow:0 0 8px #FF9F43;
                       font-family:"Oswald",sans-serif;text-align:center;'>
                {conference_name} Conference
            </h3>
            <div style='padding:8px 0;'>
        """
        for _, row in df.iterrows():
            team_abbr = str(row.get(team_col, "nba")).lower()
            team_name = f"{row.get('TeamCity', '')} {row.get('TeamName', '')}"
            logo_url = f"https://a.espncdn.com/i/teamlogos/nba/500/{team_abbr}.png"

            winpct = round(row.get('WinPCT', 0) * 100, 1)
            streak = row.get('Streak', '')
            html += f"""
            <div style='display:flex;align-items:center;justify-content:space-between;
                        margin:6px 0;padding:8px 12px;
                        background:linear-gradient(90deg,rgba(255,111,0,0.05) {winpct}%,rgba(255,111,0,0.02) {winpct}%);
                        border-radius:8px;box-shadow:0 0 8px rgba(255,111,0,0.1);'>
                <div style='display:flex;align-items:center;gap:10px;'>
                    <img src='{logo_url}' width='30' height='30' style='border-radius:6px;'>
                    <div style='color:#FFF;font-family:"Oswald",sans-serif;font-size:1.1rem;'>
                        {team_name}
                    </div>
                </div>
                <div style='color:#FFB266;font-family:"Roboto",sans-serif;'>
                    <b>{row.get('WINS', 0)}-{row.get('LOSSES', 0)}</b>
                    <span style='color:#66B3FF;font-size:0.85rem;margin-left:8px;'>{winpct}%</span>
                    <span style='color:#FF6F00;font-size:0.8rem;margin-left:8px;'>{streak}</span>
                </div>
            </div>
            """
        html += "</div></div>"
        return html

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(render_leaderboard(east, "Eastern"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_leaderboard(west, "Western"), unsafe_allow_html=True)

    # =========================================================
    # 🏀 IF THE NBA FINALS STARTED TODAY (Auto Bracket)
    # =========================================================
    st.markdown("""
    <h2 style="color:#FF3B3B;text-shadow:0 0 10px #FF3B3B;font-family:'Oswald',sans-serif;margin-top:30px;">
    🏀 If The NBA Finals Started Today
    </h2>
    """, unsafe_allow_html=True)

    def make_bracket_html(east_df, west_df):
        def matchup_block(seed1, seed8, side):
            color = "#FF3B3B" if side == "East" else "#66B3FF"
            t1 = f"{seed1.get('TeamName', seed1.get('TeamCity', ''))} ({int(seed1['PlayoffRank'])})"
            t8 = f"{seed8.get('TeamName', seed8.get('TeamCity', ''))} ({int(seed8['PlayoffRank'])})"
            return f"""
            <div style='padding:10px;border-left:3px solid {color};
                        margin-bottom:8px;background:#111;border-radius:8px;
                        box-shadow:0 0 8px {color}55;'>
                <b style='color:{color};font-family:"Oswald";font-size:1rem;'>
                    {t1} vs {t8}
                </b>
            </div>
            """

        bracket_html = "<div style='display:flex;justify-content:space-around;'>"

        # EAST
        bracket_html += "<div style='width:45%;'>"
        bracket_html += "<h4 style='color:#FF3B3B;text-shadow:0 0 6px #FF3B3B;'>Eastern Matchups</h4>"
        for i in range(4):
            bracket_html += matchup_block(east_df.iloc[i], east_df.iloc[-(i+1)], "East")
        bracket_html += "</div>"

        # WEST
        bracket_html += "<div style='width:45%;'>"
        bracket_html += "<h4 style='color:#66B3FF;text-shadow:0 0 6px #66B3FF;'>Western Matchups</h4>"
        for i in range(4):
            bracket_html += matchup_block(west_df.iloc[i], west_df.iloc[-(i+1)], "West")
        bracket_html += "</div></div>"
        return bracket_html

    st.markdown(make_bracket_html(east, west), unsafe_allow_html=True)

else:
    st.warning("Standings currently unavailable.")
