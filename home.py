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
# 🏆 NBA STANDINGS — FIXED HTML RENDERING + LAST GAME INFO
# =========================================================
st.markdown("""
<h2 style="color:#FF6F00;text-shadow:0 0 10px #FF9F43;font-family:'Oswald',sans-serif;">
🏆 NBA Standings
</h2>
""", unsafe_allow_html=True)

stand = get_standings()

@st.cache_data(ttl=300)
def fetch_recent_games():
    """Fetch recent games with team stats included."""
    all_games = []
    for days_back in range(1, 4):
        date_str = (datetime.date.today() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=date_str)
            valid_df = next((df for df in sb.get_data_frames() if "TEAM_ID" in df.columns), None)
            if valid_df is not None:
                all_games.append(valid_df)
        except Exception:
            continue
    if not all_games:
        return pd.DataFrame()
    df = pd.concat(all_games)
    df = df.rename(columns={
        "TEAM_ID": "TeamID",
        "TEAM_ABBREVIATION": "TeamAbbr",
        "PTS": "Points",
        "PLUS_MINUS": "PlusMinus",
        "MATCHUP": "Matchup"
    })
    return df

recent_games = fetch_recent_games()

if not stand.empty:
    stand.columns = [c.replace(" ", "") for c in stand.columns]
    team_id_col = next((c for c in stand.columns if "TeamID" in c), None)
    team_abbr_col = next((c for c in stand.columns if c.lower() in ["teamtricode", "teamabbreviation", "teamslug"]), None)

    east = stand[stand["Conference"] == "East"].sort_values("PlayoffRank")
    west = stand[stand["Conference"] == "West"].sort_values("PlayoffRank")

    def get_last_game(team_id):
        """Return opponent, result, and W/L for a given team ID."""
        if recent_games.empty or "TeamID" not in recent_games.columns:
            return "—", "No recent data", ""
        tg = recent_games[recent_games["TeamID"] == team_id]
        if tg.empty:
            return "—", "No recent game", ""
        g = tg.iloc[-1]
        opp_team = str(g["Matchup"]).replace(g["TeamAbbr"], "").replace("vs.", "").replace("@", "").strip()
        try:
            result = f"{int(g['Points'])}–{int(g['Points'] - g['PlusMinus'])}"
        except Exception:
            result = "N/A"
        status = "Won" if g.get("PlusMinus", 0) > 0 else "Lost"
        return opp_team, f"{status} {result}", status

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
            team_id = row.get(team_id_col)
            team_abbr = str(row.get(team_abbr_col, "")).lower()
            logo_url = f"https://a.espncdn.com/i/teamlogos/nba/500/{team_abbr}.png"
            team_name = f"{row.get('TeamCity', '')} {row.get('TeamName', '')}"

            opp_team, result, status = get_last_game(team_id)
            winpct = round(row.get('WinPCT', 0) * 100, 1)
            streak = row.get('Streak', '—')
            status_color = "#00FF80" if "Won" in result else "#FF5252"

            html += f"""
            <div style='margin:8px 0;padding:10px 12px;
                        border-radius:10px;background:#111;
                        box-shadow:0 0 8px rgba(255,111,0,0.15);
                        transition:all 0.3s ease-in-out;'>
                <div style='display:flex;align-items:center;justify-content:space-between;'>
                    <div style='display:flex;align-items:center;gap:10px;'>
                        <img src="{logo_url}" width="36" height="36" style="border-radius:6px;">
                        <div style='color:#FFF;font-family:"Oswald",sans-serif;font-size:1.05rem;'>
                            {team_name}
                        </div>
                    </div>
                    <div style='color:#FFB266;font-family:"Roboto",sans-serif;'>
                        <b>{row.get('WINS', 0)}-{row.get('LOSSES', 0)}</b>
                        <span style='color:#66B3FF;font-size:0.85rem;margin-left:8px;'>{winpct}%</span>
                    </div>
                </div>
                <div style='margin-left:44px;margin-top:6px;color:{status_color};
                            font-family:"Roboto";font-size:0.9rem;'>
                    Last Game: {result} vs {opp_team}
                </div>
                <div style='margin-left:44px;color:#FF9F43;font-size:0.9rem;
                            font-family:"Oswald";margin-top:2px;'>
                    Streak: {streak}
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

else:
    st.warning("Standings currently unavailable.")
