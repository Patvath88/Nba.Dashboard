import streamlit as st
import pandas as pd
import time
from nba_api.stats.static import players, teams
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="🎯 Upcoming Game Projections", layout="wide")
st.title("🏀 Upcoming Game Projections")

# ---------------------- REFRESH ----------------------
REFRESH_INTERVAL = 300  # seconds
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
if time.time() - st.session_state["last_refresh"] > REFRESH_INTERVAL:
    st.session_state["last_refresh"] = time.time()
    st.rerun()

st.caption(f"🔄 Auto-refresh every 5 minutes | Last updated {time.strftime('%H:%M:%S')}")
if st.button("🔁 Manual Refresh Now"):
    st.session_state["last_refresh"] = time.time()
    st.rerun()

# ---------------------- LOAD DATA ----------------------
path = "saved_projections.csv"
try:
    data = pd.read_csv(path)
except FileNotFoundError:
    st.info("No saved projections yet.")
    st.stop()

if data.empty:
    st.info("No projection data available.")
    st.stop()

nba_players = players.get_active_players()
player_map = {p["full_name"]: p["id"] for p in nba_players}
nba_teams = teams.get_teams()

# ---------------------- HELPERS ----------------------
def get_player_photo(pid):
    urls = [
        f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png",
        f"https://stats.nba.com/media/players/headshot/{pid}.png"
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                return Image.open(BytesIO(r.content))
        except Exception:
            continue
    return None


# ESPN team abbreviations mapping (normalize all possibilities)
TEAM_MAP = {
    "atl": "atl", "bkn": "bkn", "bos": "bos", "cha": "cha", "chi": "chi", "cle": "cle",
    "dal": "dal", "den": "den", "det": "det", "gsw": "gsw", "hou": "hou", "ind": "ind",
    "lac": "lac", "lal": "lal", "mem": "mem", "mia": "mia", "mil": "mil", "min": "min",
    "nop": "nop", "nyk": "nyk", "okc": "okc", "orl": "orl", "phi": "phi", "phx": "phx",
    "por": "por", "sac": "sac", "sas": "sas", "tor": "tor", "uta": "uta", "wsh": "wsh",
    "was": "wsh", "gs": "gsw", "no": "nop"
}

def normalize_team_abbr(abbr_or_name: str) -> str:
    """Try to normalize any team name or abbreviation to ESPN's short form."""
    if not abbr_or_name:
        return ""
    abbr = str(abbr_or_name).lower().strip()
    # If exact match in map
    if abbr in TEAM_MAP:
        return TEAM_MAP[abbr]
    # Try matching full team names (like "Cavaliers" → "cle")
    for t in nba_teams:
        if abbr in t["full_name"].lower() or abbr in t["nickname"].lower():
            return TEAM_MAP.get(t["abbreviation"].lower(), "")
    return ""


@st.cache_data(ttl=600)
def get_games_from_espn(date_to_fetch: date):
    """Fetch NBA games for a given date (EST) from ESPN public API."""
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_to_fetch.strftime('%Y%m%d')}"
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

            utc_time = datetime.fromisoformat(comp["date"].replace("Z", "+00:00"))
            est_time = utc_time.astimezone(ZoneInfo("America/New_York"))

            games.append({
                "date": est_time.date(),
                "time": est_time.strftime("%I:%M %p ET"),
                "home_team": home["team"]["displayName"],
                "home_abbr": home["team"]["abbreviation"].lower(),
                "away_team": away["team"]["displayName"],
                "away_abbr": away["team"]["abbreviation"].lower(),
            })
        return games
    except Exception as e:
        st.error(f"Error fetching ESPN data: {e}")
        return []


def get_next_game_for_team(team_abbr):
    """Find the next scheduled game for the given team abbreviation."""
    norm_abbr = normalize_team_abbr(team_abbr)
    if not norm_abbr:
        return None

    today = date.today()
    for d in range(0, 7):  # look up to 7 days ahead
        games = get_games_from_espn(today + timedelta(days=d))
        for g in games:
            if g["home_abbr"] == norm_abbr:
                return {
                    "date": g["date"],
                    "time": g["time"],
                    "home_away": "Home",
                    "opponent": g["away_team"]
                }
            elif g["away_abbr"] == norm_abbr:
                return {
                    "date": g["date"],
                    "time": g["time"],
                    "home_away": "Away",
                    "opponent": g["home_team"]
                }
    return None

# ---------------------- FILTER UPCOMING ----------------------
today = pd.Timestamp.now().normalize()
upcoming_games = []

for _, row in data.iterrows():
    proj_date = pd.to_datetime(row.get("game_date"), errors="coerce")
    if pd.isna(proj_date) or proj_date >= today:
        upcoming_games.append(row)

if not upcoming_games:
    st.info("No upcoming games with saved projections.")
    st.stop()

df_upcoming = pd.DataFrame(upcoming_games)

# ---------------------- DISPLAY PLAYER PROJECTIONS ----------------------
for player_name, group in df_upcoming.groupby("player"):
    pid = player_map.get(player_name)
    if not pid:
        continue

    # Determine player's team from CSV or nba_api
    team_abbr = str(group.iloc[-1].get("team_abbr", ""))
    if not team_abbr:
        # fallback: find player's current team from nba_api
        pinfo = next((p for p in nba_players if p["full_name"] == player_name), None)
        if pinfo and "team_id" in pinfo:
            team_obj = next((t for t in nba_teams if t["id"] == pinfo["team_id"]), None)
            if team_obj:
                team_abbr = team_obj["abbreviation"]

    next_game = get_next_game_for_team(team_abbr)
    latest_proj = group.iloc[-1].to_dict()

    st.markdown("---")
    col_photo, col_info = st.columns([1, 3])
    with col_photo:
        photo = get_player_photo(pid)
        if photo:
            st.image(photo, width=180)

    with col_info:
        st.subheader(player_name)
        if next_game:
            st.caption(
                f"📅 **Game Date:** {next_game['date']} | 🕒 {next_game['time']} | "
                f"🏠 **{next_game['home_away']}** | 🆚 **{next_game['opponent']}**"
            )
        else:
            st.caption("📅 **Game Date:** TBD | 🆚 **Opponent:** TBD")

    compare_stats = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "PRA"]
    cols = st.columns(4)
    for i, stat in enumerate(compare_stats):
        val = latest_proj.get(stat, 0)
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style="border:1px solid #00FFFF;
                            border-radius:12px;
                            background:#111;
                            padding:10px;
                            text-align:center;
                            box-shadow:0 0 15px #00FFFF55;
                            margin-bottom:10px;">
                    <b>{stat}</b><br>
                    <span style='color:#00FFFF'>Proj: {val}</span><br>
                    <small>Pending ⏳</small>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.info("🕒 Upcoming game — awaiting actual stats after tip-off.")
