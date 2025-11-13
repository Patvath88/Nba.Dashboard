import streamlit as st
import pandas as pd
import time
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.static import players
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="🎯 Upcoming Game Projections", layout="wide")
st.title("🏀 Upcoming Game Projections")

# ---------------------- REFRESH ----------------------
REFRESH_INTERVAL = 60  # auto-refresh every 60 seconds
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
if time.time() - st.session_state["last_refresh"] > REFRESH_INTERVAL:
    st.session_state["last_refresh"] = time.time()
    st.rerun()

st.caption(f"🔄 Auto-refresh every {REFRESH_INTERVAL}s | Last updated {time.strftime('%H:%M:%S')}")
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


def get_player_team_abbr(player_name: str) -> str:
    """Get player's current team abbreviation from NBA API."""
    try:
        pid = player_map.get(player_name)
        if not pid:
            return ""
        info = commonplayerinfo.CommonPlayerInfo(player_id=pid).get_data_frames()[0]
        team_abbr = str(info.loc[0, "TEAM_ABBREVIATION"]).lower()
        return team_abbr
    except Exception:
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
    except Exception:
        return []


def get_next_game_for_team(team_abbr):
    """Find the next scheduled game for a given team abbreviation."""
    if not team_abbr:
        return None

    today = date.today()
    for d in range(0, 7):  # look up to a week ahead
        games = get_games_from_espn(today + timedelta(days=d))
        for g in games:
            if g["home_abbr"] == team_abbr.lower():
                return {
                    "date": g["date"],
                    "time": g["time"],
                    "home_away": "Home",
                    "opponent": g["away_team"]
                }
            elif g["away_abbr"] == team_abbr.lower():
                return {
                    "date": g["date"],
                    "time": g["time"],
                    "home_away": "Away",
                    "opponent": g["home_team"]
                }
    return None


def get_latest_player_stats(pid):
    """Fetch most recent game stats for player (PTS, REB, AST, etc.)."""
    try:
        logs = playergamelog.PlayerGameLog(player_id=pid, season='2025-26').get_data_frames()[0]
        if logs.empty:
            return None
        latest_game = logs.iloc[0]
        game_date = pd.to_datetime(latest_game["GAME_DATE"]).date()
        stats = {
            "date": game_date,
            "PTS": latest_game["PTS"],
            "REB": latest_game["REB"],
            "AST": latest_game["AST"],
            "FG3M": latest_game["FG3M"],
            "STL": latest_game["STL"],
            "BLK": latest_game["BLK"],
            "TOV": latest_game["TOV"],
            "PRA": latest_game["PTS"] + latest_game["REB"] + latest_game["AST"],
        }
        return stats
    except Exception:
        return None


def generate_stat_chart(stat_name, proj_value, history_list):
    """Generate a small inline chart comparing projection vs actual progression."""
    fig, ax = plt.subplots(figsize=(1.8, 0.8))
    ax.plot(range(len(history_list)), history_list, linewidth=1.8, color="#00FFFF", label="Actual", alpha=0.9)
    ax.axhline(proj_value, color="white", linestyle="--", linewidth=1, label="Projection")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#111")
    plt.tight_layout(pad=0)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, transparent=True)
    buf.seek(0)
    plt.close(fig)
    return buf

# ---------------------- SESSION STATE (persist player stat history) ----------------------
if "player_history" not in st.session_state:
    st.session_state["player_history"] = {}

# ---------------------- FILTER UPCOMING ----------------------
today = pd.Timestamp.now().normalize()
upcoming_games = [
    row for _, row in data.iterrows()
    if pd.isna(pd.to_datetime(row.get("game_date"), errors="coerce")) or
    pd.to_datetime(row.get("game_date"), errors="coerce") >= today
]

if not upcoming_games:
    st.info("No upcoming games with saved projections.")
    st.stop()

df_upcoming = pd.DataFrame(upcoming_games)

# ---------------------- DISPLAY ----------------------
for player_name, group in df_upcoming.groupby("player"):
    pid = player_map.get(player_name)
    if not pid:
        continue

    team_abbr = get_player_team_abbr(player_name)
    next_game = get_next_game_for_team(team_abbr)
    latest_proj = group.iloc[-1].to_dict()
    live_stats = get_latest_player_stats(pid)

    # Initialize player stat history
    if player_name not in st.session_state["player_history"]:
        st.session_state["player_history"][player_name] = {stat: [] for stat in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]}

    # Update history with new live data if available
    if live_stats and live_stats["date"] == date.today():
        for stat in st.session_state["player_history"][player_name]:
            current_val = live_stats.get(stat, 0)
            st.session_state["player_history"][player_name][stat].append(current_val)

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

    # Render projection cards
    compare_stats = ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]
    cols = st.columns(4)
    for i, stat in enumerate(compare_stats):
        proj = latest_proj.get(stat, 0)
        history = st.session_state["player_history"][player_name][stat]
        chart_buf = generate_stat_chart(stat, proj, history if history else [0])
        live_val = history[-1] if history else 0
        color = "#00FF88" if live_val >= proj and live_val > 0 else "#00FFFF"
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style="border:1px solid {color};
                            border-radius:12px;
                            background:#111;
                            padding:10px;
                            text-align:center;
                            box-shadow:0 0 15px {color}55;
                            margin-bottom:10px;">
                    <b>{stat}</b><br>
                    <span style='color:#00FFFF'>Proj: {proj}</span><br>
                    <small>{'✅ Met' if live_val >= proj and live_val > 0 else 'Tracking ⏳'}</small><br>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.image(chart_buf, use_container_width=True)

    if live_stats and live_stats["date"] == date.today():
        st.success("📡 Live tracking active — stats auto-refresh every 60 s (history preserved)")
    else:
        st.info("🕒 Upcoming game — awaiting actual stats after tip-off.")
