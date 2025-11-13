import streamlit as st
import requests
import datetime
import pandas as pd
import os
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🏀 Live NBA Box Scores", layout="wide")
st.title("📊 Live & Upcoming NBA Box Scores")

REFRESH_INTERVAL = 60
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
if time.time() - st.session_state["last_refresh"] > REFRESH_INTERVAL:
    st.session_state["last_refresh"] = time.time()
    st.rerun()

st.caption(f"🔄 Auto-refresh every {REFRESH_INTERVAL}s | Last updated: {datetime.datetime.now().strftime('%I:%M:%S %p')}")

# ---------------- HELPERS ----------------
def fetch_espn_data():
    """Fetch full ESPN NBA scoreboard."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    try:
        return requests.get(url, timeout=10).json()
    except Exception:
        return {}

def fetch_boxscore(event_id):
    """Fetch ESPN box score for a specific game."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
    try:
        return requests.get(url, timeout=10).json()
    except Exception:
        return {}

def archive_completed_games(box_data):
    """Save completed game box scores to archive."""
    os.makedirs("data/archive", exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    path = f"data/archive/{date}.csv"

    if not os.path.exists(path):
        pd.DataFrame(box_data).to_csv(path, index=False)
    else:
        existing = pd.read_csv(path)
        updated = pd.concat([existing, pd.DataFrame(box_data)], ignore_index=True)
        updated.to_csv(path, index=False)

# ---------------- MAIN ----------------
data = fetch_espn_data()
events = data.get("events", [])
today = datetime.datetime.now().strftime("%Y-%m-%d")

if not events:
    st.info("🏀 No NBA games scheduled or data unavailable.")
    st.stop()

live_or_today_games = []
archived_games = []

for game in events:
    comp = game["competitions"][0]
    status = comp["status"]["type"]["state"]
    date_utc = datetime.datetime.fromisoformat(game["date"].replace("Z", "+00:00"))
    game_date = date_utc.date().isoformat()
    if game_date == today:
        if status == "post":
            archived_games.append(game)
        else:
            live_or_today_games.append(game)

# Archive finished games
if archived_games:
    archive_completed_games([
        {
            "game_id": g["id"],
            "home_team": g["competitions"][0]["competitors"][0]["team"]["displayName"],
            "away_team": g["competitions"][0]["competitors"][1]["team"]["displayName"],
            "status": "final",
        } for g in archived_games
    ])

# ---------------- DISPLAY GAMES ----------------
if not live_or_today_games:
    st.info("📅 No live or upcoming NBA games today.")
else:
    for game in live_or_today_games:
        comp = game["competitions"][0]
        competitors = comp["competitors"]
        home = next(c for c in competitors if c["homeAway"] == "home")
        away = next(c for c in competitors if c["homeAway"] == "away")

        home_team = home["team"]
        away_team = away["team"]
        home_color = "#" + home_team.get("color", "FF3B3B")
        away_color = "#" + away_team.get("color", "0066FF")

        home_score = home.get("score", "0")
        away_score = away.get("score", "0")

        status_detail = comp["status"]["type"].get("shortDetail", "")
        game_id = game["id"]

        st.markdown(f"""
        <div style='background:linear-gradient(90deg, {away_color}33, {home_color}33);
                    border-radius:15px; padding:15px; margin-bottom:20px;
                    box-shadow:0 0 15px rgba(255,255,255,0.1); text-align:center;'>
            <div style='display:flex; justify-content:space-around; align-items:center; flex-wrap:wrap;'>
                <div>
                    <img src='{away_team["logo"]}' width='60'><br>
                    <b style='color:{away_color}'>{away_team["abbreviation"]}</b><br>
                    <span style='font-size:2rem;'>{away_score}</span>
                </div>
                <div style='font-size:1.1rem; color:#EAEAEA;'>
                    <b>{status_detail}</b>
                </div>
                <div>
                    <img src='{home_team["logo"]}' width='60'><br>
                    <b style='color:{home_color}'>{home_team["abbreviation"]}</b><br>
                    <span style='font-size:2rem;'>{home_score}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tabs for Box Score + Play-by-Play
        boxscore_data = fetch_boxscore(game_id)
        tab1, tab2 = st.tabs(["📊 Box Score", "🎬 Play-by-Play"])

        with tab1:
            teams = boxscore_data.get("boxscore", {}).get("teams", [])
            if not teams:
                st.info("No box score data available yet.")
            else:
                for team in teams:
                    st.subheader(team["team"]["displayName"])
                    stats = team.get("statistics", [])
                    df = pd.DataFrame([
                        {
                            "Player": a["athlete"]["displayName"],
                            "MIN": next((s["displayValue"] for s in a["stats"] if s["name"] == "MIN"), "—"),
                            "PTS": next((s["displayValue"] for s in a["stats"] if s["name"] == "PTS"), "—"),
                            "REB": next((s["displayValue"] for s in a["stats"] if s["name"] == "REB"), "—"),
                            "AST": next((s["displayValue"] for s in a["stats"] if s["name"] == "AST"), "—"),
                            "FG": next((s["displayValue"] for s in a["stats"] if s["name"] == "FG"), "—"),
                            "3PT": next((s["displayValue"] for s in a["stats"] if s["name"] == "3PT"), "—"),
                            "FT": next((s["displayValue"] for s in a["stats"] if s["name"] == "FT"), "—")
                        }
                        for a in team.get("players", [])
                    ])
                    st.dataframe(df, hide_index=True, use_container_width=True)

        with tab2:
            pbp = boxscore_data.get("plays", [])
            if not pbp:
                st.info("No play-by-play data yet.")
            else:
                for play in pbp[-15:][::-1]:  # last 15 plays reversed
                    clock = play.get("clock", "")
                    desc = play.get("text", "")
                    st.markdown(f"🕒 **{clock}** — {desc}")

st.markdown("---")
st.caption("⚡ Hot Shot Props — Live Box Scores & Play-by-Play © 2025")
