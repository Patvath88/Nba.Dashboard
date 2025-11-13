import streamlit as st
import requests
import datetime
import pandas as pd
import os
import time

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="🏀 Live NBA Box Scores", layout="wide")
st.title("📊 Live & Upcoming NBA Box Scores")

REFRESH_INTERVAL = 60
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
if time.time() - st.session_state["last_refresh"] > REFRESH_INTERVAL:
    st.session_state["last_refresh"] = time.time()
    st.rerun()

st.caption(f"🔄 Auto-refresh every {REFRESH_INTERVAL}s | Last updated: {datetime.datetime.now().strftime('%I:%M:%S %p')}")

# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------
def fetch_espn_games(days_ahead=0):
    """Pull games from ESPN API (same as home.py)."""
    base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    date = (datetime.datetime.now() + datetime.timedelta(days=days_ahead)).strftime("%Y%m%d")
    url = f"{base_url}?dates={date}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data.get("events", [])
    except Exception:
        return []

def fetch_boxscore(event_id):
    """Fetch detailed ESPN box score and play-by-play."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
    try:
        return requests.get(url, timeout=10).json()
    except Exception:
        return {}

def archive_completed_games(box_data):
    """Archive final box scores into CSV."""
    os.makedirs("data/archive", exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    path = f"data/archive/{date}.csv"
    df_new = pd.DataFrame(box_data)
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=["game_id"])
        df_combined.to_csv(path, index=False)
    else:
        df_new.to_csv(path, index=False)

# ------------------------------------------------------
# FETCH TODAY'S GAMES
# ------------------------------------------------------
games_today = fetch_espn_games(0)
games_tomorrow = fetch_espn_games(1)

if not games_today:
    st.info("🏀 No NBA games scheduled today.")
    st.stop()

# ------------------------------------------------------
# DISPLAY BOX SCORES
# ------------------------------------------------------
for game in games_today:
    comp = game["competitions"][0]
    competitors = comp.get("competitors", [])
    if len(competitors) < 2:
        continue

    home = next(c for c in competitors if c["homeAway"] == "home")
    away = next(c for c in competitors if c["homeAway"] == "away")

    home_team, away_team = home["team"], away["team"]
    home_color, away_color = "#" + home_team.get("color", "FF3B3B"), "#" + away_team.get("color", "0066FF")

    home_score, away_score = home.get("score", "0"), away.get("score", "0")
    status = comp.get("status", {}).get("type", {}).get("shortDetail", "Scheduled")
    game_id = game["id"]

    st.markdown(f"""
    <div style='background:linear-gradient(90deg,{away_color}33,{home_color}33);
                border-radius:15px;padding:15px;margin-bottom:20px;
                box-shadow:0 0 15px rgba(255,255,255,0.1);text-align:center;'>
        <div style='display:flex;justify-content:space-around;align-items:center;flex-wrap:wrap;'>
            <div>
                <img src='{away_team["logo"]}' width='60'><br>
                <b style='color:{away_color}'>{away_team["abbreviation"]}</b><br>
                <span style='font-size:2rem;'>{away_score}</span>
            </div>
            <div style='font-size:1.1rem;color:#EAEAEA;'>
                <b>{status}</b>
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
    box_data = fetch_boxscore(game_id)
    tab1, tab2 = st.tabs(["📊 Box Score", "🎬 Play-by-Play"])

    # --- BOX SCORE TAB ---
    with tab1:
        teams = box_data.get("boxscore", {}).get("teams", [])
        if not teams:
            st.info("No box score data yet — game may not have started.")
        else:
            for team in teams:
                st.subheader(team["team"]["displayName"])
                players = team.get("players", [])
                stats_rows = []
                for player in players:
                    athlete = player.get("athlete", {})
                    statline = player.get("stats", [])
                    stat_dict = {"Player": athlete.get("displayName", "N/A")}
                    for stat in statline:
                        stat_dict[stat["name"]] = stat.get("displayValue", "")
                    stats_rows.append(stat_dict)

                if stats_rows:
                    df = pd.DataFrame(stats_rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

    # --- PLAY-BY-PLAY TAB ---
    with tab2:
        plays = box_data.get("plays", [])
        if not plays:
            st.info("Play-by-play unavailable yet.")
        else:
            for play in plays[-20:][::-1]:  # last 20 plays
                clock = play.get("clock", "")
                desc = play.get("text", "")
                st.markdown(f"🕒 **{clock}** — {desc}")

    # Archive finished games
    status_state = comp.get("status", {}).get("type", {}).get("state", "")
    if status_state == "post":
        archive_completed_games([{
            "game_id": game_id,
            "home_team": home_team["displayName"],
            "away_team": away_team["displayName"],
            "home_score": home_score,
            "away_score": away_score,
            "status": "final",
            "timestamp": datetime.datetime.now().isoformat()
        }])

# ------------------------------------------------------
# TOMORROW'S GAMES PREVIEW
# ------------------------------------------------------
st.markdown("<h3 style='margin-top:40px;color:#FF3B3B;text-shadow:0 0 8px #0066FF;'>📅 Tomorrow’s Games</h3>", unsafe_allow_html=True)

if not games_tomorrow:
    st.info("No games scheduled for tomorrow yet.")
else:
    for game in games_tomorrow:
        comp = game["competitions"][0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next(c for c in competitors if c["homeAway"] == "home")
        away = next(c for c in competitors if c["homeAway"] == "away")

        home_team, away_team = home["team"], away["team"]
        home_color, away_color = "#" + home_team.get("color", "FF3B3B"), "#" + away_team.get("color", "0066FF")

        date = datetime.datetime.fromisoformat(game["date"].replace("Z", "+00:00"))
        time_est = date.astimezone(datetime.timezone(datetime.timedelta(hours=-5))).strftime("%I:%M %p EST")

        st.markdown(f"""
        <div style='background:linear-gradient(90deg,{away_color}33,{home_color}33);
                    border-radius:15px;padding:15px;margin-bottom:20px;
                    box-shadow:0 0 15px rgba(255,255,255,0.1);text-align:center;'>
            <div style='display:flex;justify-content:space-around;align-items:center;flex-wrap:wrap;'>
                <div>
                    <img src='{away_team["logo"]}' width='60'><br>
                    <b style='color:{away_color}'>{away_team["abbreviation"]}</b>
                </div>
                <div style='font-size:1.1rem;color:#EAEAEA;'>
                    <b>Tipoff:</b> {time_est}
                </div>
                <div>
                    <img src='{home_team["logo"]}' width='60'><br>
                    <b style='color:{home_color}'>{home_team["abbreviation"]}</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("⚡ Hot Shot Props — Live Scores & Play-by-Play © 2025")
