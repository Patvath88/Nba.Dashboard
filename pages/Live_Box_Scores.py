import streamlit as st
import requests
import datetime
import time

st.set_page_config(page_title="🏀 Live NBA Box Scores", layout="wide")
st.title("📊 Live NBA Box Scores")

# Auto-refresh every minute
REFRESH_INTERVAL = 60
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
if time.time() - st.session_state["last_refresh"] > REFRESH_INTERVAL:
    st.session_state["last_refresh"] = time.time()
    st.rerun()

st.caption(f"🔄 Auto-refresh every {REFRESH_INTERVAL} seconds | Last updated: {datetime.datetime.now().strftime('%I:%M:%S %p')}")

# --- ESPN API ---
def fetch_live_games():
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        games = [g for g in data.get("events", []) if g.get("status", {}).get("type", {}).get("state") == "in"]
        return games
    except Exception:
        return []

games = fetch_live_games()

# --- DISPLAY ---
if not games:
    st.info("🏀 No live NBA games right now.")
else:
    for game in games:
        comp = game["competitions"][0]
        competitors = comp["competitors"]
        home = next(c for c in competitors if c["homeAway"] == "home")
        away = next(c for c in competitors if c["homeAway"] == "away")

        home_team, away_team = home["team"], away["team"]
        home_score, away_score = home.get("score", "0"), away.get("score", "0")
        home_color, away_color = "#" + home_team.get("color", "0066FF"), "#" + away_team.get("color", "FF3B3B")

        status = comp.get("status", {}).get("type", {}).get("shortDetail", "Live")
        period = comp.get("status", {}).get("period", 0)
        clock = comp.get("status", {}).get("displayClock", "")

        st.markdown(f"""
        <div style='background: linear-gradient(90deg, {away_color}33, {home_color}33);
                    border-radius:15px; padding:15px; margin-bottom:20px;
                    box-shadow:0 0 15px rgba(255,255,255,0.1); text-align:center;'>
            <div style='display:flex; justify-content:space-around; align-items:center;'>
                <div>
                    <img src='{away_team["logo"]}' width='60'><br>
                    <b style='color:{away_color}'>{away_team["abbreviation"]}</b><br>
                    <span style='font-size:2rem;'>{away_score}</span>
                </div>
                <div style='font-size:1.1rem; color:#EAEAEA;'>
                    <b>{status}</b><br>
                    Q{period} — {clock}
                </div>
                <div>
                    <img src='{home_team["logo"]}' width='60'><br>
                    <b style='color:{home_color}'>{home_team["abbreviation"]}</b><br>
                    <span style='font-size:2rem;'>{home_score}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("⚡ Hot Shot Props — Live Box Scores via ESPN API © 2025")
