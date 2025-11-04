import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from PIL import Image
import requests
from io import BytesIO

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="🎯 Projection Tracker", layout="wide")
st.title("🏀 Live NBA Projection Tracker & Model Accuracy")

# ---------------------- REFRESH ----------------------
REFRESH_INTERVAL = 300  # seconds
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()

if time.time() - st.session_state["last_refresh"] > REFRESH_INTERVAL:
    st.session_state["last_refresh"] = time.time()
    st.rerun()

st.caption(f"🔄 Auto-refresh every 5 minutes | Last updated: {time.strftime('%H:%M:%S')}")
if st.button("🔁 Manual Refresh Now"):
    st.session_state["last_refresh"] = time.time()
    st.rerun()

# ---------------------- LOAD PROJECTIONS ----------------------
path = "saved_projections.csv"
try:
    data = pd.read_csv(path)
except FileNotFoundError:
    st.info("No saved projections yet. Use the AI Player page to create them.")
    st.stop()

if data.empty:
    st.info("No projection data available.")
    st.stop()

# ---------------------- HELPERS ----------------------
nba_players = players.get_active_players()
player_map = {p["full_name"]: p["id"] for p in nba_players}

def get_player_photo(player_id):
    """Try to fetch NBA media-day headshot."""
    urls = [
        f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png",
        f"https://stats.nba.com/media/players/headshot/{player_id}.png"
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                return Image.open(BytesIO(resp.content))
        except Exception:
            continue
    return None

@st.cache_data(ttl=120)
def get_gamelog(pid):
    """Fetch player's game log."""
    try:
        gl = playergamelog.PlayerGameLog(player_id=pid, season="2025-26").get_data_frames()[0]
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        return gl
    except Exception:
        return pd.DataFrame()

# ---------------------- MAIN DISPLAY ----------------------
for player_name, group in data.groupby("player"):
    pid = player_map.get(player_name)
    if not pid:
        continue

    latest_proj = group.iloc[-1].to_dict()
    game_date = latest_proj.get("game_date", "Unknown")
    opponent = latest_proj.get("opponent", "Unknown")

    st.markdown("---")
    col_photo, col_info = st.columns([1, 3])

    with col_photo:
        photo = get_player_photo(pid)
        if photo:
            st.image(photo, width=180)
        else:
            st.write("🖼️")

    with col_info:
        st.subheader(f"{player_name}")
        st.caption(f"📅 **Game Date:** {game_date} | 🆚 **Opponent:** {opponent}")

    gl = get_gamelog(pid)
    if gl.empty:
        st.warning("Live or recent game data unavailable.")
        continue

    # Find matching game by date
    actual_row = gl[gl["GAME_DATE"].astype(str).str.contains(str(game_date))]
    if actual_row.empty:
        st.info("Game stats not yet posted.")
        continue
    actual = actual_row.iloc[0].to_dict()

    # --------------- COMPARISON ----------------
    compare_stats = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "PRA"]
    results = []
    for stat in compare_stats:
        pred = latest_proj.get(stat, 0)
        act = actual.get(stat, 0)
        diff = act - pred
        acc = round((1 - abs(diff) / pred) * 100, 1) if pred else 0
        results.append({"Stat": stat, "Predicted": pred, "Actual": act, "Diff": diff, "Accuracy %": acc})

    df_compare = pd.DataFrame(results)

    st.dataframe(df_compare, hide_index=True, use_container_width=True)
    avg_acc = df_compare["Accuracy %"].mean().round(1)
    st.metric("🎯 Model Accuracy for This Game", f"{avg_acc}%")

    # --------------- BAR CHART: PROJ vs ACTUAL ----------------
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_compare["Stat"], y=df_compare["Predicted"],
        name="AI Projection", marker_color="#E50914"
    ))
    fig.add_trace(go.Bar(
        x=df_compare["Stat"], y=df_compare["Actual"],
        name="Actual Result", marker_color="#00FFFF"
    ))
    fig.update_layout(
        title=f"{player_name}: Projection vs Actual ({opponent})",
        barmode="group",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=350,
        margin=dict(l=30, r=30, t=40, b=30),
        legend=dict(font=dict(color="white"))
    )
    st.plotly_chart(fig, use_container_width=True, key=f"proj_actual_{player_name}_{game_date}")

# ---------------------- END ----------------------
st.markdown("---")
st.success("✅ Dashboard updated successfully — photos, matchups, and accuracy displayed.")
