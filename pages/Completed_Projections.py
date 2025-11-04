import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="🏆 Completed Game Projections", layout="wide")
st.title("🏀 Completed Game Projection Results")

# ---------------------- REFRESH ----------------------
REFRESH_INTERVAL = 300
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

@st.cache_data(ttl=120)
def get_gamelog(pid):
    try:
        df = playergamelog.PlayerGameLog(player_id=pid, season="2025-26").get_data_frames()[0]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df
    except Exception:
        return pd.DataFrame()

# ---------------------- DISPLAY COMPLETED ----------------------
today = pd.Timestamp.now().normalize()
completed = []

for _, row in data.iterrows():
    proj_date = pd.to_datetime(row.get("game_date"), errors="coerce")
    if not pd.isna(proj_date) and proj_date < today:
        completed.append(row)

if not completed:
    st.info("No completed game projections found yet.")
    st.stop()

df_completed = pd.DataFrame(completed)

for player_name, group in df_completed.groupby("player"):
    pid = player_map.get(player_name)
    if not pid:
        continue

    latest_proj = group.iloc[-1].to_dict()
    game_date = latest_proj.get("game_date", "")
    opponent = latest_proj.get("opponent", "")
    proj_date = pd.to_datetime(game_date, errors="coerce")

    # Fetch actual game log
    gl = get_gamelog(pid)
    act = gl[gl["GAME_DATE"] == proj_date]
    if act.empty:
        continue
    row_act = act.iloc[0]
    actual_stats = {
        "PTS": row_act["PTS"], "REB": row_act["REB"], "AST": row_act["AST"], "FG3M": row_act["FG3M"],
        "STL": row_act["STL"], "BLK": row_act["BLK"], "TOV": row_act["TOV"],
        "PRA": row_act["PTS"] + row_act["REB"] + row_act["AST"]
    }

    st.markdown("---")
    col_photo, col_info = st.columns([1, 3])
    with col_photo:
        photo = get_player_photo(pid)
        if photo:
            st.image(photo, width=180)
    with col_info:
        st.subheader(player_name)
        st.caption(f"📅 **Game Date:** {game_date or 'TBD'} | 🆚 **Opponent:** {opponent or 'TBD'}")

    # Display cards
    compare_stats = ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]
    cols = st.columns(4)
    for i, stat in enumerate(compare_stats):
        pred = latest_proj.get(stat, 0)
        act_val = actual_stats.get(stat, 0)
        diff = round(act_val - pred, 1)
        hit = act_val >= pred
        border = "#00FF66" if hit else "#E50914"
        glow = "#00FF6655" if hit else "#E5091444"
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style="border:1px solid {border};
                            border-radius:12px;
                            background:#111;
                            padding:10px;
                            text-align:center;
                            box-shadow:0 0 15px {glow};
                            margin-bottom:10px;">
                    <b>{stat}</b><br>
                    <span style='color:#00FFFF'>Proj: {pred}</span><br>
                    <span style='color:#E50914'>Actual: {act_val}</span><br>
                    <small>Δ {diff}</small>
                </div>
                """, unsafe_allow_html=True
            )

    # Bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=compare_stats, y=[latest_proj.get(s,0) for s in compare_stats],
                         name="AI Projection", marker_color="#E50914"))
    fig.add_trace(go.Bar(x=compare_stats, y=[actual_stats.get(s,0) for s in compare_stats],
                         name="Actual", marker_color="#00FFFF"))
    fig.update_layout(title=f"{player_name}: Projection vs Actual ({opponent})",
                      barmode="group",
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="white"),
                      height=350)
    st.plotly_chart(fig, use_container_width=True)
