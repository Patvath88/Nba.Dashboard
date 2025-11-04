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
st.set_page_config(page_title="🎯 Projection Tracker", layout="wide")
st.title("🏀 Upcoming Game Projection Tracker")

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

# ---------------------- LOAD PROJECTIONS ----------------------
path = "saved_projections.csv"
try:
    data = pd.read_csv(path)
except FileNotFoundError:
    st.info("No saved projections yet — save some on the Player AI page.")
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
        f"https://stats.nba.com/media/players/headshot/{pid}.png",
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

# ---------------------- MAIN DISPLAY ----------------------
for player_name, group in data.groupby("player"):
    pid = player_map.get(player_name)
    if not pid:
        continue

    latest_proj = group.iloc[-1].to_dict()
    game_date = latest_proj.get("game_date", "")
    opponent = latest_proj.get("opponent", "")
    proj_date = pd.to_datetime(game_date, errors="coerce")
    today = pd.Timestamp.now().normalize()

    # --- Player header ---
    st.markdown("---")
    col_photo, col_info = st.columns([1, 3])
    with col_photo:
        photo = get_player_photo(pid)
        if photo:
            st.image(photo, width=180)
    with col_info:
        st.subheader(player_name)
        st.caption(f"📅 **Game Date:** {game_date or 'TBD'} | 🆚 **Opponent:** {opponent or 'TBD'}")

    # --- Determine if game already happened ---
    game_finished = proj_date is not pd.NaT and proj_date < today

    # --- Fetch actual if available ---
    actual_stats = {}
    if game_finished:
        gl = get_gamelog(pid)
        act = gl[gl["GAME_DATE"] == proj_date]
        if not act.empty:
            row = act.iloc[0]
            actual_stats = {
                "PTS": row["PTS"], "REB": row["REB"], "AST": row["AST"], "FG3M": row["FG3M"],
                "STL": row["STL"], "BLK": row["BLK"], "TOV": row["TOV"],
                "PRA": row["PTS"] + row["REB"] + row["AST"],
            }

    # --- Build display dataframe ---
    compare_stats = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "PRA"]
    results = []
    for s in compare_stats:
        pred = latest_proj.get(s, 0)
        actual = actual_stats.get(s, None)
        if actual is None:
            diff, acc = "—", "Pending"
        else:
            diff = round(actual - pred, 1)
            acc = f"{max(0, 1 - abs(diff)/pred)*100:.1f}%" if pred else "—"
        results.append({"Stat": s, "Projected": pred, "Actual": actual if actual is not None else "—", "Δ": diff, "Accuracy": acc})
    df_show = pd.DataFrame(results)

    # --- Metric-like view ---
    cols = st.columns(4)
    for i, row in enumerate(df_show.itertuples()):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style="border:1px solid #00FFFF;
                            border-radius:12px;
                            background:#111;
                            padding:10px;
                            text-align:center;
                            box-shadow:0 0 10px #00FFFF55;">
                    <b>{row.Stat}</b><br>
                    <span style='color:#00FFFF'>Proj: {row.Projected}</span><br>
                    <span style='color:#E50914'>Actual: {row.Actual}</span><br>
                    <small>{'Δ '+str(row._4) if row._4!='—' else 'Pending'}</small>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- Bar chart only if game finished ---
    if game_finished and actual_stats:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=compare_stats, y=[latest_proj.get(s,0) for s in compare_stats],
                             name="AI Projection", marker_color="#E50914"))
        fig.add_trace(go.Bar(x=compare_stats, y=[actual_stats.get(s,0) for s in compare_stats],
                             name="Actual", marker_color="#00FFFF"))
        fig.update_layout(
            title=f"{player_name}: Projection vs Actual ({opponent})",
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=350,
            margin=dict(l=30,r=30,t=40,b=30)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{player_name}_{game_date}")
    else:
        st.info("🕒 Upcoming game — awaiting actual stats after tip-off.")
