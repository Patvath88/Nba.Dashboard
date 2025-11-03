import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Projection Tracker", layout="wide")

st.markdown("""
<style>
body { background-color: black; color: white; }
.metric-card {
    background-color: #1e1e1e;
    border: 2px solid #E50914;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    color: #E50914;
    font-weight: bold;
    box-shadow: 0px 0px 10px #E50914;
}
.metric-card div.stat-name {
    font-size: 22px;
    margin-bottom: 5px;
}
.metric-card div.stat-value {
    font-size: 32px;
    margin-top: 5px;
}
.delete-btn {
    background-color: #b00020;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: bold;
    cursor: pointer;
}
.archive-btn {
    background-color: #00b300;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: bold;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

team_color = "#E50914"
contrast_color = "#00FFFF"

st.markdown("# 🎯 Projection Tracker")

# ---------------------- LOAD SAVED PROJECTIONS ----------------------
save_path = "saved_projections.csv"
archive_path = "archived_projections.csv"

try:
    data = pd.read_csv(save_path)
except FileNotFoundError:
    st.info("No projections saved yet.")
    st.stop()

nba_players = players.get_active_players()
player_map = {p["full_name"]: p["id"] for p in nba_players}

# ---------------------- HELPER FUNCTIONS ----------------------
def save_to_archive(player_name, row):
    """Move a projection row to archived_projections.csv"""
    df = pd.DataFrame([row])
    if os.path.exists(archive_path):
        old = pd.read_csv(archive_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(archive_path, index=False)
    st.success(f"✅ {player_name}'s projection saved to archives!")

def delete_projection(player_name):
    """Remove a player's projection from saved_projections.csv"""
    df = pd.read_csv(save_path)
    df = df[df["player"] != player_name]
    df.to_csv(save_path, index=False)
    st.warning(f"🗑️ {player_name}'s projection deleted.")

# ---------------------- DISPLAY TRACKED PROJECTIONS ----------------------
for player_name, group in data.groupby("player"):
    pid = player_map.get(player_name)
    if not pid:
        continue

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"## 🏀 {player_name}")
    with col2:
        if st.button("💾 Save Projection to Archives", key=f"archive_{player_name}"):
            latest_proj = group.iloc[-1].to_dict()
            save_to_archive(player_name, latest_proj)
            st.experimental_rerun()
    with col3:
        if st.button("❌ Delete Projection", key=f"delete_{player_name}"):
            delete_projection(player_name)
            st.experimental_rerun()

    latest_proj = group.iloc[-1].to_dict()

    # --- Get most recent game stats ---
    try:
        gl = playergamelog.PlayerGameLog(player_id=pid, season="2025-26").get_data_frames()[0]
        gl = gl.sort_values("GAME_DATE", ascending=False).iloc[0]
        real_stats = {
            "PTS": gl["PTS"],
            "REB": gl["REB"],
            "AST": gl["AST"],
            "FG3M": gl["FG3M"],
            "STL": gl["STL"],
            "BLK": gl["BLK"],
            "TOV": gl["TOV"],
            "PRA": gl["PTS"] + gl["REB"] + gl["AST"],
            "P+R": gl["PTS"] + gl["REB"],
            "P+A": gl["PTS"] + gl["AST"],
            "R+A": gl["REB"] + gl["AST"],
        }
    except Exception:
        st.info(f"Live data unavailable for {player_name}.")
        continue

    # --- Metric cards section ---
    st.markdown("### 📊 Projection vs. Actual Performance")
    stats = [s for s in latest_proj.keys() if s not in ["timestamp", "player"]]
    cols = st.columns(4)
    for i, stat in enumerate(stats):
        proj_val = latest_proj[stat]
        real_val = real_stats.get(stat, 0)
        hit = real_val >= proj_val
        border_color = "#00FF00" if hit else "#E50914"
        emoji = "✅" if hit else "❌"
        with cols[i % 4]:
            st.markdown(
                f"""
                <div class="metric-card" style="border-color:{border_color};color:{border_color};">
                    <div class="stat-name">{stat} {emoji}</div>
                    <div class="stat-value">{real_val}</div>
                    <div style="font-size:14px;margin-top:4px;">Proj: {proj_val}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- Dual bar graph comparison ---
    proj_values = [latest_proj[s] for s in stats]
    real_values = [real_stats.get(s, 0) for s in stats]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=stats, y=proj_values, name="AI Projection", marker_color=team_color))
    fig.add_trace(go.Bar(x=stats, y=real_values, name="Actual Result", marker_color=contrast_color))
    fig.update_layout(
        title=f"{player_name} — Projection vs. Actual Game Results",
        barmode="group",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=400,
        margin=dict(l=30, r=30, t=40, b=30),
        legend=dict(font=dict(color="white"))
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

st.caption("Reload this page anytime to check updated results for your saved player projections or manage archives.")
