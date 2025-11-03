import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Saved Projections", layout="wide")

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
</style>
""", unsafe_allow_html=True)

team_color = "#E50914"
contrast_color = "#00FFFF"

st.markdown("# 🏆 Saved Projections")

# ---------------------- LOAD ARCHIVED DATA ----------------------
path = "archived_projections.csv"
try:
    data = pd.read_csv(path)
except FileNotFoundError:
    st.info("No archived projections yet. Save some from your Projection Tracker to get started!")
    st.stop()

nba_players = players.get_active_players()
player_map = {p["full_name"]: p["id"] for p in nba_players}

# ---------------------- DISPLAY ARCHIVED PROJECTIONS ----------------------
for player_name, group in data.groupby("player"):
    pid = player_map.get(player_name)
    if not pid:
        continue

    st.markdown(f"## 🏀 {player_name}")

    for _, row in group.iterrows():
        proj_date = row.get("timestamp", "Unknown Date")
        latest_proj = row.to_dict()

        # --- Fetch actual game data for that player on or after projection date ---
        try:
            gl = playergamelog.PlayerGameLog(player_id=pid, season="2025-26").get_data_frames()[0]
            gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
            proj_dt = pd.to_datetime(proj_date)
            gl = gl.sort_values("GAME_DATE", ascending=False)
            game = gl[gl["GAME_DATE"] >= proj_dt].iloc[0] if not gl.empty else None
        except Exception:
            game = None

        if game is None:
            st.info(f"No matching game data found near {proj_date} for {player_name}.")
            continue

        # --- Real game stats for comparison ---
        real_stats = {
            "PTS": game["PTS"],
            "REB": game["REB"],
            "AST": game["AST"],
            "FG3M": game["FG3M"],
            "STL": game["STL"],
            "BLK": game["BLK"],
            "TOV": game["TOV"],
            "PRA": game["PTS"] + game["REB"] + game["AST"],
            "P+R": game["PTS"] + game["REB"],
            "P+A": game["PTS"] + game["AST"],
            "R+A": game["REB"] + game["AST"],
        }

        stats = [s for s in latest_proj.keys() if s not in ["timestamp", "player"]]
        correct_count = sum(real_stats.get(s, 0) >= latest_proj[s] for s in stats)
        total_stats = len(stats)

        st.markdown(
            f"**🗓️ Game Date:** {game['GAME_DATE'].strftime('%B %d, %Y')}  |  "
            f"**✅ {correct_count}/{total_stats} Stats Projected Correctly**"
        )

        # --- Metric Cards ---
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

        # --- Dual-color comparison chart ---
        proj_values = [latest_proj[s] for s in stats]
        real_values = [real_stats.get(s, 0) for s in stats]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=stats, y=proj_values, name="AI Projection", marker_color=team_color))
        fig.add_trace(go.Bar(x=stats, y=real_values, name="Actual Result", marker_color=contrast_color))
        fig.update_layout(
            title=f"{player_name} — Projection vs. Actual Results ({game['GAME_DATE'].strftime('%b %d, %Y')})",
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

st.caption("View all saved projections here with game dates and accuracy summaries.")
