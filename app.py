# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Hot Shot Props",
    layout="wide",
    page_icon="🔥"
)

# -------------------- CUSTOM ESPN STYLE --------------------
st.markdown("""
<style>
/* Global */
body {
    background-color: #121212;
    color: #F5F5F5;
    font-family: 'Roboto', sans-serif;
}
h1, h2, h3, h4 {
    font-family: 'Oswald', sans-serif;
    color: #E50914;
    letter-spacing: 0.5px;
}
div[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1rem;}
/* Player Cards */
.player-card {
    background: linear-gradient(180deg, #1E1E1E 0%, #191919 100%);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.player-card:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 25px rgba(229,9,20,0.4);
}
.metric {
    font-size: 28px;
    font-weight: 700;
    color: #00E676;
    margin: 10px 0;
}
.badge {
    background-color: #E50914;
    border-radius: 8px;
    padding: 3px 8px;
    color: white;
    font-size: 12px;
    font-weight: 600;
}
.trend-up {color: #00E676;}
.trend-down {color: #FF1744;}
/* Chart Containers */
.chart-box {
    background: #1C1C1C;
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
    box-shadow: 0 0 10px rgba(0,0,0,0.3);
}
/* Tabs */
[data-baseweb="tab-list"] {
    gap: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.title("🏀 Hot Shot Props Dashboard")
st.subheader("ESPN-Style Live Player Insights and AI Predictions")

# -------------------- DUMMY PLAYER DATA --------------------
players = [
    {"name": "Jayson Tatum", "team": "BOS", "pts": 31.4, "reb": 8.2, "ast": 5.4, "trend": "+6.2", "hot": True},
    {"name": "Donovan Mitchell", "team": "CLE", "pts": 27.8, "reb": 4.1, "ast": 6.3, "trend": "+4.0", "hot": True},
    {"name": "Luka Doncic", "team": "DAL", "pts": 35.1, "reb": 9.8, "ast": 8.9, "trend": "-2.4", "hot": False},
    {"name": "Giannis Antetokounmpo", "team": "MIL", "pts": 29.7, "reb": 11.3, "ast": 6.1, "trend": "+3.5", "hot": True},
    {"name": "Shai Gilgeous-Alexander", "team": "OKC", "pts": 30.9, "reb": 5.3, "ast": 6.2, "trend": "+2.0", "hot": True}
]

# -------------------- PLAYER CARD SECTION --------------------
cols = st.columns(5)
for i, p in enumerate(players):
    with cols[i]:
        trend_color = "trend-up" if p["hot"] else "trend-down"
        st.markdown(f"""
        <div class="player-card">
            <h3>{p['name']} <span class="badge">{p['team']}</span></h3>
            <p class="metric">{p['pts']} PPG</p>
            <p><span class="{trend_color}">{p['trend']} vs avg</span></p>
            <p>Reb: {p['reb']} | Ast: {p['ast']}</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------- PLAYER DETAIL VIEW --------------------
st.markdown("---")
selected_player = st.selectbox("Select a player to view detailed insights:", [p["name"] for p in players])

player_data = next(p for p in players if p["name"] == selected_player)

tabs = st.tabs(["📊 Stats", "🎯 Insights", "📈 Trends"])

# -------------------- TAB 1: STATS --------------------
with tabs[0]:
    st.markdown(f"### {player_data['name']} — Season Performance")

    radar_fig = go.Figure()
    categories = ['Points', 'Rebounds', 'Assists']
    values = [player_data['pts'], player_data['reb'], player_data['ast']]
    radar_fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Player Stats',
        line_color="#E50914"
    ))
    radar_fig.update_layout(
        polar=dict(
            bgcolor="#1E1E1E",
            radialaxis=dict(visible=True, range=[0, 40], color="#888"),
        ),
        showlegend=False,
        paper_bgcolor="#121212",
        font_color="#F5F5F5",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(radar_fig, use_container_width=True)

# -------------------- TAB 2: INSIGHTS --------------------
with tabs[1]:
    st.markdown(f"### AI Insights for {player_data['name']}")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="chart-box">
        <h4>🔥 Hot Form Indicator</h4>
        <p>Model detects upward trend in last 5 games.</p>
        <ul>
        <li>3+ straight overs on points</li>
        <li>True shooting: +5.4% above season avg</li>
        <li>Usage rate: up 7.2%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=84,
            title={'text': "Edge Confidence %"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#E50914"},
                   'bgcolor': "#1E1E1E",
                   'borderwidth': 2,
                   'bordercolor': "#333",
                   'steps': [
                       {'range': [0, 50], 'color': "#333"},
                       {'range': [50, 75], 'color': "#666"},
                       {'range': [75, 100], 'color': "#E50914"}
                   ]}
        ))
        fig.update_layout(
            paper_bgcolor="#121212",
            font_color="#F5F5F5",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------- TAB 3: TRENDS --------------------
with tabs[2]:
    st.markdown(f"### {player_data['name']} — Recent Game Log")
    games = pd.DataFrame({
        "Game": [f"G{i}" for i in range(1, 6)],
        "Points": np.random.randint(20, 40, 5),
        "Rebounds": np.random.randint(3, 12, 5),
        "Assists": np.random.randint(2, 10, 5)
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(x=games["Game"], y=games["Points"], name="Points", marker_color="#E50914"))
    fig.add_trace(go.Scatter(x=games["Game"], y=games["Rebounds"], mode='lines+markers', name="Rebounds", line=dict(color="#00E676")))
    fig.add_trace(go.Scatter(x=games["Game"], y=games["Assists"], mode='lines+markers', name="Assists", line=dict(color="#29B6F6")))
    fig.update_layout(
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        font_color="#F5F5F5",
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- FOOTER --------------------
st.markdown("""
---
<div style='text-align:center; color:#777; font-size:13px; margin-top:20px;'>
Hot Shot Props © {year} | Powered by AI Sports Analytics
</div>
""".format(year=datetime.now().year), unsafe_allow_html=True)
