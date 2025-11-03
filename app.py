# app.py — ESPN-style NBA Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests, os
from datetime import datetime
from io import BytesIO
from PIL import Image

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Hot Shot Props | NBA Dashboard",
    page_icon="🔥",
    layout="wide"
)

# -------------------------------------------------
# SETTINGS / API KEYS
# -------------------------------------------------
ODDS_API_KEY = "e11d4159145383afd3a188f99489969e"
BALL_API = "https://www.balldontlie.io/api/v1"
ASSETS_DIR = "assets"
PLAYER_PHOTO_DIR = os.path.join(ASSETS_DIR, "player_photos")
TEAM_LOGO_DIR = os.path.join(ASSETS_DIR, "team_logos")
os.makedirs(PLAYER_PHOTO_DIR, exist_ok=True)
os.makedirs(TEAM_LOGO_DIR, exist_ok=True)

# -------------------------------------------------
# STYLES
# -------------------------------------------------
st.markdown("""
<style>
body {background-color:#121212;color:#F5F5F5;font-family:'Roboto',sans-serif;}
h1,h2,h3,h4{font-family:'Oswald',sans-serif;color:#E50914;}
div[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1rem;}
.player-card{
  background:linear-gradient(180deg,#1E1E1E 0%,#191919 100%);
  border-radius:18px;padding:15px;text-align:center;
  transition:transform .3s ease,box-shadow .3s ease;
}
.player-card:hover{transform:scale(1.03);box-shadow:0 0 25px rgba(229,9,20,.4);}
.metric{font-size:28px;font-weight:700;color:#00E676;margin:10px 0;}
.badge{background-color:#E50914;border-radius:8px;padding:3px 8px;color:white;font-size:12px;font-weight:600;}
.trend-up{color:#00E676;}
.trend-down{color:#FF1744;}
.chart-box{
  background:#1C1C1C;padding:20px;border-radius:15px;
  margin-top:15px;box-shadow:0 0 10px rgba(0,0,0,.3);
}
[data-baseweb="tab-list"]{gap:12px;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# UTILS
# -------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_json(url):
    try:
        return requests.get(url, timeout=10).json()
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def get_player_photo(name):
    safe_name = name.replace(" ", "_").lower()
    local_path = os.path.join(PLAYER_PHOTO_DIR, f"{safe_name}.jpg")
    if os.path.exists(local_path):
        return local_path
    # ESPN CDN format guess
    search = fetch_json(f"{BALL_API}/players?search={name}")
    if search.get("data"):
        player = search["data"][0]
        espn_id = player["id"]
        img_url = f"https://a.espncdn.com/i/headshots/nba/players/full/{espn_id}.png"
        try:
            img = Image.open(BytesIO(requests.get(img_url, timeout=5).content))
            img.save(local_path)
            return local_path
        except Exception:
            pass
    return None

@st.cache_data(ttl=3600)
def get_team_logo(team_abbr):
    safe = team_abbr.lower()
    local_path = os.path.join(TEAM_LOGO_DIR, f"{safe}.png")
    if os.path.exists(local_path):
        return local_path
    try:
        url = f"https://loodibee.com/wp-content/uploads/nba-{safe}-logo.png"
        img = Image.open(BytesIO(requests.get(url, timeout=5).content))
        img.save(local_path)
        return local_path
    except Exception:
        return None

@st.cache_data(ttl=900)
def load_predictions():
    if os.path.exists("predictions.csv"):
        return pd.read_csv("predictions.csv")
    # Example fallback
    data = {
        "player_name":["Jayson Tatum","Donovan Mitchell","Luka Doncic"],
        "team":["BOS","CLE","DAL"],
        "prop_type":["Points","Points","Points"],
        "line":[27.5,26.5,31.5],
        "projection":[31.2,28.8,35.1],
        "ev":[0.14,0.08,0.09],
        "confidence":[0.87,0.79,0.83],
        "sportsbook_line":["FanDuel","FanDuel","FanDuel"],
        "edge_value":[14,8,9]
    }
    return pd.DataFrame(data)

@st.cache_data(ttl=900)
def get_odds():
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?regions=us&markets=player_points&apiKey={ODDS_API_KEY}"
    return fetch_json(url)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🏀 Hot Shot Props — NBA Dashboard")
st.subheader("ESPN-Style AI-Powered Player Prop Insights")

# -------------------------------------------------
# DATA
# -------------------------------------------------
preds = load_predictions()

# Player list / selection
player_names = preds["player_name"].unique()
selected = st.selectbox("Select a Player", player_names)

pdata = preds[preds["player_name"] == selected].iloc[0]
photo_path = get_player_photo(selected)
team_logo = get_team_logo(pdata["team"])

col1, col2 = st.columns([1,2])
with col1:
    if photo_path:
        st.image(photo_path, use_column_width=True)
with col2:
    if team_logo:
        st.image(team_logo, width=80)
    st.markdown(f"### {selected} ({pdata['team']}) — {pdata['prop_type']}")
    st.metric("Sportsbook Line", f"{pdata['line']} {pdata['prop_type']}")
    st.metric("Model Projection", f"{pdata['projection']:.1f}")
    st.metric("Edge", f"+{pdata['edge_value']}%")
    st.progress(float(pdata['confidence']))

# -------------------------------------------------
# TABS
# -------------------------------------------------
tabs = st.tabs(["📊 Stats", "🎯 Insights", "📈 Trends"])

# TAB 1: STATS
with tabs[0]:
    st.markdown("### Radar Comparison")
    radar_fig = go.Figure()
    cats = ['Line','Projection']
    vals = [pdata['line'], pdata['projection']]
    radar_fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill='toself', line_color="#E50914"
    ))
    radar_fig.update_layout(
        polar=dict(bgcolor="#1E1E1E", radialaxis=dict(visible=True,range=[0, max(vals)+10])),
        paper_bgcolor="#121212", font_color="#F5F5F5"
    )
    st.plotly_chart(radar_fig, use_container_width=True)

# TAB 2: INSIGHTS
with tabs[1]:
    st.markdown("### AI Insights")
    line = pdata["line"]; proj = pdata["projection"]; edge = pdata["edge_value"]; conf = pdata["confidence"]
    trend = "🔥 Hot" if proj > line else "🧊 Cold"
    insight = f"{trend} — Model projects **{proj:.1f} {pdata['prop_type']}** vs line of **{line}** ({edge:+.0f}% edge, confidence {conf*100:.0f}%)"
    st.info(insight)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=conf*100,
        title={'text':"Model Confidence %"},
        gauge={'axis':{'range':[0,100]},'bar':{'color':"#E50914"},
               'steps':[{'range':[0,50],'color':"#333"},
                        {'range':[50,75],'color':"#666"},
                        {'range':[75,100],'color':"#E50914"}]}
    ))
    gauge.update_layout(paper_bgcolor="#121212", font_color="#F5F5F5")
    st.plotly_chart(gauge, use_container_width=True)

# TAB 3: TRENDS
with tabs[2]:
    st.markdown("### Simulated Recent Game Log")
    games = pd.DataFrame({
        "Game":[f"G{i}" for i in range(1,6)],
        "Points":np.random.randint(20,40,5),
        "Rebounds":np.random.randint(3,12,5),
        "Assists":np.random.randint(2,10,5)
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(x=games["Game"], y=games["Points"], name="Points", marker_color="#E50914"))
    fig.add_trace(go.Scatter(x=games["Game"], y=games["Rebounds"], mode='lines+markers', name="Rebounds", line=dict(color="#00E676")))
    fig.add_trace(go.Scatter(x=games["Game"], y=games["Assists"], mode='lines+markers', name="Assists", line=dict(color="#29B6F6")))
    fig.update_layout(paper_bgcolor="#121212", plot_bgcolor="#121212", font_color="#F5F5F5")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(f"""
---
<div style='text-align:center;color:#777;font-size:13px;margin-top:20px;'>
Hot Shot Props © {datetime.now().year} | Powered by AI Sports Analytics
</div>
""", unsafe_allow_html=True)
