import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, playercareerstats

st.set_page_config(page_title="Hot Shot Props — Player AI", layout="wide")

# ---------------------- STYLE ----------------------
st.markdown("""
<style>
body { background-color: #0a0a0a; color: white; }
h1, h2, h3, h4 { color: white; }
.metric-card {
    background-color: #1a1a1a;
    border: 2px solid #E50914;
    border-radius: 12px;
    padding: 10px;
    text-align: center;
    margin: 4px;
}
.metric-value {
    font-size: 28px;
    color: #E50914;
    font-weight: bold;
}
.download-btn a {
    display: inline-block;
    background: linear-gradient(90deg,#E50914,#ff7300);
    color: white;
    font-weight: bold;
    padding: 10px 18px;
    border-radius: 10px;
    text-decoration: none;
    transition: all 0.2s ease-in-out;
    box-shadow: 0px 0px 12px #E50914;
}
.download-btn a:hover {
    box-shadow: 0px 0px 18px #ff7300;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------------- SEARCH BAR ----------------------
st.markdown("### Search or Browse Player")
nba_players = players.get_active_players()
player_names = sorted([p["full_name"] for p in nba_players])
player = st.selectbox("Select Player", player_names)

if not player:
    st.stop()

# ---------------------- BASIC PLAYER INFO ----------------------
player_id = next(p["id"] for p in nba_players if p["full_name"] == player)

@st.cache_data(ttl=900)
def get_player_image(player_name):
    try:
        formatted = player_name.lower().replace(" ", "_")
        url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{formatted}.png"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content))
    except Exception:
        pass
    return None

photo = get_player_image(player)
if photo:
    st.image(photo, width=120)
else:
    st.markdown("🧍‍♂️")

# ---------------------- TEAM + NEXT GAME ----------------------
@st.cache_data(ttl=900)
def get_team_and_next_game(player_name):
    try:
        sched = requests.get(
            "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json", timeout=10
        ).json()
        for date in sched["leagueSchedule"]["gameDates"]:
            for g in date["games"]:
                for side in ["homeTeam", "awayTeam"]:
                    if player_name.lower().split(" ")[-1] in g[side]["teamName"].lower():
                        team = g[side]["teamName"]
                        opp = g["awayTeam"]["teamName"] if side == "homeTeam" else g["homeTeam"]["teamName"]
                        loc = "Home" if side == "homeTeam" else "Away"
                        return team, f"Next Game: vs {opp} ({loc})"
        return "Team Unknown", "Next Game: TBD"
    except Exception:
        return "Team Unknown", "Next Game: TBD"

team_name, next_game = get_team_and_next_game(player)
st.markdown(f"### {player}")
st.markdown(f"**Team:** {team_name}")
st.markdown(f"*{next_game}*")
st.markdown("---")

# ---------------------- FETCH STATS ----------------------
@st.cache_data(ttl=3600)
def get_games(pid, season):
    try:
        g = playergamelog.PlayerGameLog(player_id=pid, season=season)
        df = g.get_data_frames()[0]
        return df
    except Exception:
        return pd.DataFrame()

CURRENT_SEASON = "2025-26"
LAST_SEASON = "2024-25"
games_current = get_games(player_id, CURRENT_SEASON)
games_prev = get_games(player_id, LAST_SEASON)

if games_current.empty and games_prev.empty:
    st.warning("No game data found for this player.")
    st.stop()

games = pd.concat([games_current, games_prev]).reset_index(drop=True)
games["PTS+REB"] = games["PTS"] + games["REB"]
games["PTS+AST"] = games["PTS"] + games["AST"]
games["REB+AST"] = games["REB"] + games["AST"]
games["PRA"] = games["PTS"] + games["REB"] + games["AST"]

# ---------------------- SEASON AVERAGE ----------------------
season_avg = games.mean(numeric_only=True).to_dict()

# ---------------------- SIMPLE ML MODEL ----------------------
def train_model(df):
    df = df[["PTS","REB","AST","FG3M","STL","BLK","TOV","PTS+REB","PTS+AST","REB+AST","PRA"]]
    X = df.index.values.reshape(-1,1)
    models = {}
    for col in df.columns:
        m = RandomForestRegressor(n_estimators=100, random_state=42)
        m.fit(X, df[col].values)
        models[col] = m
    return models

model = train_model(games)
next_game_index = np.array([[len(games)+1]])
pred_next = {stat: round(float(model[stat].predict(next_game_index)[0]),1) for stat in model.keys()}

# ---------------------- METRIC CARDS ----------------------
st.markdown("## 🧠 AI Predicted Next Game Stats")
cols = st.columns(4)
i = 0
for stat, val in pred_next.items():
    with cols[i % 4]:
        st.markdown(f"""
        <div class='metric-card'>
            <div>{stat}</div>
            <div class='metric-value'>{val}</div>
        </div>
        """, unsafe_allow_html=True)
    i += 1

# ---------------------- COMPARISON BAR CHART ----------------------
metrics = list(pred_next.keys())
pred_values = [pred_next[m] for m in metrics]
avg_values = [season_avg.get(m, 0) for m in metrics]

fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0a0a0a")
x = np.arange(len(metrics))
bar_width = 0.35

bars_pred = ax.bar(
    x - bar_width/2,
    pred_values,
    width=bar_width,
    color="#E50914",
    label="AI Prediction"
)
bars_avg = ax.bar(
    x + bar_width/2,
    avg_values,
    width=bar_width,
    color="#1E90FF",
    label="Season Avg"
)

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title("AI Prediction vs. Season Average", color="white", fontsize=14, pad=15)
ax.legend(facecolor="#1e1e1e", edgecolor="none", labelcolor="white")
ax.tick_params(colors="white")
ax.set_facecolor("#0a0a0a")

st.pyplot(fig)

# ---------------------- DOWNLOAD PNG BUTTON ----------------------
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="#0a0a0a")
buf.seek(0)
b64 = base64.b64encode(buf.read()).decode()
st.markdown(f"""
<div class='download-btn'>
    <a href="data:file/png;base64,{b64}" download="{player}_AI_Prediction_vs_SeasonAvg.png">
    📸 Download Chart as PNG
    </a>
</div>
""", unsafe_allow_html=True)
