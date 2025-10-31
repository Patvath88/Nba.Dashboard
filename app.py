# app.py — Hot Shot Props | NBA Player Analytics (Free, One-Page)
# - Uses nba_api (no paid key)
# - Auto-refresh every 60s (no toggles)
# - Single-page layout w/ header, headshot, metric rows, trend charts
# - NBA.com-inspired dark theme
# - Robust caching and small delays to be gentle on nba_api
# - FIX: use scoreboardv2 (not scoreboardv3)

import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playercareerstats, playergamelogs, scoreboardv2
import pandas as pd
import numpy as np
import re, time, datetime as dt
import altair as alt
import requests
from io import BytesIO

# ---------------------------------
# Page config + global constants
# ---------------------------------
st.set_page_config(
    page_title="Hot Shot Props • NBA Player Analytics (Free)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto refresh every 60s (silent background reload)
st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

STATS_COLS = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']
PREDICT_COLS = ['PTS','AST','REB','FG3M']
API_SLEEP = 0.25  # be gentle with endpoints

# ---------------------------------
# Styling (NBA.com-ish)
# ---------------------------------
st.markdown("""
<style>
:root {
  --bg: #0f1116; --panel:#121722; --ink:#e5e7eb; --muted:#9aa3b2;
  --blue:#1d4ed8; --blue2:#1e3a8a; --line:#1f2a44;
}
html, body, [data-testid="stAppViewContainer"] { background:var(--bg); color:var(--ink); }
h1,h2,h3,h4 { color:#b3d1ff !important; letter-spacing:.2px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0f1629 0%,#0f1116 100%); border-right:1px solid #1f2937; }
.stButton > button { background:var(--blue); color:white; border:none; border-radius:10px; padding:.6rem 1rem; font-weight:700; }
.stButton > button:hover { background:#2563eb; }
[data-testid="stMetric"]{
  background:var(--panel); padding:16px; border-radius:16px; border:1px solid var(--line);
  box-shadow:0 8px 28px rgba(0,0,0,.35);
}
[data-testid="stMetric"] label{ color:#cfe1ff; }
[data-testid="stMetric"] div[data-testid="stMetricValue"]{ color:#e0edff; font-size:1.5rem; }
.stDataFrame { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:4px; }
.card {
  background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:16px;
  box-shadow:0 8px 28px rgba(0,0,0,.35);
}
.badge{ display:inline-block; padding:4px 10px; font-size:.8rem; border:1px solid var(--line); border-radius:9999px; background:#0b1222; color:#9bd1ff; }
.stProgress > div > div > div > div { background-color: var(--blue) !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Helpers
# ---------------------------------
def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def extract_opp_from_matchup(matchup: str) -> str | None:
    if not isinstance(matchup, str):
        return None
    m = re.search(r'@\s*([A-Z]{3})|vs\.\s*([A-Z]{3})|VS\.\s*([A-Z]{3})', matchup, re.IGNORECASE)
    if m:
        return (m.group(1) or m.group(2) or m.group(3)).upper()
    return None

def cdn_headshot(player_id: int, size: str = "1040x760") -> BytesIO | None:
    """
    Fetches NBA media day headshot from the official CDN.
    Common sizes: 1040x760, 260x190
    """
    url = f"https://cdn.nba.com/headshots/nba/latest/{size}/{player_id}.png"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return BytesIO(r.content)
    except Exception:
        pass
    return None

# ---------------------------------
# Cached data
# ---------------------------------
@st.cache_data(ttl=6*3600)
def get_active_players():
    try: return players.get_active_players()
    except Exception as e:
        st.error(f"Error fetching active players: {e}")
        return []

@st.cache_data(ttl=12*3600)
def get_all_teams():
    try: return teams.get_teams()
    except Exception as e:
        st.error(f"Error fetching teams: {e}")
        return []

@st.cache_data(ttl=3600)
def fetch_player(player_id: int):
    """
    Returns (career_by_season_df, career_logs_df)
    """
    try:
        time.sleep(API_SLEEP)
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]
        seasons = career_stats['SEASON_ID'].tolist() if not career_stats.empty else []
        logs_list = []
        for s in seasons:
            try:
                time.sleep(API_SLEEP)
                df = playergamelogs.PlayerGameLogs(player_id_nullable=player_id, season_nullable=s).get_data_frames()[0]
                if df is not None and not df.empty:
                    logs_list.append(df)
            except Exception as se:
                st.warning(f"Game logs failed for {s}: {se}")
        logs = pd.concat(logs_list, ignore_index=True) if logs_list else pd.DataFrame()
        if 'GAME_DATE' in logs.columns:
            logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
            logs = logs.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
        return career_stats, logs
    except Exception as e:
        st.error(f"Failed to fetch player data: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=600)
def next_game_for_team(team_id: int, lookahead_days: int = 10):
    """
    Scans the next N days via ScoreboardV2 to find the next game for the team.
    Returns dict like {'date': 'YYYY-MM-DD', 'opp_abbr': 'XXX', 'home': True/False} or None.
    """
    if team_id is None:
        return None
    today = dt.date.today()
    team_map = {t['id']: t for t in get_all_teams()}  # id -> team dict (includes abbreviation)
    for d in range(lookahead_days):
        day = today + dt.timedelta(days=d)
        try:
            time.sleep(API_SLEEP)
            # ScoreboardV2 expects 'MM/DD/YYYY'
            sb = scoreboardv2.ScoreboardV2(game_date=day.strftime("%m/%d/%Y"))
            frames = sb.get_data_frames()
            game_header = None
            for f in frames:
                if {'HOME_TEAM_ID','VISITOR_TEAM_ID'}.issubset(set(f.columns)):
                    game_header = f
                    break
            if game_header is None or game_header.empty:
                continue

            for _, row in game_header.iterrows():
                home_id = int(row.get('HOME_TEAM_ID', -1))
                away_id = int(row.get('VISITOR_TEAM_ID', -1))
                if team_id in (home_id, away_id):
                    opp_id = away_id if team_id == home_id else home_id
                    opp_abbr = team_map.get(opp_id, {}).get('abbreviation', 'TBD')
                    return {'date': day.strftime("%Y-%m-%d"), 'opp_abbr': opp_abbr, 'home': (team_id == home_id)}
        except Exception:
            continue
    return None

# ---------------------------------
# Computations
# ---------------------------------
def recent_averages(logs: pd.DataFrame) -> dict:
    out = {}
    if logs is None or logs.empty: return out
    df = logs.copy()
    cols = safe_cols(df, STATS_COLS)
    if len(df) >= 5 and cols:
        sub = df.head(5)
        out['Last 5 Avg'] = sub[cols].mean().to_frame(name='Last 5 Avg').T
    return out

def predict_next(logs: pd.DataFrame, season_label: str) -> dict | None:
    if logs is None or logs.empty: return None
    df = logs.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
    cols = safe_cols(df, STATS_COLS)
    if not cols: return None

    # Season avg (prefer matching season label in logs['SEASON_YEAR'])
    season_avg = pd.Series(dtype=float)
    if 'SEASON_YEAR' in df.columns:
        cur = df[df['SEASON_YEAR'] == season_label]
        if not cur.empty: season_avg = cur[cols].mean()
    if season_avg.empty: season_avg = df[cols].mean()

    feats = {}
    for c in cols:
        feats[f'{c}_r5']  = df[c].rolling(5,  min_periods=1).mean().iloc[-1]
        feats[f'{c}_r10'] = df[c].rolling(10, min_periods=1).mean().iloc[-1]
        feats[f'{c}_r20'] = df[c].rolling(20, min_periods=1).mean().iloc[-1]

    preds = {}
    for c in PREDICT_COLS:
        r5, r10, r20 = feats.get(f'{c}_r5', np.nan), feats.get(f'{c}_r10', np.nan), feats.get(f'{c}_r20', np.nan)
        s = season_avg.get(c, 0.0) if pd.notna(season_avg.get(c, np.nan)) else 0.0
        v5  = r5  if pd.notna(r5)  else s
        v10 = r10 if pd.notna(r10) else s
        v20 = r20 if pd.notna(r20) else s
        preds[c] = round(float(0.4*v5 + 0.3*v10 + 0.2*v20 + 0.1*s), 2)
    return preds

# ---------------------------------
# Sidebar — Player Selection
# ---------------------------------
with st.sidebar:
    st.header("Select Player")
    act = get_active_players()
    all_t = get_all_teams()
    name_to_id = {p['full_name']: p['id'] for p in act}
    names_sorted = sorted(name_to_id.keys())

    q = st.text_input("Search", "", help="Filter the list by name")
    filtered = [n for n in names_sorted if q.lower() in n.lower()] if q else names_sorted
    player_name = st.selectbox("Player", filtered, index=None, placeholder="Choose a player")

# ---------------------------------
# Hero/Header
# ---------------------------------
st.markdown("""
<div class="card" style="margin-bottom: 12px;">
  <div style="display:flex; align-items:center; justify-content: space-between;">
    <div>
      <h1 style="margin:0;">Hot Shot Props — NBA Player Analytics</h1>
      <div class="badge">Free Beta • All analytics unlocked • Auto-refresh 60s</div>
    </div>
    <div style="text-align:right; font-size:.9rem; color:#9aa3b2;">Trend sparks • Weighted predictions • One-page UX</div>
  </div>
</div>
""", unsafe_allow_html=True)

if not player_name:
    st.info("Pick a player from the sidebar to load their dashboard.")
    st.stop()

player_id = name_to_id[player_name]
career_df, logs_df = fetch_player(player_id)

if career_df.empty:
    st.warning("No career data available for this player.")
    st.stop()

# Determine current team & latest season
team_abbr = "TBD"
if 'TEAM_ABBREVIATION' in career_df.columns and not career_df.empty:
    team_abbr = career_df['TEAM_ABBREVIATION'].iloc[-1] or team_abbr

latest_season = str(career_df['SEASON_ID'].dropna().iloc[-1]) if 'SEASON_ID' in career_df.columns else "N/A"

# Last game (most recent in logs)
last_game_info = "N/A"
last_game_row = None
if logs_df is not None and not logs_df.empty:
    last_game_row = logs_df.iloc[0]
    lg_date = pd.to_datetime(last_game_row['GAME_DATE']).strftime("%Y-%m-%d") if 'GAME_DATE' in last_game_row else "—"
    lg_opp = extract_opp_from_matchup(last_game_row.get('MATCHUP', '')) or "TBD"
    last_game_info = f"{lg_date} vs {lg_opp}"

# Next game (scan schedule using ScoreboardV2)
team_id_map = {t['abbreviation']: t['id'] for t in all_t}
team_id = team_id_map.get(team_abbr, None)
ng = next_game_for_team(team_id) if team_id is not None else None
next_game_info = f"{ng['date']} vs {ng['opp_abbr']}" if ng else "TBD"

# ---------------------------------
# Top header strip (4 items)
# ---------------------------------
h1, h2, h3, h4 = st.columns([1.3, 0.8, 1.2, 1.2])
with h1: st.metric("Player", player_name)
with h2: st.metric("Team", team_abbr)
with h3: st.metric("Most Recent Game", last_game_info)
with h4: st.metric("Next Game", next_game_info)

# ---------------------------------
# Headshot + Trend sparks row
# ---------------------------------
c_img, c_trends = st.columns([0.28, 0.72])
with c_img:
    img_bytes = cdn_headshot(player_id, "1040x760") or cdn_headshot(player_id, "260x190")
    if img_bytes:
        st.image(img_bytes, use_container_width=True, caption=f"{player_name} — media day headshot")
    else:
        st.info("Headshot not available.")

with c_trends:
    if logs_df is not None and not logs_df.empty:
        # Build compact sparkline dataset for last 10 games
        N = min(10, len(logs_df))
        view = logs_df.head(N).copy()
        view = view.iloc[::-1]  # chronological left->right
        trend_cols = [c for c in ['PTS','REB','AST','FG3M'] if c in view.columns]
        if trend_cols:
            trend_df = view[['GAME_DATE'] + trend_cols].copy()
            trend_df['GAME_DATE'] = pd.to_datetime(trend_df['GAME_DATE'])
            for stat in trend_cols:
                ch = alt.Chart(trend_df).mark_line(point=True).encode(
                    x=alt.X('GAME_DATE:T', title=''),
                    y=alt.Y(f'{stat}:Q', title=stat),
                    tooltip=[alt.Tooltip('GAME_DATE:T', title='Game'), alt.Tooltip(f'{stat}:Q', title=stat)]
                ).properties(height=110)
                st.altair_chart(ch, use_container_width=True)
        else:
            st.info("No trend stats available.")
    else:
        st.info("No recent games to chart.")

# ---------------------------------
# Metric Rows
# ---------------------------------
def metric_row(title: str, data: dict | pd.Series | pd.DataFrame, fallback: str = "N/A"):
    st.markdown(f"#### {title}")
    if data is None:
        data = {}
    if isinstance(data, pd.DataFrame):
        row = data.iloc[0].to_dict() if not data.empty else {}
    elif isinstance(data, pd.Series):
        row = data.to_dict()
    else:
        row = dict(data)

    def fmt(v): 
        return f"{float(v):.2f}" if isinstance(v, (int,float,np.floating)) and pd.notna(v) else fallback

    cols = st.columns(5)
    with cols[0]: st.metric("PTS", fmt(row.get('PTS')))
    with cols[1]: st.metric("REB", fmt(row.get('REB')))
    with cols[2]: st.metric("AST", fmt(row.get('AST')))
    with cols[3]: st.metric("3PM", fmt(row.get('FG3M')))
    with cols[4]: st.metric("MIN", fmt(row.get('MIN')))

# Row 1: Current season averages (per-game)
season_row = None
if not career_df.empty:
    cur = career_df.iloc[-1]  # latest season row
    gp = cur.get('GP', 0)
    if gp and gp > 0:
        season_row = {c: (cur.get(c, 0)/gp) for c in STATS_COLS}
metric_row("Current Season Averages", season_row)

# Row 2: Last game stats
last_game_stats = None
if last_game_row is not None:
    last_game_stats = {c: last_game_row.get(c, np.nan) for c in STATS_COLS}
metric_row("Last Game Stats", last_game_stats)

# Row 3: Last 5 games averages
ra = recent_averages(logs_df)
last5 = None
if 'Last 5 Avg' in ra and not ra['Last 5 Avg'].empty:
    last5 = ra['Last 5 Avg'].iloc[0].to_dict()
metric_row("Last 5 Games Averages", last5)

# Row 4: Prediction for next game
preds = predict_next(logs_df, latest_season)
metric_row("Predicted Next Game (Model)", preds)

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("""
<div style="margin-top:16px; display:flex; justify-content:space-between; align-items:center; opacity:.9;">
  <div class="badge">Early Access • One-page pro UI</div>
  <div style="font-size:.85rem; color:#9aa3b2;">Hot Shot Props © — Built with nba_api & Streamlit</div>
</div>
""", unsafe_allow_html=True)
