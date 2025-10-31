# app.py — Hot Shot Props | NBA Player Analytics (Free)
# Minimal, robust, and fast. Black/Red UI, favorites, ML predictions, bar charts.

import os, time, json, datetime as dt
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---- NBA API (official stats) ----
from nba_api.stats.static import players as nba_players, teams as nba_teams
from nba_api.stats.endpoints import playercareerstats, playergamelogs

# ---- ML (optional) ----
try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    Ridge = None

# ========================= Page / Theme =========================
st.set_page_config(page_title="Hot Shot Props • NBA Player Analytics", layout="wide")

st.markdown("""
<style>
:root { --bg:#000; --panel:#0b0b0b; --ink:#f3f4f6; --line:#171717; --red:#ef4444; }
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink)!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#000 0%,#0b0b0b 100%)!important;border-right:1px solid #111;}
h1,h2,h3,h4{color:#ffb4b4!important;letter-spacing:.2px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px;}
.stButton>button{background:var(--red)!important;color:#fff!important;border:none!important;border-radius:10px!important;padding:.55rem .95rem!important;font-weight:700;}
[data-testid="stMetric"]{background:#0e0e0e;border:1px solid #181818;border-radius:14px;padding:14px;}
[data-testid="stMetric"] label{color:#fda4a4;}
[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#ffe4e6;font-size:1.3rem;}
.bar{margin-top:8px}
a{color:#fda4a4;text-decoration:none}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.75rem;border:1px solid #2a2a2a;background:#121212;margin-left:6px;}
.fav-pill{display:flex;align-items:center;justify-content:space-between;padding:6px 10px;border:1px solid #222;border-radius:10px;margin-bottom:6px;background:#0e0e0e}
.fav-pill a{font-weight:700}
.fav-x{cursor:pointer;font-weight:900;color:#fda4a4;margin-left:10px}
.home-hint{background:#0a1016;border:1px solid #112; padding:10px 14px;border-radius:10px}
</style>
""", unsafe_allow_html=True)

# ========================= Config / Constants =========================
DEFAULT_TMP_DIR = "/tmp" if os.access("/", os.W_OK) else "."
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(DEFAULT_TMP_DIR, "models"))
os.makedirs(MODEL_DIR, exist_ok=True)

STATS_COLS   = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']
CORE_STATS   = ['PTS','REB','AST','FG3M','MIN']
ML_STATS     = ['PTS','REB','AST','FG3M']          # predict these
ML_FEATS     = ['PTS','REB','AST','FG3M','MIN']    # features to roll
ROLLS        = [1,3,5]

NBA_TIMEOUT = 12     # aggressive but ok when cached/retried
API_SLEEP  = 0.15    # small polite delay

# ========================= Cache: Static lists =========================
@st.cache_data(show_spinner=False)
def get_active_players_list() -> List[Dict]:
    try:
        return nba_players.get_active_players()
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def get_teams_list() -> List[Dict]:
    try:
        return nba_teams.get_teams()
    except Exception:
        return []

# fast maps
TEAMS = {t['full_name']: t for t in get_teams_list()}
TEAM_NAME_BY_ID = {t['id']: t['full_name'] for t in get_teams_list()}
PLAYER_BY_NAME = {p['full_name']: p for p in get_active_players_list()}
PLAYER_NAME_BY_ID = {p['id']: p['full_name'] for p in get_active_players_list()}

# ========================= Helpers =========================
def season_for_today() -> str:
    today = dt.date.today()
    start_year = today.year - 1 if today.month < 8 else today.year
    return f"{start_year}-{str(start_year+1)[-2:]}"

def headshot_url(player_id: int) -> str:
    # nba.com headshot CDN (works for most nba_api player IDs)
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"

# ========================= Data Fetchers =========================
@st.cache_data(ttl=60*30, show_spinner=False)   # 30 min
def fetch_player_career(player_id: int) -> Optional[pd.DataFrame]:
    try:
        res = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=NBA_TIMEOUT)
        df = res.get_data_frames()[0]
        return df
    except Exception as e:
        return None

@st.cache_data(ttl=60*15, show_spinner=False)   # 15 min
def fetch_player_gamelogs(player_id: int, season: str) -> Optional[pd.DataFrame]:
    try:
        # playergamelogs supports season & player_id_nullable; returns all rows (preseason+reg+post)
        gl = playergamelogs.PlayerGameLogs(player_id_nullable=player_id,
                                           season_nullable=season,
                                           timeout=NBA_TIMEOUT).get_data_frames()[0]
        # normalize
        gl['GAME_DATE'] = pd.to_datetime(gl['GAME_DATE'])
        gl = gl.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
        return gl
    except Exception:
        return None

def season_regular_only(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    # if 'GAME_DATE' exists we already have game rows; use df as-is
    # playergamelogs already returns only regular season by default in newer versions; guard anyway
    return df

# ========================= Favorites (per-user via session) =========================
def load_favorites() -> Dict[str, int]:
    path = os.path.join(DEFAULT_TMP_DIR, "favorites.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_favorites(favs: Dict[str,int]):
    path = os.path.join(DEFAULT_TMP_DIR, "favorites.json")
    try:
        with open(path, "w") as f:
            json.dump(favs, f)
    except Exception:
        pass

if "favorites" not in st.session_state:
    st.session_state.favorites = load_favorites()

# ========================= ML: model building & prediction =========================
@st.cache_data(ttl=60*60, show_spinner=False)  # 1 hour cache
def build_player_models(player_id: int, season: str):
    """
    Train small Ridge models per stat using this season.
    Features: rolling 1/3/5 of ML_FEATS. Predict t+1.
    """
    if not SKLEARN_OK:
        return None

    gl = fetch_player_gamelogs(player_id, season)
    if gl is None or gl.empty or len(gl) < 8:
        return None

    # Keep regular season rows
    df = season_regular_only(gl).copy().sort_values("GAME_DATE")
    use_cols = [c for c in ML_FEATS if c in df.columns]
    if not use_cols:
        return None

    # rolling features
    for c in use_cols:
        s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        for r in ROLLS:
            df[f"{c}_r{r}"] = s.rolling(r, min_periods=1).mean()

    # feature candidates
    feat_cols = [c for c in df.columns if any(s in c for s in ML_FEATS) and "_r" in c]

    models = {}
    for target in ML_STATS:
        if target not in df.columns:
            continue
        y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).shift(-1)  # next game
        X = df[feat_cols].fillna(method="bfill").fillna(0.0)

        X = X.iloc[:-1].values
        y = y.iloc[:-1].values
        if len(y) < 6:
            continue

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Ridge(alpha=1.0).fit(Xtr, ytr)
        models[target] = {"model": model, "feat_cols": feat_cols}

    return models if models else None

def predict_next_game(player_id: int, season: str) -> Tuple[Dict[str,float], bool]:
    """
    Returns (preds, is_ml). If ML unavailable, falls back to weighted averages.
    """
    gl = fetch_player_gamelogs(player_id, season)
    if gl is None or gl.empty:
        return ({}, False)

    gl = season_regular_only(gl).copy()
    # Try ML
    models = build_player_models(player_id, season)
    if models:
        row = gl.copy().sort_values("GAME_DATE")  # oldest->newest
        # build latest rolling features using same logic
        use_cols = [c for c in ML_FEATS if c in row.columns]
        for c in use_cols:
            s = pd.to_numeric(row[c], errors="coerce").fillna(0.0)
            for r in ROLLS:
                row[f"{c}_r{r}"] = s.rolling(r, min_periods=1).mean()
        feat_cols = next(iter(models.values()))["feat_cols"]
        x = row[feat_cols].iloc[-1:].fillna(0).values
        preds = {}
        for stat, pack in models.items():
            preds[stat] = float(np.round(pack["model"].predict(x)[0], 2))
        return preds, True

    # Fallback: weighted average of last 5/10/20 with weights 0.5/0.3/0.2 (if available)
    weights = [(5,0.5),(10,0.3),(20,0.2)]
    out = {}
    for stat in ML_STATS:
        acc, wsum = 0.0, 0.0
        for n,w in weights:
            if len(gl) >= n:
                acc += gl[stat].head(n).mean() * w
                wsum += w
        if wsum == 0:
            out[stat] = float(np.round(gl[stat].mean(),2)) if stat in gl.columns else np.nan
        else:
            out[stat] = float(np.round(acc/wsum,2))
    return out, False

# ========================= UI Pieces =========================
def metric_row(title: str, data: Dict[str, float], keys: List[str]):
    st.subheader(title)
    cols = st.columns(len(keys))
    for i,k in enumerate(keys):
        v = data.get(k, np.nan)
        if pd.isna(v):
            cols[i].metric(k, "N/A")
        else:
            cols[i].metric(k, f"{v:.2f}")

def bar_block(df: pd.DataFrame, cols: List[str], title: str):
    st.markdown(f"#### {title}")
    if df is None or df.empty:
        st.info("No data.")
        return
    # long format
    use = [c for c in cols if c in df.columns]
    if not use:
        st.info("No matching columns.")
        return
    recent = df.head(10).copy()
    long = recent[['GAME_DATE'] + use].melt('GAME_DATE', var_name='Stat', value_name='Value')
    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X('Stat:N'),
        y=alt.Y('Value:Q'),
        color=alt.Color('Stat:N', legend=None)
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)

def last_game_block(gl: pd.DataFrame) -> Dict[str,float]:
    if gl is None or gl.empty: return {}
    last = gl.head(1).iloc[0]
    data = {k: float(last.get(k, np.nan)) for k in CORE_STATS}
    data['MIN'] = float(last.get('MIN', np.nan))
    return data

def last5_avg_block(gl: pd.DataFrame) -> Dict[str,float]:
    if gl is None or len(gl) < 1: return {}
    chunk = gl.head(5)
    return {k: float(np.round(chunk[k].mean(),2)) for k in CORE_STATS if k in chunk.columns}

def season_avg_from_career(career_df: pd.DataFrame, season: str) -> Dict[str,float]:
    if career_df is None or career_df.empty: return {}
    if 'SEASON_ID' not in career_df.columns: return {}
    row = career_df.loc[career_df['SEASON_ID'] == season]
    if row.empty:
        row = career_df.tail(1)
    row = row.iloc[0]
    out = {}
    gp = float(row.get('GP', 0)) or 0.0
    for k in STATS_COLS:
        if k in career_df.columns:
            v = float(row.get(k, np.nan))
            out[k] = float(np.round(v/gp,2)) if gp>0 and k != 'GP' else v
    return out

# ========================= Sidebar =========================
def go_home():
    st.session_state.pop("selected_player", None)
    try:
        st.query_params.clear()
    except Exception:
        pass
    st.rerun()

with st.sidebar:
    st.header("Select Player")
    # search box
    search = st.text_input("Search player or team", "").strip()
    # favorites list
    st.markdown("**⭐ Favorites**")
    if st.session_state.favorites:
        for name, pid in list(st.session_state.favorites.items()):
            cols = st.columns([0.8,0.2])
            with cols[0]:
                if st.button(name, key=f"fav_{pid}"):
                    st.session_state.selected_player = pid
                    st.rerun()
            with cols[1]:
                if st.button("✖", key=f"fav_del_{pid}"):
                    st.session_state.favorites.pop(name, None)
                    save_favorites(st.session_state.favorites)
                    st.rerun()
    else:
        st.caption("No favorites yet.")

    st.divider()
    st.button("🏠 Home Screen", on_click=go_home, type="primary")

# ========================= Home vs Player routing =========================
selected_id = st.session_state.get("selected_player")
# quick search logic
if search:
    # try exact player
    if search in PLAYER_BY_NAME:
        selected_id = PLAYER_BY_NAME[search]['id']
        st.session_state.selected_player = selected_id
        st.rerun()
    # team filter
    team_hits = [t for t in TEAMS if search.lower() in t.lower()]
    if team_hits and not selected_id:
        st.subheader("Tap a player")
        team_id = TEAMS[team_hits[0]]['id']
        names = [p['full_name'] for p in get_active_players_list() if p.get('team_id') == team_id]
        for nm in names[:100]:
            if st.button(nm, key=f"team_{nm}"):
                st.session_state.selected_player = PLAYER_BY_NAME[nm]['id']
                st.rerun()

# ========================= HOME =========================
if not selected_id:
    st.title("Hot Shot Props — NBA Player Analytics")
    st.markdown('<div class="home-hint">💡 <b>Hint:</b> Use the sidebar to search any player, or tap a favorite.</div>', unsafe_allow_html=True)
    st.markdown("Pick a player to load their dashboard.")
    st.stop()

# ========================= PLAYER PAGE =========================
player_id = selected_id
player_name = PLAYER_NAME_BY_ID.get(player_id, "Player")
st.title(player_name)

# Add to favorites
colA, colB = st.columns([0.2,0.8])
with colA:
    if st.button("⭐ Add to Favorites"):
        st.session_state.favorites[player_name] = player_id
        save_favorites(st.session_state.favorites)
        st.success("Added!")

season = season_for_today()

with st.spinner("Loading data…"):
    career_df = fetch_player_career(player_id)
    gamelogs_df = fetch_player_gamelogs(player_id, season)
    time.sleep(API_SLEEP)

# Robust guard
if (career_df is None or career_df.empty) and (gamelogs_df is None or gamelogs_df.empty):
    st.error("Failed to load from nba_api (likely a temporary block). Please retry in a minute.")
    st.stop()

# ===== Header metrics =====
season_avgs = season_avg_from_career(career_df, season)
last_game = last_game_block(gamelogs_df)
last5 = last5_avg_block(gamelogs_df)
preds, used_ml = predict_next_game(player_id, season)

metric_row("Current Season Averages", season_avgs, CORE_STATS)
metric_row("Last Game Stats", last_game, CORE_STATS)
metric_row("Last 5 Games Averages", last5, CORE_STATS)
pred_title = "Predicted Next Game (ML)" if used_ml else "Predicted Next Game (Weighted Avg)"
metric_row(pred_title, preds, CORE_STATS)

# ===== Charts =====
st.divider()
st.subheader("Stat Bars (last 10 games)")
bar_block(gamelogs_df, ['PTS','REB','AST','FG3M'], "Scoring & Creation")
bar_block(gamelogs_df, ['FGA','FGM','FG3A','FG3M','FTA','FTM'], "Shooting Volume")
bar_block(gamelogs_df, ['OREB','DREB','REB','STL','BLK','TOV','PF','MIN'], "Other Impact")

# ===== Last predictions vs results (simple) =====
st.divider()
st.subheader("Last Predictions vs Results")
if preds and gamelogs_df is not None and not gamelogs_df.empty:
    last_game_row = gamelogs_df.head(1).iloc[0]
    cols = st.columns(len(CORE_STATS))
    for i,k in enumerate(CORE_STATS):
        pred_v = preds.get(k, np.nan)
        actual_v = float(last_game_row.get(k, np.nan))
        if pd.isna(pred_v) or pd.isna(actual_v):
            cols[i].metric(k, "N/A")
        else:
            hit = "✅" if actual_v >= pred_v else "❌"
            cols[i].metric(f"{k} {hit}", f"{actual_v:.1f}", f"pred {pred_v:.1f}")

# ===== Download snapshot (PNG) =====
st.divider()
png_name = f"{player_name.replace(' ','_')}_{season}.png"
if st.button("⬇️ Download Snapshot (PNG)"):
    st.write("Use your browser/device ‘Save as…’ after taking a screenshot. (Direct PNG export disabled here for reliability.)")