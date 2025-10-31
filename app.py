# app.py — Hot Shot Props | NBA Player Analytics
# Home page widgets (last night results, standings by conference, league leaders)
# Clean sidebar (search + favorites), fast fetch (cache + concurrency),
# ML predictions (global background + player ad-hoc) with fallback,
# media-day headshots with overlay team logo, metric rows, bar charts,
# "Last predictions vs results", mobile-first UI, and Download PNG snapshot.

import os, json, re, time, threading, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit.components.v1 as components

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playercareerstats, playergamelogs, scoreboardv2, leagueleaders, leaguedashteamstats
)
# leaguestandingsv3 may not exist for some nba_api versions; we rely on dash team stats.

# ------------------ Optional ML deps ------------------
try:
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    joblib = None
    Ridge = None
    SKLEARN_OK = False

# ------------------ Page & global CSS -----------------
st.set_page_config(page_title="Hot Shot Props • NBA Player Analytics (Free)",
                   layout="wide", initial_sidebar_state="expanded")

# Hide Streamlit multipage sidebar nav (in case a /pages dir exists)
st.markdown("""
<style>
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"],
[data-testid="stSidebarHeader"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# Mobile viewport & iOS notch support
components.html("""
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
""", height=0)

# Global theme + mobile tweaks (black bg, red accents)
st.markdown("""
<style>
:root { --bg:#000; --panel:#0b0b0b; --ink:#f3f4f6; --line:#171717; --red:#ef4444; }
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink)!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#000 0%,#0b0b0b 100%)!important;border-right:1px solid #111;}
.stButton>button{background:var(--red)!important;color:#fff!important;border:none!important;border-radius:10px!important;padding:.55rem .95rem!important;font-weight:700!important;}
h1,h2,h3,h4{color:#ffb4b4!important;letter-spacing:.2px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 8px 28px rgba(0,0,0,.5);}
.badge{display:inline-block;padding:4px 10px;font-size:.8rem;border:1px solid var(--line);border-radius:9999px;background:#140606;color:#fca5a5;}
.hr{border:0;border-top:1px solid var(--line);margin:.75rem 0;}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.75rem;border:1px solid var(--line);margin-left:6px;}
.tag.ok{color:#bbf7d0;border-color:#064e3b;background:#052e16;}
.tag.warn{color:#fde68a;border-color:#7c2d12;background:#3b0a0a;}
.tag.dim{color:#e5e7eb;border-color:#374151;background:#111827;}
[data-testid="stMetric"]{background:#0e0e0e;border:1px solid #181818;border-radius:14px;padding:14px;box-shadow:0 10px 30px rgba(0,0,0,.6);}
[data-testid="stMetric"] label{color:#fda4a4;}
[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#ffe4e6;font-size:1.45rem;}
/* Mobile */
@media (max-width: 780px){
  .block-container { padding: 0.6rem 0.6rem !important; }
  section.main > div { padding-top: 0 !important; }
  [data-testid="column"] { width: 100% !important; display: block !important; }
  [data-testid="stMetric"]{ margin-bottom: 10px !important; }
  .stButton>button{ width: 100% !important; padding: 14px !important; font-size: 1.05rem !important; }
  .stTextInput>div>div>input, .stSelectbox>div>div>div { font-size: 1.05rem !important; }
  .st-emotion-cache-1wmy9hl, .st-emotion-cache-1r6slb0 { height: 260px !important; }
}
[data-testid="stSidebar"] .stButton>button, [data-testid="stSidebar"] .stSelectbox { font-size: 1.02rem !important; }
.leader-card{display:flex;gap:14px;align-items:center;background:#0e0e0e;border:1px solid #181818;border-radius:14px;padding:10px;margin-bottom:10px;}
.leader-img{width:56px;height:56px;border-radius:10px;overflow:hidden;border:1px solid #222;}
.leader-name a{color:#ffb4b4;text-decoration:none;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ------------------ API robustness (fast defaults) ----
NBA_TIMEOUT = 15
NBA_RETRIES = 2
NBA_BACKOFF_BASE = 1.6
API_SLEEP = 0.15
FAST_MODE = True
UI_SEASON_LIMIT = 2

# ------------------ Constants & storage ---------------
STATS_COLS   = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']
PREDICT_COLS = ['PTS','AST','REB','FG3M']

DEFAULT_TMP_DIR = "/tmp" if os.access("/", os.W_OK) else "."
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(DEFAULT_TMP_DIR, "models"))
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILES  = {t: os.path.join(MODEL_DIR, f"model_{t}.pkl") for t in PREDICT_COLS}
FEAT_TEMPLATE = lambda tgt: [f'{tgt}_r5', f'{tgt}_r10', f'{tgt}_r20', f'{tgt}_season_mean', 'IS_HOME', 'DAYS_REST']

USER_ROOT = os.path.join(DEFAULT_TMP_DIR, "userdata", "guest")
os.makedirs(USER_ROOT, exist_ok=True)
FAV_FILE = os.path.join(USER_ROOT, "favorites.json")
PRED_HISTORY_FILE = os.path.join(USER_ROOT, "pred_history.json")

CACHE_ROOT = os.path.join(USER_ROOT, "cache")
os.makedirs(CACHE_ROOT, exist_ok=True)
CACHE_TTL_HOURS = 6

def _cache_path_for_player(player_id: int) -> str:
    return os.path.join(CACHE_ROOT, f"player_{player_id}_logs.parquet")
def _cache_path_csv(player_id: int) -> str:
    return os.path.join(CACHE_ROOT, f"player_{player_id}_logs.csv")

def read_logs_cache(player_id: int) -> Optional[pd.DataFrame]:
    p_parq = _cache_path_for_player(player_id)
    p_csv  = _cache_path_csv(player_id)
    path = p_parq if os.path.exists(p_parq) else (p_csv if os.path.exists(p_csv) else None)
    if not path: return None
    try:
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(path))
        if (dt.datetime.now() - mtime).total_seconds() > CACHE_TTL_HOURS * 3600:
            return None
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return None

def write_logs_cache(player_id: int, df: pd.DataFrame) -> None:
    try:
        if df is not None and not df.empty:
            try:
                df.to_parquet(_cache_path_for_player(player_id), index=False)
            except Exception:
                df.to_csv(_cache_path_csv(player_id), index=False)
    except Exception:
        pass

# ------------------ Small helpers ---------------------
def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def extract_opp_from_matchup(matchup: str) -> Optional[str]:
    if not isinstance(matchup, str): return None
    m = re.search(r'@\s*([A-Z]{3})|vs\.\s*([A-Z]{3})|VS\.\s*([A-Z]{3})', matchup, re.IGNORECASE)
    return (m.group(1) or m.group(2) or m.group(3)).upper() if m else None

def cdn_headshot(player_id: int, size: str = "1040x760") -> Optional[Image.Image]:
    url = f"https://cdn.nba.com/headshots/nba/latest/{size}/{player_id}.png"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200: return Image.open(BytesIO(r.content)).convert("RGBA")
    except Exception: pass
    return None

def cdn_team_logo(team_id: int) -> Optional[Image.Image]:
    for url in [
        f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.png",
        f"https://cdn.nba.com/logos/nba/{team_id}/global/D/logo.png",
    ]:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200: return Image.open(BytesIO(r.content)).convert("RGBA")
        except Exception:
            continue
    return None

def overlay_logo_top_right(headshot: Image.Image, logo: Optional[Image.Image],
                           padding_ratio=0.035, logo_width_ratio=0.22) -> Image.Image:
    base = headshot.copy()
    if not logo: return base
    W, H = base.size
    pad = int(W * padding_ratio)
    logo_w = int(W * logo_width_ratio)
    aspect = logo.size[1] / logo.size[0]
    logo_h = int(logo_w * aspect)
    logo_resized = logo.resize((logo_w, logo_h), Image.LANCZOS)
    base.alpha_composite(logo_resized, (W - logo_w - pad, pad))
    return base

def load_favorites() -> list:
    try:
        if os.path.exists(FAV_FILE):
            with open(FAV_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []

def save_favorites(favs: list) -> None:
    try:
        with open(FAV_FILE, "w") as f: json.dump(favs, f)
    except Exception:
        pass

def _load_pred_history() -> dict:
    try:
        if os.path.exists(PRED_HISTORY_FILE):
            with open(PRED_HISTORY_FILE, "r") as f: return json.load(f)
    except Exception:
        pass
    return {}

def _save_pred_history(data: dict) -> None:
    try:
        with open(PRED_HISTORY_FILE, "w") as f: json.dump(data, f)
    except Exception:
        pass

def record_prediction(player_id: int, player_name: str, pred_date: Optional[str],
                      engine: str, preds: dict) -> None:
    if not preds: return
    db = _load_pred_history()
    key = str(player_id)
    entries = db.get(key, [])
    if pred_date: entries = [e for e in entries if e.get("pred_date") != pred_date]
    entry = {"player_id": player_id, "player_name": player_name, "pred_date": pred_date,
             "engine": engine, "preds": {k: float(v) for k,v in preds.items() if k in PREDICT_COLS}}
    entries.append(entry)
    entries = sorted(entries, key=lambda x: (x.get("pred_date") or ""), reverse=True)[:30]
    db[key] = entries
    _save_pred_history(db)

def get_player_history(player_id: int) -> list:
    return _load_pred_history().get(str(player_id), [])

# ------------------ Cached fetchers -------------------
@st.cache_data(ttl=6*3600)
def get_active_players_fast():
    try:
        return players.get_active_players()
    except Exception as e:
        st.error(f"Error fetching active players: {e}")
        return []

@st.cache_data(ttl=12*3600)
def get_all_teams():
    try:
        return teams.get_teams()
    except Exception as e:
        st.error(f"Error fetching teams: {e}")
        return []

def get_frames_with_retry(endpoint_cls, label: str, **kwargs):
    last_err = None
    for attempt in range(NBA_RETRIES):
        try:
            ep = endpoint_cls(timeout=NBA_TIMEOUT, **kwargs)
            return ep.get_data_frames()
        except Exception as e:
            last_err = e
            time.sleep(NBA_BACKOFF_BASE ** attempt)
    st.error(f"{label}: {last_err}")     # <- fixed f-string
    return []

def get_df_with_retry(endpoint_cls, label: str, frame_idx: int = 0, **kwargs) -> pd.DataFrame:
    frames = get_frames_with_retry(endpoint_cls, label, **kwargs)
    if not frames or frame_idx >= len(frames):
        return pd.DataFrame()
    df = frames[frame_idx]
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_player(player_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fast path:
      - Try disk cache for logs first.
      - Fetch career summary once.
      - In FAST_MODE, restrict to last N seasons for UI.
      - Fetch season logs in parallel with short timeouts.
      - Save combined logs to disk cache and return.
    """
    cached_logs = read_logs_cache(player_id)
    try:
        time.sleep(API_SLEEP)
        career_stats = get_df_with_retry(playercareerstats.PlayerCareerStats,
                                         "Career stats timeout", player_id=player_id)
    except Exception as e:
        st.error(f"Career stats failed: {e}")
        career_stats = pd.DataFrame()

    if cached_logs is not None and not career_stats.empty:
        logs = cached_logs.copy()
        if 'GAME_DATE' in logs.columns:
            logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
            logs = logs.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
        return career_stats, logs

    seasons = career_stats['SEASON_ID'].tolist() if not career_stats.empty else []
    if not seasons:
        return career_stats, pd.DataFrame()

    if FAST_MODE:
        seasons = seasons[-UI_SEASON_LIMIT:]

    def fetch_one(season_id: str) -> pd.DataFrame:
        try:
            time.sleep(API_SLEEP)
            df = get_df_with_retry(
                playergamelogs.PlayerGameLogs,
                f"Game logs timeout ({season_id})",
                player_id_nullable=player_id,
                season_nullable=season_id
            )
            return df if df is not None and not df.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    logs_list = []
    with ThreadPoolExecutor(max_workers=min(4, len(seasons))) as ex:
        futures = {ex.submit(fetch_one, s): s for s in seasons}
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                logs_list.append(df)

    logs = pd.concat(logs_list, ignore_index=True) if logs_list else pd.DataFrame()
    if 'GAME_DATE' in logs.columns:
        logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
        logs = logs.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)

    write_logs_cache(player_id, logs)
    return career_stats, logs

@st.cache_data(ttl=600)
def next_game_for_team(team_abbr: str, lookahead_days: int = 10):
    if not team_abbr:
        return None
    tmap = {t['abbreviation']: t['id'] for t in get_all_teams()}
    team_id = tmap.get(team_abbr)
    if team_id is None: return None
    today = dt.date.today()
    for d in range(lookahead_days):
        day = today + dt.timedelta(days=d)
        try:
            time.sleep(API_SLEEP)
            frames = get_frames_with_retry(scoreboardv2.ScoreboardV2,
                                           f"Scoreboard timeout ({day})",
                                           game_date=day.strftime("%m/%d/%Y"))
            gh = None
            for f in frames:
                if isinstance(f, pd.DataFrame) and {'HOME_TEAM_ID','VISITOR_TEAM_ID'}.issubset(f.columns):
                    gh = f; break
            if gh is None or gh.empty: continue
            for _, row in gh.iterrows():
                home_id = int(row.get('HOME_TEAM_ID', -1))
                away_id = int(row.get('VISITOR_TEAM_ID', -1))
                if team_id in (home_id, away_id):
                    opp_id  = away_id if team_id == home_id else home_id
                    opp_abbr = next((t['abbreviation'] for t in get_all_teams() if t['id']==opp_id), 'TBD')
                    home_flag = (team_id == home_id)
                    return {'date': day.strftime("%Y-%m-%d"), 'opp_abbr': opp_abbr, 'home': home_flag}
        except Exception:
            continue
    return None

# ------------------ Home page data --------------------
@st.cache_data(ttl=900)
def get_last_night_results():
    yday = dt.date.today() - dt.timedelta(days=1)
    df = get_df_with_retry(scoreboardv2.ScoreboardV2, "Scoreboard last night", game_date=yday.strftime("%m/%d/%Y"))
    # Try to find basic lines if standard frame missing
    if df.empty:
        frames = get_frames_with_retry(scoreboardv2.ScoreboardV2, "Scoreboard last night (frames)", game_date=yday.strftime("%m/%d/%Y"))
        if frames:
            for f in frames:
                if isinstance(f, pd.DataFrame) and {'GAME_DATE_EST','HOME_TEAM_ID','VISITOR_TEAM_ID'}.issubset(f.columns):
                    df = f; break
    if df.empty:
        return pd.DataFrame()
    teams_map = {t['id']: t['abbreviation'] for t in get_all_teams()}
    cols = {}
    for c in ['HOME_TEAM_ID','VISITOR_TEAM_ID','PTS_HOME','PTS_VISITOR','GAME_STATUS_TEXT']:
        if c in df.columns: cols[c] = df[c]
    if not cols: return pd.DataFrame()
    out = pd.DataFrame(cols)
    out['HOME'] = out['HOME_TEAM_ID'].map(teams_map)
    out['AWAY'] = out['VISITOR_TEAM_ID'].map(teams_map)
    if 'PTS_HOME' not in out.columns: out['PTS_HOME'] = np.nan
    if 'PTS_VISITOR' not in out.columns: out['PTS_VISITOR'] = np.nan
    return out[['AWAY','PTS_VISITOR','HOME','PTS_HOME','GAME_STATUS_TEXT']]

def _nba_season_str(today=None):
    # Returns season string like "2024-25"
    d = today or dt.date.today()
    start_year = d.year if d.month >= 10 else d.year - 1
    return f"{start_year}-{str((start_year+1)%100).zfill(2)}"

@st.cache_data(ttl=1800)
def get_standings_by_conf():
    """
    Robust standings using LeagueDashTeamStats with minimal, widely-supported args.
    Splits East/West and sorts by win% descending.
    Falls back cleanly if columns differ across nba_api versions.
    """
    season = _nba_season_str()
    try:
        # Minimal args only (avoid 'measure_type_*' which varies by version)
        df = get_df_with_retry(
            leaguedashteamstats.LeagueDashTeamStats,
            "Team stats standings",
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="Totals",
        )
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Conference column can vary
        conf_col = "TEAM_CONFERENCE" if "TEAM_CONFERENCE" in df.columns else ("CONF" if "CONF" in df.columns else None)

        cols = [c for c in ["TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION","W","L","W_PCT"] if c in df.columns]
        out = df[cols + ([conf_col] if conf_col else [])].copy()

        # If no conference column, infer via a static east set
        if not conf_col:
            EAST = {"ATL","BOS","BKN","CHA","CHI","CLE","DET","IND","MIA","MIL","NYK","ORL","PHI","TOR","WAS"}
            out["CONF_IMPUTED"] = out["TEAM_ABBREVIATION"].map(lambda a: "EAST" if a in EAST else "WEST")
            conf_col = "CONF_IMPUTED"

        # Sort by win%
        if "W_PCT" in out.columns:
            out = out.sort_values("W_PCT", ascending=False)
        elif "W" in out.columns:
            out = out.sort_values("W", ascending=False)

        east = out[out[conf_col].str.upper().eq("EAST")] if conf_col in out.columns else out.iloc[:0]
        west = out[out[conf_col].str.upper().eq("WEST")] if conf_col in out.columns else out.iloc[:0]

        def tidy(x):
            view = x.copy()
            if "W_PCT" in view.columns:
                view["WIN%"] = (view["W_PCT"] * 100).round(1)
            if "TEAM_ABBREVIATION" in view.columns:
                view = view.rename(columns={"TEAM_ABBREVIATION": "TEAM"})
            keep = [c for c in ["TEAM","W","L","WIN%"] if c in view.columns]
            return view[keep].reset_index(drop=True)

        return tidy(east), tidy(west)

    except Exception as e:
        st.error(f"Standings fetch failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=900)
def get_league_leaders():
    leaders = {}
    stat_map = {'PTS':'PTS','REB':'REB','AST':'AST','FG3M':'FG3M'}
    for label, col in stat_map.items():
        try:
            df = get_df_with_retry(leagueleaders.LeagueLeaders, f"Leaders {label}",
                                   stat_category_abbreviation=label, per_mode48="PerGame")
            if df.empty: continue
            need = [c for c in ['PLAYER','PLAYER_ID','TEAM','PTS','REB','AST','FG3M'] if c in df.columns]
            d = df[need].head(10).copy()
            leaders[label] = d
        except Exception:
            continue
    return leaders

# ------------------ ML utils --------------------------
@st.cache_data(ttl=3600)
def load_models():
    models = {}
    if not SKLEARN_OK: return models
    for tgt, path in MODEL_FILES.items():
        try:
            if os.path.exists(path):
                models[tgt] = joblib.load(path)
        except Exception:
            continue
    return models

def build_features_for_training(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy().sort_values('GAME_DATE').reset_index(drop=True)
    d['IS_HOME'] = d['MATCHUP'].astype(str).str.contains(" vs ", case=False, regex=False).astype(int)
    d['DAYS_REST'] = d['GAME_DATE'].diff().dt.days
    d['DAYS_REST'] = d['DAYS_REST'].fillna(3).clip(0, 7)
    for k in [5, 10, 20]:
        for c in STATS_COLS:
            d[f'{c}_r{k}'] = d[c].rolling(k, min_periods=1).mean().shift(1)
    if 'SEASON_YEAR' in d.columns:
        for c in STATS_COLS:
            d[f'{c}_season_mean'] = d.groupby('SEASON_YEAR')[c].expanding().mean().shift(1).reset_index(level=0, drop=True)
    else:
        for c in STATS_COLS:
            d[f'{c}_season_mean'] = d[c].expanding().mean().shift(1)
    d = d.dropna().reset_index(drop=True)
    return d

def build_features_for_inference(logs_df: pd.DataFrame, next_game_date, is_home_next: int) -> Optional[pd.DataFrame]:
    if logs_df is None or logs_df.empty: return None
    df = logs_df.copy()
    if 'GAME_DATE' not in df.columns: return None
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
    last_date = pd.to_datetime(df['GAME_DATE'].iloc[-1]).date()
    if isinstance(next_game_date, str):
        try: next_game_date = dt.datetime.strptime(next_game_date, "%Y-%m-%d").date()
        except Exception: next_game_date = None
    days_rest = 3 if next_game_date is None else max(0, min(7, (next_game_date - last_date).days))
    feat = {}
    for k in [5, 10, 20]:
        for c in STATS_COLS:
            s = df[c].rolling(k, min_periods=1).mean().shift(1)
            val = s.iloc[-1] if pd.notna(s.iloc[-1]) else (df[c].iloc[:-1].mean() if len(df) > 1 else df[c].iloc[-1])
            feat[f'{c}_r{k}'] = float(val)
    if 'SEASON_YEAR' in df.columns:
        for c in STATS_COLS:
            smean = df.groupby('SEASON_YEAR')[c].expanding().mean().shift(1).reset_index(level=0, drop=True)
            val = smean.iloc[-1] if pd.notna(smean.iloc[-1]) else df[c].expanding().mean().shift(1).iloc[-1]
            if pd.isna(val):
                val = df[c].iloc[:-1].mean() if len(df) > 1 else df[c].iloc[-1]
            feat[f'{c}_season_mean'] = float(val)
    else:
        for c in STATS_COLS:
            smean = df[c].expanding().mean().shift(1)
            val = smean.iloc[-1] if pd.notna(smean.iloc[-1]) else (df[c].iloc[:-1].mean() if len(df) > 1 else df[c].iloc[-1])
            feat[f'{c}_season_mean'] = float(val)
    feat['IS_HOME']   = int(is_home_next)
    feat['DAYS_REST'] = int(days_rest)
    return pd.DataFrame([feat])

def predict_next_ml(logs_df: pd.DataFrame, next_game_date, is_home_next: int, models: dict) -> Optional[dict]:
    if not models: return None
    feats_df = build_features_for_inference(logs_df, next_game_date, is_home_next)
    if feats_df is None or feats_df.empty: return None
    preds = {}
    for tgt, model in models.items():
        feat_cols = FEAT_TEMPLATE(tgt)
        if not all(c in feats_df.columns for c in feat_cols): continue
        try:
            val = float(model.predict(feats_df[feat_cols])[0])
            preds[tgt] = round(val, 2)
        except Exception:
            continue
    return preds if preds else None

def train_models_core() -> bool:
    if not SKLEARN_OK: return False
    os.makedirs(MODEL_DIR, exist_ok=True)
    act = get_active_players_fast()
    rows = []
    for p in act:
        pid = p.get('id')
        try:
            cstats = get_df_with_retry(playercareerstats.PlayerCareerStats, "Career stats timeout (trainer)", player_id=pid)
            seasons = cstats['SEASON_ID'].tolist() if not cstats.empty else []
            logs_list = []
            for s in seasons:
                try:
                    time.sleep(API_SLEEP)
                    df = get_df_with_retry(playergamelogs.PlayerGameLogs, f"Game logs timeout (trainer {s})",
                                           player_id_nullable=pid, season_nullable=s)
                    if df is not None and not df.empty:
                        logs_list.append(df)
                except Exception:
                    pass
            if not logs_list: continue
            logs = pd.concat(logs_list, ignore_index=True)
            if 'GAME_DATE' not in logs.columns: continue
            logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
            logs = logs.sort_values('GAME_DATE').reset_index(drop=True)
            feats = build_features_for_training(logs)
            if feats.empty: continue
            rows.append(feats)
        except Exception:
            continue
    if not rows: return False
    data = pd.concat(rows, ignore_index=True)
    for target in PREDICT_COLS:
        feat_cols = FEAT_TEMPLATE(target)
        if not all(c in data.columns for c in feat_cols): continue
        X = data[feat_cols]; y = data[target]
        if len(X) < 40: continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = Ridge(alpha=1.0, random_state=42).fit(X_train, y_train)
        joblib.dump(model, MODEL_FILES[target])
    return True

def ensure_background_training():
    if not SKLEARN_OK or st.session_state.get("_bg_training_running"):
        return
    st.session_state["_bg_training_running"] = True
    st.session_state["_bg_training_started_at"] = dt.datetime.now().strftime("%H:%M:%S")
    def _runner():
        try:
            train_models_core()
        finally:
            st.session_state["_bg_training_running"] = False
    threading.Thread(target=_runner, daemon=True).start()

def train_player_models_in_memory(logs_df: pd.DataFrame) -> dict:
    models = {}
    if not SKLEARN_OK or logs_df is None or logs_df.empty: return models
    feats = build_features_for_training(logs_df)
    if feats.empty or len(feats) < 25: return models
    for target in PREDICT_COLS:
        feat_cols = FEAT_TEMPLATE(target)
        if not all(c in feats.columns for c in feat_cols): continue
        X = feats[feat_cols]; y = feats[target]
        if len(X) < 25: continue
        model = Ridge(alpha=1.0, random_state=42).fit(X, y)
        models[target] = model
    return models

# ------------------ Sidebar (clean) -------------------
with st.sidebar:
    st.subheader("Select Player")
    active = get_active_players_fast()
    name_to_id = {p['full_name']: p['id'] for p in active}
    player_names = sorted(name_to_id.keys())

    # Read query param ?pid= to auto-select from deep-link
    qp = st.query_params
    default_index = None
    if 'pid' in qp:
        try:
            pid = int(qp['pid'])
            pname = next(n for n, i in name_to_id.items() if i == pid)
            default_index = player_names.index(pname)
            st.session_state['selected_player_id'] = pid
        except Exception:
            default_index = None
    elif 'selected_player_id' in st.session_state and st.session_state['selected_player_id'] is not None:
        try:
            pid = st.session_state['selected_player_id']
            pname = next(n for n, i in name_to_id.items() if i == pid)
            default_index = player_names.index(pname)
        except Exception:
            default_index = None

    selection = st.selectbox("Search player", player_names,
                             index=default_index if default_index is not None else None,
                             placeholder="Type a player's name…")

    player_id = None
    player_name = None
    if selection:
        player_id = name_to_id[selection]
        player_name = selection
        st.session_state['selected_player_id'] = player_id
        st.query_params.update({"pid": str(player_id)})

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("⭐ Favorites")

    if "favorites" not in st.session_state:
        st.session_state.favorites = load_favorites()

    if player_name and player_id and st.button(f"➕ Add {player_name}", use_container_width=True):
        if not any(f.get("id")==player_id for f in st.session_state.favorites):
            st.session_state.favorites.append({"name": player_name, "id": player_id})
            save_favorites(st.session_state.favorites)
            st.success(f"Added {player_name} to favorites.")

    if st.session_state.favorites:
        for idx, fav in enumerate(list(st.session_state.favorites)):
            nm = fav.get("name", "(unknown)"); pid = fav.get("id")
            colN, colX = st.columns([0.8, 0.2])
            with colN:
                if st.button(nm, key=f"fav_open_{idx}_{pid}", use_container_width=True):
                    st.session_state['selected_player_id'] = pid
                    st.query_params.update({"pid": str(pid)})
                    st.rerun()
            with colX:
                if st.button("×", key=f"fav_del_{idx}_{pid}"):
                    st.session_state.favorites.pop(idx)
                    save_favorites(st.session_state.favorites)
                    st.rerun()
    else:
        st.caption("No favorites yet.")

# ------------------ Header card -----------------------
st.markdown(f"""
<div class="card" style="margin-bottom: 12px;">
  <div style="display:flex; align-items:center; justify-content: space-between;">
    <div>
      <h1 style="margin:0;">Hot Shot Props — NBA Player Analytics</h1>
      <div class="badge">ML: {"Enabled" if SKLEARN_OK else "Disabled"} • Models dir: {MODEL_DIR}</div>
    </div>
    <div style="text-align:right; font-size:.9rem; color:#fda4a4;">Black/Red UI • Clean sidebar • Fast load</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------ HOME PAGE (no player selected) ----
if not player_id:
    st.info("Pick a player from the sidebar to load their dashboard.")
    st.markdown("### Last Night’s Results")
    ln = get_last_night_results()
    if ln.empty:
        st.caption("No results available (scoreboard may not have returned data).")
    else:
        st.dataframe(ln, use_container_width=True)

    st.markdown("### Standings by Conference (Best → Worst)")
    east, west = get_standings_by_conf()
    colE, colW = st.columns(2)
    with colE:
        st.subheader("Eastern Conference")
        if not east.empty:
            st.dataframe(east, use_container_width=True)
        else:
            st.caption("Standings unavailable.")
    with colW:
        st.subheader("Western Conference")
        if not west.empty:
            st.dataframe(west, use_container_width=True)
        else:
            st.caption("Standings unavailable.")

    st.markdown("### League Leaders (Per Game)")
    leaders = get_league_leaders()
    if not leaders:
        st.caption("Leaders data unavailable.")
    else:
        act_map = {p['full_name']: p['id'] for p in get_active_players_fast()}
        for stat, df in leaders.items():
            st.markdown(f"#### {stat}")
            for _, row in df.iterrows():
                pname = row.get('PLAYER')
                pid = int(row.get('PLAYER_ID', 0))
                if not pid and pname in act_map: pid = act_map[pname]
                url = f"?pid={pid}" if pid else "#"
                val = row.get(stat, None)
                val_txt = f"{float(val):.2f}" if isinstance(val,(int,float,np.floating)) else "—"
                st.markdown(f"- **[{pname}]({url})** — {row.get('TEAM','')} • {stat}: **{val_txt}**")
    st.stop()

# ------------------ Player page -----------------------
career_df, logs_df = fetch_player(player_id)
if career_df.empty:
    st.warning("No career data available for this player.")
    st.stop()

# Start background global ML after first successful load
if SKLEARN_OK:
    ensure_background_training()
    if st.session_state.get("_bg_training_running"):
        st.caption(f"🟥 Training global ML in background… (started {st.session_state.get('_bg_training_started_at','now')})")

# Resolve team + season
team_abbr = career_df['TEAM_ABBREVIATION'].iloc[-1] if 'TEAM_ABBREVIATION' in career_df.columns else "TBD"
latest_season = str(career_df['SEASON_ID'].dropna().iloc[-1]) if 'SEASON_ID' in career_df.columns else "N/A"

# Last + next game
last_game_info = "N/A"; last_game_stats = None
if logs_df is not None and not logs_df.empty:
    lg_df = logs_df.head(1).copy()
    lg_cols = safe_cols(lg_df, STATS_COLS)
    if lg_cols: last_game_stats = lg_df[lg_cols].iloc[0].to_dict()
    lg_row  = lg_df.iloc[0]
    lg_date = pd.to_datetime(lg_row['GAME_DATE']).strftime("%Y-%m-%d") if 'GAME_DATE' in lg_row else "—"
    lg_opp  = extract_opp_from_matchup(lg_row.get('MATCHUP', '')) or "TBD"
    last_game_info = f"{lg_date} vs {lg_opp}"

ng = next_game_for_team(team_abbr) if team_abbr else None
if ng:
    icon = "🏠" if ng.get('home') else "✈️"
    next_game_info = f"{icon} {ng['date']} vs {ng['opp_abbr']}"
    is_home_next = 1 if ng.get('home') else 0
else:
    next_game_info = "TBD"; is_home_next = 0

# Header metrics
c1, c2, c3, c4 = st.columns([1.3, 0.8, 1.2, 1.2])
with c1: st.metric("Player", player_name)
with c2: st.metric("Team", team_abbr)
with c3: st.metric("Most Recent Game", last_game_info)
with c4: st.metric("Next Game", next_game_info)

# Headshot + logo + bar charts
col_img, col_trend = st.columns([0.30, 0.70])
with col_img:
    head = cdn_headshot(player_id, "1040x760") or cdn_headshot(player_id, "260x190")
    logo = None
    try:
        tid = next((t['id'] for t in get_all_teams() if t['abbreviation']==team_abbr), None)
        if tid: logo = cdn_team_logo(tid)
    except Exception:
        pass
    if head:
        composed = overlay_logo_top_right(head, logo, padding_ratio=0.035, logo_width_ratio=0.22)
        st.image(composed, use_container_width=True, caption=f"{player_name} — media day headshot")
    else:
        st.info("Headshot not available.")

with col_trend:
    st.markdown("#### Last 10 Games — Bar Charts")
    if logs_df is not None and not logs_df.empty:
        N = min(10, len(logs_df))
        view = logs_df.head(N).copy().iloc[::-1]
        view['GAME_DATE'] = pd.to_datetime(view['GAME_DATE'])
        view['Game'] = view['GAME_DATE'].dt.strftime('%m-%d')
        core_stats = ['PTS','REB','AST','FG3M']
        available_stats = [c for c in STATS_COLS if c in view.columns]
        show_all = st.toggle("Show all categories", value=False)
        chosen = available_stats if show_all else [c for c in core_stats if c in available_stats]
        if not chosen:
            st.info("No stats available to chart.")
        else:
            for i, stat in enumerate(chosen):
                chart_df = view[['Game', stat]].copy()
                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X('Game:N', title=''),
                    y=alt.Y(f'{stat}:Q', title=stat),
                    tooltip=[alt.Tooltip('Game:N', title='Game'), alt.Tooltip(f'{stat}:Q', title=stat)],
                ).properties(height=140)
                if i % 2 == 0:
                    colA, colB = st.columns(2)
                    with colA: st.altair_chart(chart, use_container_width=True)
                else:
                    with colB: st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No recent games to chart.")

# Metric row helper
def metric_row(title: str, data: dict | pd.Series | pd.DataFrame, fallback: str = "—"):
    st.markdown(f"#### {title}")
    if data is None: data = {}
    if isinstance(data, pd.DataFrame):
        row = data.iloc[0].to_dict() if not data.empty else {}
    elif isinstance(data, pd.Series):
        row = data.to_dict()
    else:
        row = dict(data)
    def fmt(v): return f"{float(v):.2f}" if isinstance(v, (int,float,np.floating)) and pd.notna(v) else fallback
    cols = st.columns(5)
    with cols[0]: st.metric("PTS", fmt(row.get('PTS')))
    with cols[1]: st.metric("REB", fmt(row.get('REB')))
    with cols[2]: st.metric("AST", fmt(row.get('AST')))
    with cols[3]: st.metric("3PM", fmt(row.get('FG3M')))
    with cols[4]: st.metric("MIN", fmt(row.get('MIN')))

# Row 1: Season averages (per game)
season_row = None
if not career_df.empty:
    cur = career_df.iloc[-1]
    gp  = cur.get('GP', 0)
    if gp and gp > 0:
        season_row = {c: (cur.get(c, 0)/gp) for c in STATS_COLS}
metric_row("Current Season Averages", season_row)

# Row 2: Last game
metric_row("Last Game Stats", last_game_stats)

# Row 3: Last 5 averages
def recent_averages(logs: pd.DataFrame) -> dict:
    out = {}
    if logs is None or logs.empty: return out
    df = logs.copy()
    cols = safe_cols(df, STATS_COLS)
    if len(df) >= 5 and cols:
        sub = df.head(5)
        out['Last 5 Avg'] = sub[cols].mean().to_frame(name='Last 5 Avg').T
    return out
ra = recent_averages(logs_df)
last5 = ra['Last 5 Avg'].iloc[0].to_dict() if 'Last 5 Avg' in ra and not ra['Last 5 Avg'].empty else None
metric_row("Last 5 Games Averages", last5)

# Row 4: Predictions (Global ML -> Player ML -> Fallback)
@st.cache_data(ttl=3600)
def load_models_cached():
    return load_models()

engine_mode = "fallback"
ml_models = load_models_cached() if SKLEARN_OK else {}

def predict_next_fallback(logs: pd.DataFrame, season_label: str) -> Optional[dict]:
    if logs is None or logs.empty: return None
    df = logs.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
    cols = safe_cols(df, STATS_COLS)
    if not cols: return None
    season_avg = pd.Series(dtype=float)
    if 'SEASON_YEAR' in df.columns:
        cur = df[df['SEASON_YEAR'] == season_label]
        if not cur.empty: season_avg = cur[cols].mean()
    if season_avg.empty: season_avg = df[cols].mean()
    feats = {}
    for k in [5,10,20]:
        for c in cols:
            s = df[c].rolling(k, min_periods=1).mean().shift(1)
            feats[f'{c}_r{k}'] = s.iloc[-1] if pd.notna(s.iloc[-1]) else season_avg.get(c, 0.0)
    preds = {}
    for c in PREDICT_COLS:
        r5, r10, r20 = feats.get(f'{c}_r5', np.nan), feats.get(f'{c}_r10', np.nan), feats.get(f'{c}_r20', np.nan)
        s = season_avg.get(c, 0.0) if pd.notna(season_avg.get(c, np.nan)) else 0.0
        v5, v10, v20 = (r5 if pd.notna(r5) else s), (r10 if pd.notna(r10) else s), (r20 if pd.notna(r20) else s)
        preds[c] = round(float(0.4*v5 + 0.3*v10 + 0.2*v20 + 0.1*s), 2)
    return preds

ml_preds  = predict_next_ml(logs_df, ng.get('date') if ng else None, 1 if (ng and ng.get('home')) else 0, ml_models) if ml_models else None

if ml_preds:
    engine_mode = "global_ml"; metric_row("Predicted Next Game (ML)", ml_preds)
else:
    player_models = train_player_models_in_memory(logs_df) if SKLEARN_OK else {}
    player_ml_preds = predict_next_ml(logs_df, ng.get('date') if ng else None, 1 if (ng and ng.get('home')) else 0, player_models) if player_models else None
    if player_ml_preds:
        engine_mode = "player_ml"; ml_preds = player_ml_preds; metric_row("Predicted Next Game (ML)", ml_preds)
    else:
        preds = predict_next_fallback(logs_df, latest_season)
        engine_mode = "fallback"; ml_preds = preds; metric_row("Predicted Next Game (Model)", preds)

label = {
    "global_ml": '<span class="tag ok">Using Global ML</span>',
    "player_ml": '<span class="tag ok">Using Player ML (ad-hoc)</span>',
    "fallback":  '<span class="tag dim">Using weighted fallback model</span>' if SKLEARN_OK else
                 '<span class="tag warn">Fallback — install scikit-learn & joblib</span>',
}[engine_mode]
st.markdown(label, unsafe_allow_html=True)

# Save prediction (if date known)
pred_date_str = ng['date'] if (ng and isinstance(ng.get('date'), str)) else None
if ml_preds:
    record_prediction(player_id, player_name, pred_date_str, engine_mode, ml_preds)

# -------- Last Predictions vs Results --------
st.markdown("### Last Predictions vs Results")
def find_actual_row_by_date(logs: pd.DataFrame, iso_date: str):
    if logs is None or logs.empty or not iso_date: return None
    try: d = dt.datetime.strptime(iso_date, "%Y-%m-%d").date()
    except Exception: return None
    cand = logs.copy()
    cand['GD'] = pd.to_datetime(cand['GAME_DATE']).dt.date
    matches = cand[cand['GD'] == d]
    return matches.iloc[0] if not matches.empty else None

history = get_player_history(player_id)
if not history:
    st.caption("No stored predictions yet. A prediction is saved when a next game date is known.")
else:
    shown = 0
    for entry in sorted(history, key=lambda x: (x.get("pred_date") or ""), reverse=True):
        if shown >= 3: break
        pdate = entry.get("pred_date")
        preds_h = entry.get("preds", {})
        engine = entry.get("engine", "unknown")
        actual_row = find_actual_row_by_date(logs_df, pdate) if pdate else None
        hdr = f"**Predicted for {pdate}**" if pdate else "**Prediction (no date recorded)**"
        eng = {"global_ml":"Global ML","player_ml":"Player ML","fallback":"Fallback","blended_ml":"Blended ML"}.get(engine,"Model")
        st.markdown(f"{hdr} — _{eng}_")
        cols = st.columns(4)
        for i, k in enumerate(['PTS','REB','AST','FG3M']):
            with cols[i]:
                pred_val = preds_h.get(k, None)
                if actual_row is None:
                    st.metric(f"{k} • Pending", value=str(pred_val) if pred_val is not None else "—", delta="Game not played")
                    st.caption('<span class="pending">Awaiting result</span>', unsafe_allow_html=True)
                else:
                    act = actual_row.get(k, None)
                    hit = (act is not None and pred_val is not None and float(act) >= float(pred_val))
                    status = "✅ Hit" if hit else "❌ Miss"
                    delta = f"Pred {pred_val:.2f} → Act {act:.2f}" if (isinstance(pred_val,(int,float)) and isinstance(act,(int,float))) else "—"
                    st.metric(f"{k} • {status}", value=f"{act:.2f}" if isinstance(act,(int,float)) else "—", delta=delta)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        shown += 1

# ------------------ Download PNG snapshot -------------
def compose_share_card(player_name: str, team_abbr: str,
                       headshot_img: Optional[Image.Image],
                       team_logo_img: Optional[Image.Image],
                       season_row: dict|None,
                       last_game_row: dict|None,
                       last5_row: dict|None,
                       pred_row: dict|None) -> bytes:
    W, H = 1200, 1400
    bg = Image.new("RGBA", (W, H), (10,10,10,255))
    draw = ImageDraw.Draw(bg)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
        font_h2    = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        font_body  = ImageFont.truetype("DejaVuSans.ttf", 28)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font_title = font_h2 = font_body = font_small = None
    title = f"Hot Shot Props — {player_name} ({team_abbr})"
    draw.text((36, 30), title, fill=(255,180,180,255), font=font_title)

    x_img, y_img = 36, 90
    if headshot_img:
        h = headshot_img.copy().convert("RGBA"); h.thumbnail((520, 380), Image.LANCZOS)
        bg.alpha_composite(h, (x_img, y_img))
        if team_logo_img:
            logo = team_logo_img.copy().convert("RGBA")
            lw = int(h.width * 0.22); lh = int(logo.height * (lw / logo.width))
            logo = logo.resize((lw, lh), Image.LANCZOS)
            bg.alpha_composite(logo, (x_img + h.width - lw - 12, y_img + 12))

    def block(label, row, x, y):
        draw.rounded_rectangle((x, y, x+520, y+160), radius=16, outline=(35,35,35,255), width=2, fill=(18,18,18,255))
        draw.text((x+16,y+12), label, fill=(250,164,164,255), font=font_h2)
        labels = [("PTS","PTS"),("REB","REB"),("AST","AST"),("3PM","FG3M"),("MIN","MIN")]
        x0 = x+16; step = 100
        for i,(lab,key) in enumerate(labels):
            val = row.get(key,"—") if isinstance(row, dict) else "—"
            val = f"{val:.2f}" if isinstance(val,(int,float,np.floating)) and pd.notna(val) else ("N/A" if val is None else str(val))
            draw.text((x0 + i*step, y+64), lab, fill=(230,230,230,255), font=font_small)
            draw.text((x0 + i*step, y+96), val, fill=(255,228,230,255), font=font_body)

    y0 = 90
    block("Current Season Averages", season_row or {},  644, y0)
    block("Last Game Stats",        last_game_row or {}, 36,  y0+400)
    block("Last 5 Games Averages",  last5_row or {},      644, y0+400)
    block("Predicted Next Game",    pred_row or {},       36,  y0+800)
    draw.text((36, H-60), "Generated with Hot Shot Props • nba_api • Streamlit", fill=(210,210,210,255), font=font_small)
    buf = BytesIO(); bg.convert("RGB").save(buf, format="PNG", quality=95); buf.seek(0)
    return buf.getvalue()

st.markdown("## Download a snapshot")
def _row_or_none(obj):
    if obj is None: return {}
    if isinstance(obj, pd.DataFrame) and not obj.empty: return obj.iloc[0].to_dict()
    if isinstance(obj, dict): return obj
    if isinstance(obj, pd.Series): return obj.to_dict()
    return {}

season_row_dict = _row_or_none(season_row)
last_game_row_dict = _row_or_none(last_game_stats)
last5_row_dict = _row_or_none(last5)
pred_row_dict = _row_or_none(ml_preds)

png_bytes = compose_share_card(
    player_name=player_name,
    team_abbr=team_abbr,
    headshot_img=head if 'head' in locals() else None,
    team_logo_img=logo if 'logo' in locals() else None,
    season_row=season_row_dict,
    last_game_row=last_game_row_dict,
    last5_row=last5_row_dict,
    pred_row=pred_row_dict
)

st.download_button(
    "⬇️ Download snapshot (PNG)",
    data=png_bytes,
    file_name=f"{player_name.replace(' ', '_')}_hotshotprops.png",
    mime="image/png",
    use_container_width=True
)
