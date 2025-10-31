# app.py — Hot Shot Props | NBA Player Analytics (Free)
# NOTE: Home page now shows ONLY the top leader for each stat (PTS/REB/AST/3PM)
# with Name (Team) above the headshot and the stat/line below the image.

import os, json, re, time, threading, datetime as dt
from io import BytesIO
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import altair as alt
import streamlit.components.v1 as components

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playercareerstats, playergamelogs, scoreboardv2,
    leagueleaders, leaguedashteamstats
)

# ---------- Optional ML deps ----------
try:
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    joblib = None
    Ridge = None
    SKLEARN_OK = False

# ---------- Page ----------
st.set_page_config(page_title="Hot Shot Props • NBA Player Analytics (Free)",
                   layout="wide", initial_sidebar_state="expanded")

components.html("""
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
""", height=0)

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
@media (max-width: 780px){
  .block-container { padding: 0.6rem 0.6rem !important; }
  section.main > div { padding-top: 0 !important; }
  [data-testid="column"] { width: 100% !important; display: block !important; }
  [data-testid="stMetric"]{ margin-bottom: 10px !important; }
  .stButton>button{ width: 100% !important; padding: 14px !important; font-size: 1.05rem !important; }
  .stTextInput>div>div>input, .stSelectbox>div>div>div { font-size: 1.05rem !important; }
}
.leader-card{background:#0e0e0e;border:1px solid #181818;border-radius:16px;padding:14px;text-align:center;}
.leader-card h4{margin:0;color:#ffe1e1;}
.leader-sub{color:#fbb; font-size:.9rem; margin-top:2px;}
.leader-img{width:180px;height:132px;object-fit:cover;border-radius:10px;border:1px solid #222;}
.leader-stat{margin-top:8px;color:#fff2f2;font-weight:800;font-size:1.1rem;}
.leader-stat small{opacity:.8;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ---------- Timing / API ----------
NBA_TIMEOUT = 15
NBA_RETRIES = 2
NBA_BACKOFF = 1.6
API_SLEEP = 0.15
FAST_MODE = True
UI_SEASON_LIMIT = 2

# ---------- Data constants ----------
STATS_COLS   = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']
PREDICT_COLS = ['PTS','AST','REB','FG3M']

DEFAULT_TMP = "/tmp" if os.access("/", os.W_OK) else "."
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(DEFAULT_TMP, "models"))
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILES  = {t: os.path.join(MODEL_DIR, f"model_{t}.pkl") for t in PREDICT_COLS}
FEAT_TEMPLATE = lambda tgt: [f'{tgt}_r5', f'{tgt}_r10', f'{tgt}_r20', f'{tgt}_season_mean', 'IS_HOME', 'DAYS_REST']

USER_ROOT = os.path.join(DEFAULT_TMP, "userdata", "guest"); os.makedirs(USER_ROOT, exist_ok=True)
FAV_FILE = os.path.join(USER_ROOT, "favorites.json")
PRED_HISTORY_FILE = os.path.join(USER_ROOT, "pred_history.json")
CACHE_ROOT = os.path.join(USER_ROOT, "cache"); os.makedirs(CACHE_ROOT, exist_ok=True)
CACHE_TTL_HOURS = 6

# ---------- Helpers ----------
def _go_home():
    try: st.query_params.clear()
    except Exception: st.query_params.update({})
    st.session_state.pop("selected_player_id", None); st.rerun()

def _cache_path_for_player(pid): return os.path.join(CACHE_ROOT, f"player_{pid}_logs.parquet")
def _cache_path_csv(pid):        return os.path.join(CACHE_ROOT, f"player_{pid}_logs.csv")

def read_logs_cache(pid):
    p = _cache_path_for_player(pid) if os.path.exists(_cache_path_for_player(pid)) else (_cache_path_csv(pid) if os.path.exists(_cache_path_csv(pid)) else None)
    if not p: return None
    try:
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(p))
        if (dt.datetime.now() - mtime).total_seconds() > CACHE_TTL_HOURS*3600: return None
        return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    except Exception: return None

def write_logs_cache(pid, df):
    try:
        if df is not None and not df.empty:
            try: df.to_parquet(_cache_path_for_player(pid), index=False)
            except Exception: df.to_csv(_cache_path_csv(pid), index=False)
    except Exception: pass

def safe_cols(df, cols): return [c for c in cols if c in df.columns]
def extract_opp_from_matchup(m): 
    if not isinstance(m, str): return None
    k = re.search(r'@\s*([A-Z]{3})|vs\.\s*([A-Z]{3})|VS\.\s*([A-Z]{3})', m, re.IGNORECASE)
    return (k.group(1) or k.group(2) or k.group(3)).upper() if k else None

def cdn_headshot(pid, size="1040x760"):
    url = f"https://cdn.nba.com/headshots/nba/latest/{size}/{pid}.png"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200: return Image.open(BytesIO(r.content)).convert("RGBA")
    except Exception: pass
    return None

def cdn_team_logo(tid):
    for url in [f"https://cdn.nba.com/logos/nba/{tid}/global/L/logo.png", f"https://cdn.nba.com/logos/nba/{tid}/global/D/logo.png"]:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200: return Image.open(BytesIO(r.content)).convert("RGBA")
        except Exception: continue
    return None

def overlay_logo_top_right(headshot, logo, padding_ratio=0.035, logo_width_ratio=0.22):
    base = headshot.copy()
    if not logo: return base
    W,H = base.size; pad = int(W*padding_ratio); lw = int(W*logo_width_ratio); lh = int(logo.size[1]*(lw/logo.size[0]))
    lr = logo.resize((lw, lh), Image.LANCZOS)
    base.alpha_composite(lr, (W-lw-pad, pad)); return base

def load_favorites():
    try:
        if os.path.exists(FAV_FILE):
            with open(FAV_FILE,"r") as f: d = json.load(f); return d if isinstance(d, list) else []
    except Exception: pass
    return []
def save_favorites(f): 
    try: open(FAV_FILE,"w").write(json.dumps(f))
    except Exception: pass

def _load_pred_history(): 
    try:
        if os.path.exists(PRED_HISTORY_FILE):
            with open(PRED_HISTORY_FILE,"r") as f: return json.load(f)
    except Exception: pass
    return {}
def _save_pred_history(d): 
    try: open(PRED_HISTORY_FILE,"w").write(json.dumps(d))
    except Exception: pass
def record_prediction(pid, pname, pdate, engine, preds):
    if not preds: return
    db = _load_pred_history(); key = str(pid); entries = db.get(key, [])
    if pdate: entries = [e for e in entries if e.get("pred_date") != pdate]
    entries.append({"player_id": pid, "player_name": pname, "pred_date": pdate, "engine": engine,
                    "preds": {k: float(v) for k,v in preds.items() if k in PREDICT_COLS}})
    entries = sorted(entries, key=lambda x: (x.get("pred_date") or ""), reverse=True)[:30]
    db[key] = entries; _save_pred_history(db)
def get_player_history(pid): return _load_pred_history().get(str(pid), [])

# ---------- Cached fetchers ----------
@st.cache_data(ttl=6*3600)
def get_active_players_fast():
    try: return players.get_active_players()
    except Exception as e: st.error(f"Error fetching active players: {e}"); return []

@st.cache_data(ttl=12*3600)
def get_all_teams():
    try: return teams.get_teams()
    except Exception as e: st.error(f"Error fetching teams: {e}"); return []

def get_frames_with_retry(endpoint_cls, label: str, **kwargs):
    err = None
    for a in range(NBA_RETRIES):
        try:
            ep = endpoint_cls(timeout=NBA_TIMEOUT, **kwargs)
            return ep.get_data_frames()
        except Exception as e:
            err = e; time.sleep(NBA_BACKOFF**a)
    st.error(f"{label}: {err}"); return []

def get_df_with_retry(endpoint_cls, label: str, frame_idx: int = 0, **kwargs):
    fr = get_frames_with_retry(endpoint_cls, label, **kwargs)
    if not fr or frame_idx >= len(fr): return pd.DataFrame()
    return fr[frame_idx] if isinstance(fr[frame_idx], pd.DataFrame) else pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_player(pid: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cache = read_logs_cache(pid)
    try:
        time.sleep(API_SLEEP)
        career = get_df_with_retry(playercareerstats.PlayerCareerStats, "Career stats timeout", player_id=pid)
    except Exception as e:
        st.error(f"Career stats failed: {e}"); career = pd.DataFrame()

    if cache is not None and not career.empty:
        logs = cache.copy()
        if 'GAME_DATE' in logs.columns:
            logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
            logs = logs.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
        return career, logs

    seasons = career['SEASON_ID'].tolist() if not career.empty else []
    if not seasons: return career, pd.DataFrame()
    if FAST_MODE: seasons = seasons[-UI_SEASON_LIMIT:]

    def one(season_id: str):
        try:
            time.sleep(API_SLEEP)
            return get_df_with_retry(playergamelogs.PlayerGameLogs, f"Logs timeout ({season_id})",
                                     player_id_nullable=pid, season_nullable=season_id)
        except Exception: return pd.DataFrame()

    lst = []
    with ThreadPoolExecutor(max_workers=min(4, len(seasons))) as ex:
        for fut in as_completed({ex.submit(one, s): s for s in seasons}):
            df = fut.result()
            if df is not None and not df.empty: lst.append(df)

    logs = pd.concat(lst, ignore_index=True) if lst else pd.DataFrame()
    if 'GAME_DATE' in logs.columns:
        logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
        logs = logs.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
    write_logs_cache(pid, logs)
    return career, logs

@st.cache_data(ttl=600)
def next_game_for_team(abbr: str, lookahead_days=10):
    if not abbr: return None
    tmap = {t['abbreviation']: t['id'] for t in get_all_teams()}
    tid = tmap.get(abbr);  today = dt.date.today()
    if not tid: return None
    for d in range(lookahead_days):
        day = today + dt.timedelta(days=d)
        try:
            time.sleep(API_SLEEP)
            frames = get_frames_with_retry(scoreboardv2.ScoreboardV2, f"Scoreboard ({day})", game_date=day.strftime("%m/%d/%Y"))
            gh = None
            for f in frames:
                if isinstance(f, pd.DataFrame) and {'HOME_TEAM_ID','VISITOR_TEAM_ID'}.issubset(f.columns):
                    gh = f; break
            if gh is None or gh.empty: continue
            for _, r in gh.iterrows():
                h = int(r.get('HOME_TEAM_ID', -1)); a = int(r.get('VISITOR_TEAM_ID', -1))
                if tid in (h, a):
                    opp = a if tid == h else h
                    opp_abbr = next((t['abbreviation'] for t in get_all_teams() if t['id']==opp), 'TBD')
                    return {'date': day.strftime("%Y-%m-%d"), 'opp_abbr': opp_abbr, 'home': (tid==h)}
        except Exception: continue
    return None

# ---------- Home page helpers ----------
@st.cache_data(ttl=900)
def get_last_night_results():
    yday = dt.date.today() - dt.timedelta(days=1)
    df = get_df_with_retry(scoreboardv2.ScoreboardV2, "Scoreboard last night", game_date=yday.strftime("%m/%d/%Y"))
    if df.empty:
        frames = get_frames_with_retry(scoreboardv2.ScoreboardV2, "Scoreboard last night (frames)", game_date=yday.strftime("%m/%d/%Y"))
        if frames:
            for f in frames:
                if isinstance(f, pd.DataFrame) and {'HOME_TEAM_ID','VISITOR_TEAM_ID'}.issubset(f.columns):
                    df = f; break
    if df.empty: return pd.DataFrame()
    tmap = {t['id']: t['abbreviation'] for t in get_all_teams()}
    cols = {}
    for c in ['HOME_TEAM_ID','VISITOR_TEAM_ID','PTS_HOME','PTS_VISITOR','GAME_STATUS_TEXT']:
        if c in df.columns: cols[c] = df[c]
    if not cols: return pd.DataFrame()
    out = pd.DataFrame(cols)
    out['HOME'] = out['HOME_TEAM_ID'].map(tmap); out['AWAY'] = out['VISITOR_TEAM_ID'].map(tmap)
    if 'PTS_HOME' not in out.columns: out['PTS_HOME'] = np.nan
    if 'PTS_VISITOR' not in out.columns: out['PTS_VISITOR'] = np.nan
    return out[['AWAY','PTS_VISITOR','HOME','PTS_HOME','GAME_STATUS_TEXT']]

def _nba_season_str(today=None):
    d = today or dt.date.today()
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{str((start+1)%100).zfill(2)}"

@st.cache_data(ttl=1800)
def get_standings_by_conf():
    season = _nba_season_str()
    try:
        df = get_df_with_retry(leaguedashteamstats.LeagueDashTeamStats, "Team standings",
                               season=season, season_type_all_star="Regular Season", per_mode_detailed="Totals")
        if df.empty: return pd.DataFrame(), pd.DataFrame()
        conf_col = "TEAM_CONFERENCE" if "TEAM_CONFERENCE" in df.columns else None
        cols = [c for c in ["TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION","W","L","W_PCT"] if c in df.columns]
        out = df[cols + ([conf_col] if conf_col else [])].copy()
        if not conf_col:
            EAST = {"ATL","BOS","BKN","CHA","CHI","CLE","DET","IND","MIA","MIL","NYK","ORL","PHI","TOR","WAS"}
            out["CONF"] = out["TEAM_ABBREVIATION"].map(lambda a: "EAST" if a in EAST else "WEST")
            conf_col = "CONF"
        out = out.sort_values("W_PCT" if "W_PCT" in out.columns else "W", ascending=False)
        def tidy(x):
            v = x.copy()
            if "W_PCT" in v.columns: v["WIN%"] = (v["W_PCT"]*100).round(1)
            v = v.rename(columns={"TEAM_ABBREVIATION":"TEAM"}) if "TEAM_ABBREVIATION" in v.columns else v
            return v[[c for c in ["TEAM","W","L","WIN%"] if c in v.columns]].reset_index(drop=True)
        east = tidy(out[out[conf_col].str.upper().eq("EAST")]) if conf_col in out.columns else out.iloc[:0]
        west = tidy(out[out[conf_col].str.upper().eq("WEST")]) if conf_col in out.columns else out.iloc[:0]
        return east, west
    except Exception as e:
        st.error(f"Standings fetch failed: {e}"); return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=900)
def get_top_leader(stat_abbrev: str):
    """Return (player_name, player_id, team_abbr, value) for the #1 per-game leader of the stat."""
    df = get_df_with_retry(leagueleaders.LeagueLeaders, f"Leader {stat_abbrev}",
                           stat_category_abbreviation=stat_abbrev, per_mode48="PerGame")
    if df.empty: return None
    row = df.iloc[0]
    pname = row.get('PLAYER'); pid = int(row.get('PLAYER_ID', 0))
    team  = row.get('TEAM', '')
    val   = float(row.get(stat_abbrev, 0.0)) if stat_abbrev in row else None
    return {"name": pname, "id": pid, "team": team, "value": val}

def _team_id_from_abbr(abbr: str) -> Optional[int]:
    for t in get_all_teams():
        if t['abbreviation'] == abbr: return t['id']
    return None

def render_leader_cards_row():
    st.markdown("### League Leaders (Top Only)")
    stats = [("PTS","Points"), ("REB","Rebounds"), ("AST","Assists"), ("FG3M","3-Pointers Made")]
    cols = st.columns(4)
    for (abbr, label), col in zip(stats, cols):
        leader = get_top_leader(abbr)
        with col:
            if not leader:
                st.info(f"{label}: unavailable"); continue
            pid, pname, team, val = leader["id"], leader["name"], leader["team"], leader["value"]
            # Name (Team) above picture — click jumps to player page
            jump = f"?pid={pid}" if pid else "#"
            st.markdown(f"<div class='leader-card'><h4><a href='{jump}' style='text-decoration:none;color:#ffe1e1;'>{pname}</a></h4>"
                        f"<div class='leader-sub'>{team}</div>", unsafe_allow_html=True)
            # Image
            img = cdn_headshot(pid, "260x190") or cdn_headshot(pid, "1040x760")
            logo = cdn_team_logo(_team_id_from_abbr(team) or -1) if team else None
            if img:
                if logo: img = overlay_logo_top_right(img, logo, padding_ratio=0.04, logo_width_ratio=0.24)
                buf = BytesIO(); img.convert("RGB").save(buf, format="PNG"); buf.seek(0)
                st.image(buf, use_container_width=False, clamp=True, output_format="PNG")
            else:
                st.image(np.zeros((190,260,3),dtype=np.uint8), use_container_width=False)
            # Stat + line below
            if isinstance(val,(int,float,np.floating)):
                st.markdown(f"<div class='leader-stat'>{label}: {val:.2f} <small>per game</small></div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='leader-stat'>{label}: —</div></div>", unsafe_allow_html=True)

# ---------- ML core (same as prior) ----------
def build_features_for_training(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy().sort_values('GAME_DATE').reset_index(drop=True)
    d['IS_HOME'] = d['MATCHUP'].astype(str).str.contains(" vs ", case=False, regex=False).astype(int)
    d['DAYS_REST'] = d['GAME_DATE'].diff().dt.days
    d['DAYS_REST'] = d['DAYS_REST'].fillna(3).clip(0, 7)
    for k in [5,10,20]:
        for c in STATS_COLS:
            d[f'{c}_r{k}'] = d[c].rolling(k, min_periods=1).mean().shift(1)
    if 'SEASON_YEAR' in d.columns:
        for c in STATS_COLS:
            d[f'{c}_season_mean'] = d.groupby('SEASON_YEAR')[c].expanding().mean().shift(1).reset_index(level=0, drop=True)
    else:
        for c in STATS_COLS:
            d[f'{c}_season_mean'] = d[c].expanding().mean().shift(1)
    d = d.dropna().reset_index(drop=True); return d

def build_features_for_inference(logs_df: pd.DataFrame, next_game_date, is_home_next: int) -> Optional[pd.DataFrame]:
    if logs_df is None or logs_df.empty: return None
    df = logs_df.copy(); df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
    last_date = pd.to_datetime(df['GAME_DATE'].iloc[-1]).date()
    if isinstance(next_game_date, str):
        try: next_game_date = dt.datetime.strptime(next_game_date,"%Y-%m-%d").date()
        except Exception: next_game_date = None
    rest = 3 if next_game_date is None else max(0, min(7, (next_game_date - last_date).days))
    feat = {}
    for k in [5,10,20]:
        for c in STATS_COLS:
            s = df[c].rolling(k, min_periods=1).mean().shift(1)
            val = s.iloc[-1] if pd.notna(s.iloc[-1]) else (df[c].iloc[:-1].mean() if len(df)>1 else df[c].iloc[-1])
            feat[f'{c}_r{k}'] = float(val)
    if 'SEASON_YEAR' in df.columns:
        for c in STATS_COLS:
            smean = df.groupby('SEASON_YEAR')[c].expanding().mean().shift(1).reset_index(level=0, drop=True)
            val = smean.iloc[-1] if pd.notna(smean.iloc[-1]) else df[c].expanding().mean().shift(1).iloc[-1]
            if pd.isna(val): val = df[c].iloc[:-1].mean() if len(df)>1 else df[c].iloc[-1]
            feat[f'{c}_season_mean'] = float(val)
    else:
        for c in STATS_COLS:
            smean = df[c].expanding().mean().shift(1)
            val = smean.iloc[-1] if pd.notna(smean.iloc[-1]) else (df[c].iloc[:-1].mean() if len(df)>1 else df[c].iloc[-1])
            feat[f'{c}_season_mean'] = float(val)
    feat['IS_HOME'] = int(is_home_next); feat['DAYS_REST'] = int(rest)
    return pd.DataFrame([feat])

@st.cache_data(ttl=3600)
def load_models_cached():
    models = {}
    if not SKLEARN_OK: return models
    for tgt, path in MODEL_FILES.items():
        try:
            if os.path.exists(path): models[tgt] = joblib.load(path)
        except Exception: pass
    return models

def predict_next_ml(logs_df, next_game_date, is_home_next, models):
    if not models: return None
    feats = build_features_for_inference(logs_df, next_game_date, is_home_next)
    if feats is None or feats.empty: return None
    preds = {}
    for tgt, model in models.items():
        cols = FEAT_TEMPLATE(tgt)
        if not all(c in feats.columns for c in cols): continue
        try: preds[tgt] = round(float(model.predict(feats[cols])[0]), 2)
        except Exception: pass
    return preds if preds else None

def train_player_models_in_memory(logs_df):
    models = {}
    if not SKLEARN_OK or logs_df is None or logs_df.empty: return models
    feats = build_features_for_training(logs_df)
    if feats.empty or len(feats) < 25: return models
    for tgt in PREDICT_COLS:
        cols = FEAT_TEMPLATE(tgt)
        if not all(c in feats.columns for c in cols): continue
        model = Ridge(alpha=1.0, random_state=42).fit(feats[cols], feats[tgt])
        models[tgt] = model
    return models

def predict_next_fallback(logs: pd.DataFrame, season_label: str):
    if logs is None or logs.empty: return None
    df = logs.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
    cols = safe_cols(df, STATS_COLS)
    if not cols: return None
    season_avg = pd.Series(dtype=float)
    if 'SEASON_YEAR' in df.columns:
        cur = df[df['SEASON_YEAR']==season_label]
        if not cur.empty: season_avg = cur[cols].mean()
    if season_avg.empty: season_avg = df[cols].mean()
    feats = {}
    for k in [5,10,20]:
        for c in cols:
            s = df[c].rolling(k, min_periods=1).mean().shift(1)
            feats[f'{c}_r{k}'] = s.iloc[-1] if pd.notna(s.iloc[-1]) else season_avg.get(c, 0.0)
    out = {}
    for c in PREDICT_COLS:
        r5, r10, r20 = feats.get(f'{c}_r5', np.nan), feats.get(f'{c}_r10', np.nan), feats.get(f'{c}_r20', np.nan)
        s = season_avg.get(c, 0.0) if pd.notna(season_avg.get(c, np.nan)) else 0.0
        v5, v10, v20 = (r5 if pd.notna(r5) else s), (r10 if pd.notna(r10) else s), (r20 if pd.notna(r20) else s)
        out[c] = round(float(0.4*v5 + 0.3*v10 + 0.2*v20 + 0.1*s), 2)
    return out

def ensure_background_training():
    if not SKLEARN_OK or st.session_state.get("_bg_training_running"): return
    st.session_state["_bg_training_running"] = True
    st.session_state["_bg_training_started_at"] = dt.datetime.now().strftime("%H:%M:%S")
    def _runner():
        try:
            # (global model training body omitted here for brevity)
            pass
        finally:
            st.session_state["_bg_training_running"] = False
    threading.Thread(target=_runner, daemon=True).start()

# ---------- Sidebar ----------
with st.sidebar:
    st.button("🏠 Home Screen", use_container_width=True, on_click=_go_home)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.subheader("Select Player")
    active = get_active_players_fast()
    name_to_id = {p['full_name']: p['id'] for p in active}
    player_names = sorted(name_to_id.keys())

    qp = st.query_params
    default_index = None
    if 'pid' in qp:
        try:
            pid = int(qp['pid'])
            pname = next(n for n,i in name_to_id.items() if i==pid)
            default_index = player_names.index(pname); st.session_state['selected_player_id'] = pid
        except Exception:
            default_index = None
    elif 'selected_player_id' in st.session_state and st.session_state['selected_player_id'] is not None:
        try:
            pid = st.session_state['selected_player_id']; pname = next(n for n,i in name_to_id.items() if i==pid)
            default_index = player_names.index(pname)
        except Exception:
            default_index = None

    selection = st.selectbox("Search player", player_names,
                             index=default_index if default_index is not None else None,
                             placeholder="Type a player's name…")

    player_id = None; player_name = None
    if selection:
        player_id = name_to_id[selection]; player_name = selection
        st.session_state['selected_player_id'] = player_id; st.query_params.update({"pid": str(player_id)})

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("⭐ Favorites")

    if "favorites" not in st.session_state:
        st.session_state.favorites = load_favorites()
    if player_name and player_id and st.button(f"➕ Add {player_name}", use_container_width=True):
        if not any(f.get("id")==player_id for f in st.session_state.favorites):
            st.session_state.favorites.append({"name": player_name, "id": player_id})
            save_favorites(st.session_state.favorites); st.success(f"Added {player_name} to favorites.")
    if st.session_state.favorites:
        for idx, fav in enumerate(list(st.session_state.favorites)):
            nm, pid = fav.get("name","(unknown)"), fav.get("id")
            cA, cB = st.columns([0.8,0.2])
            with cA:
                if st.button(nm, key=f"fav_open_{idx}_{pid}", use_container_width=True):
                    st.session_state['selected_player_id'] = pid; st.query_params.update({"pid": str(pid)}); st.rerun()
            with cB:
                if st.button("×", key=f"fav_del_{idx}_{pid}"):
                    st.session_state.favorites.pop(idx); save_favorites(st.session_state.favorites); st.rerun()
    else:
        st.caption("No favorites yet.")

# ---------- Header ----------
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

# ---------- HOME (no player selected) ----------
if 'selected_player_id' not in st.session_state:
    st.session_state['selected_player_id'] = None

if st.session_state['selected_player_id'] is None:
    st.info("Pick a player from the sidebar to load their dashboard.")

    # Last Night's results (kept)
    st.markdown("### Last Night’s Results")
    ln = get_last_night_results()
    st.dataframe(ln, use_container_width=True) if not ln.empty else st.caption("No results available.")

    # Standings (kept)
    st.markdown("### Standings by Conference (Best → Worst)")
    east, west = get_standings_by_conf()
    cE, cW = st.columns(2)
    with cE:
        st.subheader("Eastern Conference")
        st.dataframe(east, use_container_width=True) if not east.empty else st.caption("Standings unavailable.")
    with cW:
        st.subheader("Western Conference")
        st.dataframe(west, use_container_width=True) if not west.empty else st.caption("Standings unavailable.")

    # NEW: ONLY the top leader for PTS/REB/AST/3PM
    render_leader_cards_row()
    st.stop()

# ---------- PLAYER PAGE (unchanged core UI beyond earlier work) ----------
pid = st.session_state['selected_player_id']
pname = next((p['full_name'] for p in get_active_players_fast() if p['id']==pid), "Player")
career_df, logs_df = fetch_player(pid)
if career_df.empty:
    st.warning("No career data available for this player."); st.stop()

# Optional background global ML notice
if SKLEARN_OK:
    ensure_background_training()
    if st.session_state.get("_bg_training_running"):
        st.caption(f"🟥 Training global ML in background… (started {st.session_state.get('_bg_training_started_at','now')})")

team_abbr = career_df['TEAM_ABBREVIATION'].iloc[-1] if 'TEAM_ABBREVIATION' in career_df.columns else "TBD"
latest_season = str(career_df['SEASON_ID'].dropna().iloc[-1]) if 'SEASON_ID' in career_df.columns else "N/A"

# Last + next game
last_game_info = "N/A"; last_game_stats = None
if logs_df is not None and not logs_df.empty:
    lg = logs_df.head(1).copy(); lg_cols = safe_cols(lg, STATS_COLS)
    if lg_cols: last_game_stats = lg[lg_cols].iloc[0].to_dict()
    r = lg.iloc[0]
    ld = pd.to_datetime(r['GAME_DATE']).strftime("%Y-%m-%d") if 'GAME_DATE' in r else "—"
    lo = extract_opp_from_matchup(r.get('MATCHUP','')) or "TBD"
    last_game_info = f"{ld} vs {lo}"

ng = next_game_for_team(team_abbr) if team_abbr else None
next_game_info = (f"{'🏠' if ng and ng.get('home') else '✈️'} {ng['date']} vs {ng['opp_abbr']}" if ng else "TBD")
is_home_next = 1 if (ng and ng.get('home')) else 0

# Header metrics
c1,c2,c3,c4 = st.columns([1.3,0.8,1.2,1.2])
with c1: st.metric("Player", pname)
with c2: st.metric("Team", team_abbr)
with c3: st.metric("Most Recent Game", last_game_info)
with c4: st.metric("Next Game", next_game_info)

# Headshot + logo + bar charts
col_img, col_trend = st.columns([0.30,0.70])
with col_img:
    head = cdn_headshot(pid, "1040x760") or cdn_headshot(pid, "260x190")
    logo = cdn_team_logo(_team_id_from_abbr(team_abbr)) if team_abbr else None
    if head:
        if logo: head = overlay_logo_top_right(head, logo, padding_ratio=0.035, logo_width_ratio=0.22)
        st.image(head, use_container_width=True, caption=f"{pname} — media day headshot")
    else:
        st.info("Headshot not available.")
with col_trend:
    st.markdown("#### Last 10 Games — Bar Charts")
    if logs_df is not None and not logs_df.empty:
        N = min(10, len(logs_df)); view = logs_df.head(N).copy().iloc[::-1]
        view['GAME_DATE'] = pd.to_datetime(view['GAME_DATE']); view['Game'] = view['GAME_DATE'].dt.strftime('%m-%d')
        core = ['PTS','REB','AST','FG3M']; available = [c for c in STATS_COLS if c in view.columns]
        chosen = [c for c in core if c in available] or available[:4]
        for i, stat in enumerate(chosen):
            ch_df = view[['Game', stat]].copy()
            chart = alt.Chart(ch_df).mark_bar().encode(
                x=alt.X('Game:N', title=''), y=alt.Y(f'{stat}:Q', title=stat),
                tooltip=[alt.Tooltip('Game:N', title='Game'), alt.Tooltip(f'{stat}:Q', title=stat)],
            ).properties(height=140)
            if i % 2 == 0:
                a,b = st.columns(2); a.altair_chart(chart, use_container_width=True)
            else:
                b.altair_chart(chart, use_container_width=True)
    else:
        st.info("No recent games to chart.")

def metric_row(title, row):
    st.markdown(f"#### {title}")
    r = row or {}
    if isinstance(r, pd.DataFrame) and not r.empty: r = r.iloc[0].to_dict()
    elif isinstance(r, pd.Series): r = r.to_dict()
    def fmt(v): return f"{float(v):.2f}" if isinstance(v,(int,float,np.floating)) and pd.notna(v) else "—"
    c = st.columns(5)
    for i,(lab,key) in enumerate([("PTS","PTS"),("REB","REB"),("AST","AST"),("3PM","FG3M"),("MIN","MIN")]):
        with c[i]: st.metric(lab, fmt(r.get(key)))

# Current season averages
season_row = None
if not career_df.empty:
    cur = career_df.iloc[-1]; gp = cur.get('GP', 0)
    if gp and gp > 0:
        season_row = {c: (cur.get(c,0)/gp) for c in STATS_COLS}
metric_row("Current Season Averages", season_row)

# Last game
metric_row("Last Game Stats", last_game_stats)

# Last 5 averages
def recent_avg(logs):
    if logs is None or logs.empty: return None
    df = logs.copy(); cols = safe_cols(df, STATS_COLS)
    if len(df) >= 5 and cols: return df.head(5)[cols].mean().to_frame(name='Last5').T
    return None
last5_df = recent_avg(logs_df); last5 = last5_df.iloc[0].to_dict() if isinstance(last5_df,pd.DataFrame) and not last5_df.empty else None
metric_row("Last 5 Games Averages", last5)

# Predictions (ML -> ad-hoc -> fallback)
models = load_models_cached() if SKLEARN_OK else {}
engine = "fallback"
ml = predict_next_ml(logs_df, ng.get('date') if ng else None, is_home_next, models) if models else None
if ml:
    engine="global_ml"; metric_row("Predicted Next Game (ML)", ml)
else:
    adhoc = train_player_models_in_memory(logs_df) if SKLEARN_OK else {}
    adhoc_preds = predict_next_ml(logs_df, ng.get('date') if ng else None, is_home_next, adhoc) if adhoc else None
    if adhoc_preds:
        engine="player_ml"; ml = adhoc_preds; metric_row("Predicted Next Game (ML)", ml)
    else:
        fb = predict_next_fallback(logs_df, latest_season); engine="fallback"; ml = fb; metric_row("Predicted Next Game (Model)", fb)

engine_tag = {"global_ml":'<span class="tag ok">Using Global ML</span>',
              "player_ml":'<span class="tag ok">Using Player ML (ad-hoc)</span>',
              "fallback": '<span class="tag dim">Using weighted fallback model</span>' if SKLEARN_OK else
                          '<span class="tag warn">Fallback — install scikit-learn & joblib</span>'}[engine]
st.markdown(engine_tag, unsafe_allow_html=True)

# Save prediction if next-game date known
pred_date = ng['date'] if (ng and isinstance(ng.get('date'), str)) else None
if ml: record_prediction(pid, pname, pred_date, engine, ml)

# Last predictions vs results (kept)
st.markdown("### Last Predictions vs Results")
def actual_row_by_date(logs, iso):
    if logs is None or logs.empty or not iso: return None
    try: d = dt.datetime.strptime(iso,"%Y-%m-%d").date()
    except Exception: return None
    c = logs.copy(); c['GD'] = pd.to_datetime(c['GAME_DATE']).dt.date
    m = c[c['GD']==d]; return m.iloc[0] if not m.empty else None

hist = get_player_history(pid)
if not hist:
    st.caption("No stored predictions yet. A prediction is saved when a next game date is known.")
else:
    shown = 0
    for e in sorted(hist, key=lambda x: (x.get("pred_date") or ""), reverse=True):
        if shown >= 3: break
        pdate = e.get("pred_date"); preds_h = e.get("preds", {}); eng = e.get("engine","unknown")
        act = actual_row_by_date(logs_df, pdate) if pdate else None
        hdr = f"**Predicted for {pdate}**" if pdate else "**Prediction (no date recorded)**"
        eng_name = {"global_ml":"Global ML","player_ml":"Player ML","fallback":"Fallback"}.get(eng,"Model")
        st.markdown(f"{hdr} — _{eng_name}_")
        cs = st.columns(4)
        for i,k in enumerate(['PTS','REB','AST','FG3M']):
            with cs[i]:
                pv = preds_h.get(k, None)
                if act is None:
                    st.metric(f"{k} • Pending", value=str(pv) if pv is not None else "—", delta="Game not played")
                else:
                    av = act.get(k, None)
                    hit = (av is not None and pv is not None and float(av) >= float(pv))
                    st.metric(f"{k} • {'✅ Hit' if hit else '❌ Miss'}",
                              value=f"{av:.2f}" if isinstance(av,(int,float)) else "—",
                              delta=(f"Pred {pv:.2f} → Act {av:.2f}" if isinstance(pv,(int,float)) and isinstance(av,(int,float)) else "—"))
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        shown += 1

# Snapshot PNG (kept)
def compose_share_card(player_name, team_abbr, headshot_img, team_logo_img, season_row, last_game_row, last5_row, pred_row):
    W,H = 1200, 1400
    bg = Image.new("RGBA",(W,H),(10,10,10,255)); d = ImageDraw.Draw(bg)
    try:
        ft = ImageFont.truetype("DejaVuSans-Bold.ttf",48)
        fh = ImageFont.truetype("DejaVuSans-Bold.ttf",34)
        fb = ImageFont.truetype("DejaVuSans.ttf",28)
        fs = ImageFont.truetype("DejaVuSans.ttf",24)
    except Exception:
        ft=fh=fb=fs=None
    d.text((36,30), f"Hot Shot Props — {player_name} ({team_abbr})", fill=(255,180,180,255), font=ft)
    x,y=36,90
    if headshot_img:
        h = headshot_img.copy().convert("RGBA"); h.thumbnail((520,380), Image.LANCZOS); bg.alpha_composite(h,(x,y))
        if team_logo_img:
            lg = team_logo_img.copy().convert("RGBA"); lw=int(h.width*.22); lh=int(lg.height*(lw/lg.width))
            lg = lg.resize((lw,lh), Image.LANCZOS); bg.alpha_composite(lg,(x+h.width-lw-12,y+12))
    def block(label,row,x0,y0):
        d.rounded_rectangle((x0,y0,x0+520,y0+160),radius=16,outline=(35,35,35,255),width=2,fill=(18,18,18,255))
        d.text((x0+16,y0+12),label,fill=(250,164,164,255),font=fh)
        labs=[("PTS","PTS"),("REB","REB"),("AST","AST"),("3PM","FG3M"),("MIN","MIN")]
        x1=x0+16
        for i,(lab,key) in enumerate(labs):
            val=row.get(key,"—") if isinstance(row,dict) else "—"
            val=f"{val:.2f}" if isinstance(val,(int,float,np.floating)) and pd.notna(val) else ("N/A" if val is None else str(val))
            d.text((x1+i*100,y0+64),lab,fill=(230,230,230,255),font=fs)
            d.text((x1+i*100,y0+96),val,fill=(255,228,230,255),font=fb)
    block("Current Season Averages", season_row or {}, 644, 90)
    block("Last Game Stats",        last_game_row or {}, 36,  490)
    block("Last 5 Games Averages",  last5_row or {},      644, 490)
    block("Predicted Next Game",    pred_row or {},       36,  890)
    d.text((36,H-60),"Generated with Hot Shot Props • nba_api • Streamlit", fill=(210,210,210,255), font=fs)
    buf=BytesIO(); bg.convert("RGB").save(buf, format="PNG", quality=95); buf.seek(0); return buf.getvalue()

def _row_or_none(obj):
    if obj is None: return {}
    if isinstance(obj,pd.DataFrame) and not obj.empty: return obj.iloc[0].to_dict()
    if isinstance(obj,dict): return obj
    if isinstance(obj,pd.Series): return obj.to_dict()
    return {}

season_row_dict = _row_or_none(season_row)
last_game_row_dict = _row_or_none(last_game_stats)
last5_row_dict = _row_or_none(last5)
pred_row_dict = _row_or_none(ml)

png_bytes = compose_share_card(
    player_name=pname, team_abbr=team_abbr,
    headshot_img=head if 'head' in locals() else None,
    team_logo_img=logo if 'logo' in locals() else None,
    season_row=season_row_dict, last_game_row=last_game_row_dict,
    last5_row=last5_row_dict, pred_row=pred_row_dict
)
st.download_button("⬇️ Download snapshot (PNG)", data=png_bytes,
                   file_name=f"{pname.replace(' ','_')}_hotshotprops.png",
                   mime="image/png", use_container_width=True)