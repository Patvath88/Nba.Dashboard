# app.py — Hot Shot Props | NBA Player Analytics
# - One-page UX (NBA-like)
# - Sidebar search: "TEAM_ABBR — Player Name"
# - Sidebar favorites under search; each has a tiny × remove button
# - Headshot with team-logo overlay (top-right)
# - Auto ML training (background thread) starts on EVERY player selection (no freshness check)
# - If ML models present -> uses ML; otherwise weighted fallback
# - Shows which prediction engine was used in the UI

import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playercareerstats,
    playergamelogs,
    scoreboardv2,
    commonteamroster,
)
import pandas as pd
import numpy as np
import re, time, datetime as dt, json, os, threading
import altair as alt
import requests
from io import BytesIO
from PIL import Image

# ---- ML deps (optional; used if available) ----
try:
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
except Exception:
    joblib = None
    Ridge = None

# ---------------------------------
# Page config + constants
# ---------------------------------
st.set_page_config(
    page_title="Hot Shot Props • NBA Player Analytics (Free)",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATS_COLS   = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']
PREDICT_COLS = ['PTS','AST','REB','FG3M']
API_SLEEP    = 0.25
FAV_FILE     = "favorites.json"
MODEL_DIR    = "models"
MODEL_FILES  = {t: os.path.join(MODEL_DIR, f"model_{t}.pkl") for t in PREDICT_COLS}
FEAT_TEMPLATE = lambda tgt: [f'{tgt}_r5', f'{tgt}_r10', f'{tgt}_r20', f'{tgt}_season_mean', 'IS_HOME', 'DAYS_REST']

# ---------------------------------
# Styling
# ---------------------------------
st.markdown("""
<style>
:root { --bg:#0f1116; --panel:#121722; --ink:#e5e7eb; --muted:#9aa3b2; --blue:#1d4ed8; --line:#1f2a44; }
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--ink);}
h1,h2,h3,h4{color:#b3d1ff!important;letter-spacing:.2px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f1629 0%,#0f1116 100%);border-right:1px solid #1f2937;}
.stButton>button{background:var(--blue);color:white;border:none;border-radius:10px;padding:.5rem .9rem;font-weight:700;}
.stButton>button:hover{background:#2563eb;}
[data-testid="stMetric"]{background:var(--panel);padding:16px;border-radius:16px;border:1px solid var(--line);box-shadow:0 8px 28px rgba(0,0,0,.35);}
[data-testid="stMetric"] label{color:#cfe1ff;}
[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#e0edff;font-size:1.5rem;}
.stDataFrame{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:4px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 8px 28px rgba(0,0,0,.35);}
.badge{display:inline-block;padding:4px 10px;font-size:.8rem;border:1px solid var(--line);border-radius:9999px;background:#0b1222;color:#9bd1ff;}
.inline-x button{background:#172036!important;border:1px solid #24324f!important;color:#cbd5e1!important;padding:.1rem .4rem!important;font-weight:800;border-radius:6px;}
.hr{border:0;border-top:1px solid var(--line);margin:.5rem 0;}
.fav-btn button{background:transparent!important;color:#93c5fd!important;border:none!important;box-shadow:none!important;text-decoration:underline;cursor:pointer;padding:.25rem 0;}
.fav-btn button:hover{color:#bfdbfe!important;text-decoration:underline;}
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

def cdn_headshot(player_id: int, size: str = "1040x760") -> Image.Image | None:
    """NBA media day headshot from official CDN."""
    url = f"https://cdn.nba.com/headshots/nba/latest/{size}/{player_id}.png"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content)).convert("RGBA")
    except Exception:
        pass
    return None

def cdn_team_logo(team_id: int, size: str = "L") -> Image.Image | None:
    """Team logo from NBA CDN."""
    candidates = [
        f"https://cdn.nba.com/logos/nba/{team_id}/global/{size}/logo.png",
        f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.png",
        f"https://cdn.nba.com/logos/nba/{team_id}/global/D/logo.png",
    ]
    for url in candidates:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return Image.open(BytesIO(r.content)).convert("RGBA")
        except Exception:
            continue
    return None

def overlay_logo_top_right(headshot: Image.Image, logo: Image.Image,
                           padding_ratio: float = 0.035, logo_width_ratio: float = 0.22) -> Image.Image:
    """Paste team logo onto headshot's top-right corner with padding."""
    if headshot is None:
        return None
    base = headshot.copy()
    if logo is None:
        return base
    W, H = base.size
    pad = int(W * padding_ratio)
    logo_w = int(W * logo_width_ratio)
    aspect = logo.size[1] / logo.size[0]
    logo_h = int(logo_w * aspect)
    logo_resized = logo.resize((logo_w, logo_h), Image.LANCZOS)
    x = W - logo_w - pad
    y = pad
    base.alpha_composite(logo_resized, (x, y))
    return base

# --- Favorites persistence (store name + id) ---
def load_favorites() -> list[dict]:
    try:
        if os.path.exists(FAV_FILE):
            with open(FAV_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    out = []
                    for x in data:
                        if isinstance(x, dict) and "name" in x and "id" in x:
                            out.append(x)
                        elif isinstance(x, str):
                            out.append({"name": x, "id": None})
                    return out
    except Exception:
        pass
    return []

def save_favorites(favs: list[dict]) -> None:
    try:
        with open(FAV_FILE, "w") as f:
            json.dump(favs, f)
    except Exception:
        pass

# ---------------------------------
# Cached data
# ---------------------------------
@st.cache_data(ttl=6*3600)
def get_active_players():
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

@st.cache_data(ttl=12*3600)
def build_team_player_index() -> dict[str, int]:
    """
    Returns mapping: 'TEAM_ABBR — Player Name' -> PLAYER_ID
    Built from each team's roster (cached), plus extra active players marked 'FA'.
    """
    mapping: dict[str, int] = {}
    all_t = get_all_teams()
    for t in all_t:
        tid  = t.get("id")
        abbr = t.get("abbreviation", "TBD")
        if not tid:
            continue
        try:
            time.sleep(API_SLEEP)
            roster = commonteamroster.CommonTeamRoster(team_id=tid).get_data_frames()[0]
            if roster is None or roster.empty:
                continue
            for _, row in roster.iterrows():
                name = str(row.get("PLAYER", "")).strip()
                pid  = int(row.get("PLAYER_ID")) if pd.notna(row.get("PLAYER_ID")) else None
                if name and pid:
                    mapping[f"{abbr} — {name}"] = pid
        except Exception:
            continue
    # Add any remaining active players not captured by rosters (FA, two-way, etc.)
    act = get_active_players()
    present_names = set(label.split("—",1)[-1].strip() for label in mapping.keys())
    for p in act:
        name = p.get('full_name')
        pid  = p.get('id')
        if name and pid and name not in present_names:
            mapping[f"FA — {name}"] = pid
    return mapping

@st.cache_data(ttl=3600)
def fetch_player(player_id: int):
    """Returns (career_by_season_df, career_logs_df)."""
    try:
        time.sleep(API_SLEEP)
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]
        seasons = career_stats['SEASON_ID'].tolist() if not career_stats.empty else []
        logs_list = []
        for s in seasons:
            try:
                time.sleep(API_SLEEP)
                df = playergamelogs.PlayerGameLogs(
                    player_id_nullable=player_id, season_nullable=s
                ).get_data_frames()[0]
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
    """Scan next days via ScoreboardV2 to find the team's next game."""
    if team_id is None:
        return None
    today = dt.date.today()
    team_map = {t['id']: t for t in get_all_teams()}
    for d in range(lookahead_days):
        day = today + dt.timedelta(days=d)
        try:
            time.sleep(API_SLEEP)
            sb = scoreboardv2.ScoreboardV2(game_date=day.strftime("%m/%d/%Y"))
            frames = sb.get_data_frames()
            game_header = next((f for f in frames if {'HOME_TEAM_ID','VISITOR_TEAM_ID'}.issubset(f.columns)), None)
            if game_header is None or game_header.empty:
                continue
            for _, row in game_header.iterrows():
                home_id = int(row.get('HOME_TEAM_ID', -1))
                away_id = int(row.get('VISITOR_TEAM_ID', -1))
                if team_id in (home_id, away_id):
                    opp_id  = away_id if team_id == home_id else home_id
                    opp_abr = team_map.get(opp_id, {}).get('abbreviation', 'TBD')
                    home_flag = (team_id == home_id)
                    return {'date': day.strftime("%Y-%m-%d"), 'opp_abbr': opp_abr, 'home': home_flag}
        except Exception:
            continue
    return None

# ---------------------------------
# ML: loaders + predictors + background training
# ---------------------------------
@st.cache_data(ttl=3600)
def load_models():
    """Load model pkl files if available. Returns dict target->model."""
    models = {}
    if joblib is None:
        return models
    for tgt, path in MODEL_FILES.items():
        try:
            if os.path.exists(path):
                models[tgt] = joblib.load(path)
        except Exception:
            continue
    return models

def build_features_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling r5/r10/r20, season_mean (shifted), IS_HOME, DAYS_REST."""
    if df is None or df.empty:
        return pd.DataFrame()
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

def build_features_for_inference(logs_df: pd.DataFrame, next_game_date: dt.date | None, is_home_next: int) -> pd.DataFrame | None:
    """Single-row features for NEXT game (same schema as training)."""
    if logs_df is None or logs_df.empty:
        return None
    df = logs_df.copy()
    if 'GAME_DATE' not in df.columns:
        return None
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)

    last_date = pd.to_datetime(df['GAME_DATE'].iloc[-1]).date()
    if isinstance(next_game_date, str):
        try:
            next_game_date = dt.datetime.strptime(next_game_date, "%Y-%m-%d").date()
        except Exception:
            next_game_date = None
    days_rest = 3 if next_game_date is None else max(0, min(7, (next_game_date - last_date).days))

    feat = {}
    for k in [5, 10, 20]:
        for c in STATS_COLS:
            series = df[c].rolling(k, min_periods=1).mean().shift(1)
            val = series.iloc[-1] if pd.notna(series.iloc[-1]) else (df[c].iloc[:-1].mean() if len(df) > 1 else df[c].iloc[-1])
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

def predict_next_ml(logs_df: pd.DataFrame, next_game_date: dt.date | None, is_home_next: int, models: dict[str, object]) -> dict | None:
    if not models:
        return None
    feats_df = build_features_for_inference(logs_df, next_game_date, is_home_next)
    if feats_df is None or feats_df.empty:
        return None
    preds = {}
    for tgt, model in models.items():
        feat_cols = FEAT_TEMPLATE(tgt)
        if not all(c in feats_df.columns for c in feat_cols):
            continue
        try:
            val = float(model.predict(feats_df[feat_cols])[0])
            preds[tgt] = round(val, 2)
        except Exception:
            continue
    return preds if preds else None

def train_models_core(active_players=None):
    """Core trainer with no Streamlit UI calls (safe for background thread)."""
    if joblib is None or Ridge is None:
        return False
    os.makedirs(MODEL_DIR, exist_ok=True)
    act = active_players if active_players is not None else players.get_active_players()
    rows = []
    for p in act:
        pid = p.get('id')
        try:
            cstats = playercareerstats.PlayerCareerStats(player_id=pid).get_data_frames()[0]
            seasons = cstats['SEASON_ID'].tolist() if not cstats.empty else []
            logs_list = []
            for s in seasons:
                try:
                    time.sleep(API_SLEEP)
                    df = playergamelogs.PlayerGameLogs(player_id_nullable=pid, season_nullable=s).get_data_frames()[0]
                    if df is not None and not df.empty:
                        logs_list.append(df)
                except Exception:
                    pass
            if not logs_list:
                continue
            logs = pd.concat(logs_list, ignore_index=True)
            if 'GAME_DATE' not in logs.columns:
                continue
            logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
            logs = logs.sort_values('GAME_DATE').reset_index(drop=True)
            feats = build_features_for_training(logs)
            if feats.empty:
                continue
            rows.append(feats)
        except Exception:
            continue
    if not rows:
        return False
    data = pd.concat(rows, ignore_index=True)
    for target in PREDICT_COLS:
        feat_cols = FEAT_TEMPLATE(target)
        if not all(c in data.columns for c in feat_cols):
            continue
        X = data[feat_cols]; y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = Ridge(alpha=1.0, random_state=42).fit(X_train, y_train)
        joblib.dump(model, MODEL_FILES[target])
    return True

def ensure_background_training():
    """
    ALWAYS attempt background training on selection (no freshness check).
    Guarded so only one thread runs at a time.
    """
    if joblib is None or Ridge is None:
        return
    if st.session_state.get("_bg_training_running"):
        return
    st.session_state["_bg_training_running"] = True
    st.session_state["_bg_training_started_at"] = dt.datetime.now().strftime("%H:%M:%S")

    def _runner():
        try:
            train_models_core()
        finally:
            st.session_state["_bg_training_running"] = False

    threading.Thread(target=_runner, daemon=True).start()

# ---------------------------------
# Fallback (non-ML) prediction
# ---------------------------------
def recent_averages(logs: pd.DataFrame) -> dict:
    out = {}
    if logs is None or logs.empty:
        return out
    df = logs.copy()
    cols = safe_cols(df, STATS_COLS)
    if len(df) >= 5 and cols:
        sub = df.head(5)
        out['Last 5 Avg'] = sub[cols].mean().to_frame(name='Last 5 Avg').T
    return out

def predict_next_fallback(logs: pd.DataFrame, season_label: str) -> dict | None:
    """Simple weighted-averages model as a fallback."""
    if logs is None or logs.empty:
        return None
    df = logs.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
    cols = safe_cols(df, STATS_COLS)
    if not cols:
        return None
    season_avg = pd.Series(dtype=float)
    if 'SEASON_YEAR' in df.columns:
        cur = df[df['SEASON_YEAR'] == season_label]
        if not cur.empty:
            season_avg = cur[cols].mean()
    if season_avg.empty:
        season_avg = df[cols].mean()
    feats = {f'{c}_r5': df[c].rolling(5, min_periods=1).mean().iloc[-1] for c in cols}
    feats.update({f'{c}_r10': df[c].rolling(10, min_periods=1).mean().iloc[-1] for c in cols})
    feats.update({f'{c}_r20': df[c].rolling(20, min_periods=1).mean().iloc[-1] for c in cols})
    preds = {}
    for c in PREDICT_COLS:
        r5, r10, r20 = feats.get(f'{c}_r5', np.nan), feats.get(f'{c}_r10', np.nan), feats.get(f'{c}_r20', np.nan)
        s = season_avg.get(c, 0.0) if pd.notna(season_avg.get(c, np.nan)) else 0.0
        v5, v10, v20 = (r5 if pd.notna(r5) else s), (r10 if pd.notna(r10) else s), (r20 if pd.notna(r20) else s)
        preds[c] = round(float(0.4*v5 + 0.3*v10 + 0.2*v20 + 0.1*s), 2)
    return preds

# ---------------------------------
# Sidebar — Search + Favorites (with inline remove)
# ---------------------------------
with st.sidebar:
    st.header("Select Player")
    label_to_pid = build_team_player_index()
    options = sorted(label_to_pid.keys())

    default_index = None
    if 'selected_player_id' in st.session_state and st.session_state['selected_player_id'] is not None:
        try:
            default_index = next(i for i, lbl in enumerate(options) if label_to_pid[lbl] == st.session_state['selected_player_id'])
        except StopIteration:
            default_index = None

    selection = st.selectbox(
        "Search by team or player",
        options,
        index=default_index if default_index is not None else None,
        placeholder="e.g., LAL — LeBron James  •  BOS — Jayson Tatum  •  Nikola Jokic"
    )

    player_id = None
    player_name = None
    if selection:
        player_id = label_to_pid[selection]
        player_name = selection.split("—", 1)[-1].strip() if "—" in selection else selection.strip()
        st.session_state['selected_player_id'] = player_id

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("⭐ Favorites")

    if "favorites" not in st.session_state:
        st.session_state.favorites = load_favorites()

    # Quick-add current to favorites
    if player_name and player_id and st.button(f"➕ Add {player_name}", use_container_width=True):
        if not any(f.get("id")==player_id for f in st.session_state.favorites):
            st.session_state.favorites.append({"name": player_name, "id": player_id})
            save_favorites(st.session_state.favorites)
            st.success(f"Added {player_name} to favorites.")

    # Render favorites list: [Name] [×]
    if st.session_state.favorites:
        for idx, fav in enumerate(st.session_state.favorites):
            nm = fav.get("name", "(unknown)")
            pid = fav.get("id")
            colN, colX = st.columns([0.8, 0.2], vertical_alignment="center")
            with colN:
                if st.container().button(nm, key=f"fav_open_{idx}_{pid}", help="Open player", use_container_width=True):
                    st.session_state['selected_player_id'] = pid
                    st.rerun()
            with colX:
                if st.container().button("×", key=f"fav_del_{idx}_{pid}", help="Remove", type="secondary"):
                    st.session_state.favorites.pop(idx)
                    save_favorites(st.session_state.favorites)
                    st.rerun()
    else:
        st.caption("No favorites yet.")

# ---------------------------------
# Header card
# ---------------------------------
st.markdown("""
<div class="card" style="margin-bottom: 12px;">
  <div style="display:flex; align-items:center; justify-content: space-between;">
    <div>
      <h1 style="margin:0;">Hot Shot Props — NBA Player Analytics</h1>
      <div class="badge">One-page • Sidebar favorites • ML predictions (background only)</div>
    </div>
    <div style="text-align:right; font-size:.9rem; color:#9aa3b2;">Trends • Weighted fallback • Clean UX</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# Main content
# ---------------------------------
if player_id:
    # kick off background training EVERY time you select a player
    ensure_background_training()
    if st.session_state.get("_bg_training_running"):
        st.caption(f"🟦 Training ML models in background… (started {st.session_state.get('_bg_training_started_at','now')})")

if not player_id:
    st.info("Pick a player from the sidebar to load their dashboard.")
else:
    # Fetch data
    career_df, logs_df = fetch_player(player_id)
    if career_df.empty:
        st.warning("No career data available for this player.")
        st.stop()

    # Resolve name (if missing)
    if not player_name:
        for p in get_active_players():
            if p['id'] == player_id:
                player_name = p['full_name']; break

    all_t = get_all_teams()
    team_by_abbr = {t['abbreviation']: t for t in all_t}

    team_abbr = career_df['TEAM_ABBREVIATION'].iloc[-1] if 'TEAM_ABBREVIATION' in career_df.columns else "TBD"
    team_id = team_by_abbr.get(team_abbr, {}).get('id')
    latest_season = str(career_df['SEASON_ID'].dropna().iloc[-1]) if 'SEASON_ID' in career_df.columns else "N/A"

    # Last game
    last_game_info = "N/A"
    last_game_stats = None
    if logs_df is not None and not logs_df.empty:
        lg_df = logs_df.head(1).copy()
        lg_cols = safe_cols(lg_df, STATS_COLS)
        if lg_cols:
            last_game_stats = lg_df[lg_cols].iloc[0].to_dict()
        lg_row  = lg_df.iloc[0]
        lg_date = pd.to_datetime(lg_row['GAME_DATE']).strftime("%Y-%m-%d") if 'GAME_DATE' in lg_row else "—"
        lg_opp  = extract_opp_from_matchup(lg_row.get('MATCHUP', '')) or "TBD"
        last_game_info = f"{lg_date} vs {lg_opp}"

    # Next game
    ng = next_game_for_team(team_id) if team_id is not None else None
    if ng:
        icon = "🏠" if ng.get('home') else "✈️"
        next_game_info = f"{icon} {ng['date']} vs {ng['opp_abbr']}"
        is_home_next = 1 if ng.get('home') else 0
        ng_date_for_feat = dt.datetime.strptime(ng['date'], "%Y-%m-%d").date()
    else:
        next_game_info = "TBD"
        is_home_next = 0
        ng_date_for_feat = None

    # Header strip metrics
    c1, c2, c3, c4 = st.columns([1.3, 0.8, 1.2, 1.2])
    with c1: st.metric("Player", player_name)
    with c2: st.metric("Team", team_abbr)
    with c3: st.metric("Most Recent Game", last_game_info)
    with c4: st.metric("Next Game", next_game_info)

    # Headshot + logo overlay + trend charts
    col_img, col_trend = st.columns([0.32, 0.68])
    with col_img:
        head = cdn_headshot(player_id, "1040x760") or cdn_headshot(player_id, "260x190")
        logo = cdn_team_logo(team_id) if team_id else None
        if head:
            composed = overlay_logo_top_right(head, logo, padding_ratio=0.035, logo_width_ratio=0.22)
            st.image(composed, use_container_width=True, caption=f"{player_name} — media day headshot")
        else:
            st.info("Headshot not available.")

    with col_trend:
        if logs_df is not None and not logs_df.empty:
            N = min(10, len(logs_df))
            view = logs_df.head(N).copy().iloc[::-1]
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

    # Metric row helper
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

    # Row 1: Current season per-game
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
    ra = recent_averages(logs_df)
    last5 = ra['Last 5 Avg'].iloc[0].to_dict() if 'Last 5 Avg' in ra and not ra['Last 5 Avg'].empty else None
    metric_row("Last 5 Games Averages", last5)

    # Row 4: Predicted next game (ML or fallback) + indicator of which engine
    ml_models = load_models()
    source_label = ""
    if ml_models:
        ml_preds  = predict_next_ml(logs_df, ng_date_for_feat, is_home_next, ml_models)
    else:
        ml_preds = None

    if ml_preds:
        metric_row("Predicted Next Game (ML)", ml_preds)
        source_label = "Using ML models"
    else:
        preds = predict_next_fallback(logs_df, latest_season)
        metric_row("Predicted Next Game (Model)", preds)
        if joblib is None or Ridge is None:
            source_label = "Using weighted fallback model (install scikit-learn & joblib to enable ML)"
        else:
            source_label = "Using weighted fallback model (ML models not available yet)"

    st.caption(source_label)
