# app.py — Hot Shot Props | NBA Player Analytics
# - Home: 2025–26 per-game leaders (PTS/REB/AST/3PM) with headshots + links
# - Player: headshot, season/last-5 metrics, NEXT GAME prediction with clear label:
#           "(Machine Learning)" when a background Ridge model is ready,
#           otherwise "(Weighted Avg)" fallback (WMA)
# - Background ML: per-player ad-hoc Ridge model auto-trains in a background thread
#                  the first time you open a player page (non-blocking)
# - Robust caching, short retries, width='stretch' (no deprecated args), no balldontlie

import time
import math
import threading
from typing import Dict, Optional, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st

from nba_api.stats.static import players as static_players, teams as static_teams
from nba_api.stats.endpoints import leagueleaders, playergamelogs

# -------- Optional ML deps (safe fallback if not installed) --------
try:
    from sklearn.linear_model import Ridge
    SKLEARN_OK = True
except Exception:
    Ridge = None
    SKLEARN_OK = False

# ---------------- Page / Theme ----------------
st.set_page_config(page_title="Hot Shot Props — NBA Player Analytics", layout="wide")

st.markdown("""
<style>
:root{--bg:#0b0b0b;--panel:#111;--ink:#eaeaea;--line:#1d1d1d;--accent:#ff6b6b;}
html, body, [data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink)!important;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;}
h1,h2,h3{color:#ffd5d5;}
a, a:visited{color:#ffa6a6;text-decoration:none;}
.leader-card{display:flex;gap:12px;align-items:center;}
.leader-img{width:64px;height:64px;border-radius:10px;overflow:hidden;background:#000;border:1px solid #222;}
.leader-name{font-weight:800;font-size:1.05rem;line-height:1.15;}
.leader-meta{opacity:.9;font-size:.92rem;}
.small{opacity:.8;font-size:.85rem;}
hr{border:0;border-top:1px solid var(--line);margin:.35rem 0 1rem;}
[data-testid="stMetric"]{background:#121212;border:1px solid #1d1d1d;border-radius:12px;padding:12px;}
[data-testid="stMetric"] label{color:#ffbcbc}
[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#fff3f3}
</style>
""", unsafe_allow_html=True)

# -------------- Season Override ---------------
SEASON_OVERRIDE = "2025-26"  # <- force 2025–26 everywhere

def season_str_today() -> str:
    return SEASON_OVERRIDE

# -------------- Session state init ------------
def _init_state():
    if "ml_models" not in st.session_state:
        # ml_models[player_id] = {"model": Ridge or None, "features": [cols], "trained_at": ts}
        st.session_state["ml_models"] = {}
    if "active_training" not in st.session_state:
        # active_training holds player_ids currently training in background to avoid duplicate threads
        st.session_state["active_training"] = set()
_init_state()

# -------------- Caching helpers ---------------
@st.cache_data(ttl=60*30, show_spinner=False)
def _players_index():
    plist = static_players.get_players()
    by_id = {p["id"]: p for p in plist}
    by_name = {p["full_name"].lower(): p["id"] for p in plist}
    return by_id, by_name

@st.cache_data(ttl=60*30, show_spinner=False)
def _teams_index():
    tlist = static_teams.get_teams()
    by_id = {t["id"]: t for t in tlist}
    by_abbr = {t["abbreviation"]: t for t in tlist}
    return by_id, by_abbr

# -------------- Headshots ---------------------
def headshot_url(player_id: int) -> str:
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"

# -------------- League Leaders (Per Game) -----
@st.cache_data(ttl=60*30, show_spinner=False)
def league_leaders_df(stat: str) -> pd.DataFrame:
    """
    stat: one of 'PTS','REB','AST','FG3M'
    Returns per-game leaders for the forced season.
    """
    last_exc = None
    for _ in range(3):
        try:
            obj = leagueleaders.LeagueLeaders(
                season=season_str_today(),
                season_type_all_star="Regular Season",
                stat_category_abbreviation=stat.upper(),
                per_mode_simple="PerGame",        # averages
                timeout=10
            )
            df = obj.get_data_frames()[0]
            key = "FG3M" if stat.upper() in ("FG3M", "3PM") else stat.upper()
            if key in df.columns:
                df = df.sort_values(key, ascending=False).reset_index(drop=True)
            return df
        except Exception as e:
            last_exc = e
            time.sleep(0.6)
    raise last_exc

def leader_card(df: pd.DataFrame, label: str):
    if df is None or df.empty:
        st.error(f"{label} unavailable.")
        return
    row = df.iloc[0].to_dict()
    name = row.get("PLAYER", "Unknown")
    team = row.get("TEAM", "")
    # resolve to player_id by name
    by_id, _ = _players_index()
    pid = None
    for pid_cand, pdata in by_id.items():
        if pdata.get("full_name", "").lower() == name.lower():
            pid = pid_cand
            break

    # pick value
    stat_val = None
    for k in ("PTS","REB","AST","FG3M","3PM"):
        if k in df.columns and k in row:
            stat_val = row[k]
    stat_str = f"{stat_val:.2f}" if isinstance(stat_val,(int,float)) and not math.isnan(stat_val) else "—"

    cimg, ctxt = st.columns([1, 4])
    with cimg:
        st.markdown('<div class="leader-img">', unsafe_allow_html=True)
        if pid:
            st.image(headshot_url(pid), width=64)
        st.markdown("</div>", unsafe_allow_html=True)
    with ctxt:
        if pid:
            st.markdown(f"""<div class="leader-name"><a href="?player_id={pid}">{name}</a></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="leader-name">{name}</div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="leader-meta">{team}</div>""", unsafe_allow_html=True)
        st.metric(label=label, value=stat_str)

# -------------- Player logs -------------------
@st.cache_data(ttl=60*15, show_spinner=False)
def player_logs(player_id: int) -> pd.DataFrame:
    """
    Return most recent games for this season for a player (descending by date).
    """
    last_exc = None
    for _ in range(3):
        try:
            gl = playergamelogs.PlayerGameLogs(
                season_nullable=season_str_today(),
                player_id_nullable=player_id,
                timeout=10
            ).get_data_frames()[0]
            gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
            gl = gl.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
            return gl
        except Exception as e:
            last_exc = e
            time.sleep(0.6)
    raise last_exc

# -------------- WMA fallback ------------------
def wma(series: pd.Series, window: int = 8) -> Optional[float]:
    series = series.dropna().head(window)
    if series.empty:
        return None
    weights = pd.Series(range(len(series), 0, -1), index=series.index)
    val = (series * weights).sum() / weights.sum()
    try:
        return float(val)
    except Exception:
        return None

def predict_wma(gl: pd.DataFrame) -> dict:
    out = {}
    for col in ("PTS","REB","AST","FG3M"):
        if col in gl.columns:
            out[col] = wma(gl[col], window=8)
        else:
            out[col] = None
    return out

# -------------- Background ML -----------------
ML_TARGETS = ("PTS","REB","AST","FG3M")
BASE_FEATURES = ("MIN","FGA","FGM","FG3A","FG3M","FTA","FTM","TOV","REB","AST","PTS")

def _build_feature_frame(gl: pd.DataFrame) -> Optional[pd.DataFrame]:
    if gl is None or gl.empty:
        return None
    use_cols = [c for c in BASE_FEATURES if c in gl.columns]
    if not use_cols:
        return None
    df = gl.copy().sort_values("GAME_DATE").reset_index(drop=True)
    # rolling means, shift by 1 to avoid leakage
    for win in (3,5,10):
        for c in use_cols:
            df[f"{c}_r{win}"] = df[c].rolling(win, min_periods=1).mean().shift(1)
    # drop rows with NaNs created by shift (need at least 1 prior)
    df = df.dropna().reset_index(drop=True)
    return df

def _train_model_for_player(pid: int, gl: pd.DataFrame):
    """Trains a separate Ridge model per target (simple multi-output style), stores in session."""
    if not SKLEARN_OK:
        return  # nothing to do
    featdf = _build_feature_frame(gl)
    if featdf is None or len(featdf) < 15:
        return
    # features
    X_cols = [c for c in featdf.columns if c.endswith(("_r3","_r5","_r10"))]
    if not X_cols:
        return
    X = featdf[X_cols].astype(float)
    model_map = {}
    # Train a Ridge per target (keeps it simple & robust)
    for tgt in ML_TARGETS:
        if tgt not in featdf.columns:
            continue
        y = featdf[tgt].astype(float)
        try:
            mdl = Ridge(alpha=1.0)
            mdl.fit(X, y)
            model_map[tgt] = mdl
        except Exception:
            continue
    if not model_map:
        return
    st.session_state["ml_models"][pid] = {
        "models": model_map,
        "features": X_cols,
        "trained_at": time.time()
    }

def _ensure_training_thread(pid: int, gl: pd.DataFrame):
    """Launch training thread once per player if not already training."""
    if not SKLEARN_OK:
        return
    if pid in st.session_state["active_training"]:
        return
    # if we already have a trained model from last 6 hours, skip
    meta = st.session_state["ml_models"].get(pid)
    if meta and (time.time() - meta.get("trained_at", 0)) < 6*60*60:
        return
    # mark and launch
    st.session_state["active_training"].add(pid)
    def _runner():
        try:
            _train_model_for_player(pid, gl)
        finally:
            # always remove the flag even on error
            st.session_state["active_training"].discard(pid)
    t = threading.Thread(target=_runner, daemon=True)
    t.start()

def predict_with_ml_if_ready(pid: int, gl: pd.DataFrame) -> Tuple[dict, bool]:
    """
    If a trained model exists, use it; else return WMA. Returns (preds, used_ml_flag).
    """
    meta = st.session_state["ml_models"].get(pid)
    if not meta or "models" not in meta or "features" not in meta:
        # not ready; fallback
        return predict_wma(gl), False

    X_cols = meta["features"]
    # Build one-row feature vector using latest rolling means
    # We need the same engineered features as training
    featdf = _build_feature_frame(gl)
    if featdf is None or featdf.empty:
        return predict_wma(gl), False
    last_row = featdf.iloc[[-1]][X_cols].astype(float)
    preds = {}
    used = False
    for tgt, mdl in meta["models"].items():
        try:
            preds[tgt] = float(mdl.predict(last_row)[0])
            used = True
        except Exception:
            preds[tgt] = None
    # If nothing predicted, fallback
    if not used:
        return predict_wma(gl), False
    # Return predictions for all keys in ML_TARGETS (fill missing from WMA)
    for tgt in ML_TARGETS:
        if tgt not in preds or preds[tgt] is None or (isinstance(preds[tgt], float) and math.isnan(preds[tgt])):
            preds[tgt] = predict_wma(gl).get(tgt)
    return preds, True

# -------------- Sidebar search ----------------
def sidebar_search():
    st.sidebar.header("Select Player")
    _, by_name = _players_index()
    name = st.sidebar.text_input("Search player", value="", placeholder="Type a player's full name…")
    if name:
        pid = by_name.get(name.strip().lower())
        if pid:
            st.query_params.update({"player_id": str(pid)})
            st.rerun()
        else:
            st.sidebar.info("Not found. Try the full name exactly as it appears on NBA.com.")

# -------------- Pages -------------------------
def page_home():
    st.title("🏠 Home")
    st.caption("Hint: use the sidebar to search any player by name.")

    st.subheader(f"League Leaders — {season_str_today()} (Per Game)")
    cols = st.columns(4, gap="large")
    for (stat, label), c in zip(
        [("PTS","PTS Leader"), ("REB","REB Leader"), ("AST","AST Leader"), ("FG3M","3PM Leader")],
        cols
    ):
        with c:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            try:
                df = league_leaders_df(stat)
                leader_card(df, label)
            except Exception as e:
                st.error(f"Leaders {stat} unavailable: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

def _metric_row(title: str, values: Dict[str, Optional[float]]):
    st.subheader(title)
    c1, c2, c3, c4 = st.columns(4)
    def _fmt(v): return f"{v:.2f}" if isinstance(v,(int,float)) and not (v is None or math.isnan(v)) else "—"
    with c1: st.metric("PTS", _fmt(values.get("PTS")))
    with c2: st.metric("REB", _fmt(values.get("REB")))
    with c3: st.metric("AST", _fmt(values.get("AST")))
    with c4: st.metric("3PM", _fmt(values.get("FG3M")))

def page_player(player_id: int):
    by_id, _ = _players_index()
    pmeta = by_id.get(player_id, {})
    pname = pmeta.get("full_name", f"Player {player_id}")
    st.header(pname)

    # Load logs (and kick off background training)
    with st.spinner("Loading games…"):
        try:
            gl = player_logs(player_id)
        except Exception as e:
            st.error(f"Failed to load from nba_api (likely a temporary block). Try again in a minute.\n\n{e}")
            return

    # Trigger background ML training (non-blocking)
    _ensure_training_thread(player_id, gl)

    # Layout
    c1, c2 = st.columns([1,3])
    with c1:
        st.image(headshot_url(player_id), width=260)

    if gl is None or gl.empty:
        st.info("No games found this season.")
        return

    # Current season averages
    want = ["PTS","REB","AST","FG3M","MIN","FGA","FG3A","FTA","TOV"]
    present = [w for w in want if w in gl.columns]
    season_avg = gl[present].mean(numeric_only=True).to_dict()
    _metric_row("Current Season Averages", season_avg)

    # Last game
    last_game_vals = {}
    if not gl.empty:
        lg = gl.loc[0]
        for k in ("PTS","REB","AST","FG3M","MIN"):
            if k in gl.columns:
                last_game_vals[k] = float(lg.get(k))
    _metric_row("Last Game", last_game_vals)

    # Last 5 averages
    last5_vals = gl.head(5)[present].mean(numeric_only=True).to_dict()
    _metric_row("Last 5 Games Averages", last5_vals)

    # Prediction: ML if ready, else WMA
    preds, used_ml = predict_with_ml_if_ready(player_id, gl)
    label = "Predicted Next Game (Machine Learning)" if used_ml else "Predicted Next Game (Weighted Avg)"
    _metric_row(label, preds)

    # Table: last 10
    st.subheader("Breakdown (Last 10 Games)")
    show_cols = ["GAME_DATE","MATCHUP","WL","PTS","REB","AST","FG3M","MIN"]
    show_cols = [c for c in show_cols if c in gl.columns]
    st.dataframe(gl.head(10)[show_cols], width='stretch')

# -------------- Main router -------------------
def main():
    sidebar_search()
    q = st.query_params
    pid = q.get("player_id", [None])
    if isinstance(pid, list):
        pid = pid[0]
    if pid:
        try:
            page_player(int(pid))
        except Exception:
            page_home()
    else:
        page_home()

if __name__ == "__main__":
    main()