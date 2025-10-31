# app.py — Hot Shot Props | NBA Player Analytics (NBA-only, ML + Expanders)
# - nba_api with fast timeout + cdn.nba.com mirror fallback
# - ML: per-player Ridge model for PTS/REB/AST/FG3M (this season)
# - Fallback to weighted average when ML not available
# - Expanders with bar charts for all major stat categories
# - Home: league leaders (click to open player page)
# - Sidebar: Home, search, favorites

import os, json, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from nba_api.stats.static import players
from nba_api.stats.endpoints import (
    leagueleaders, playercareerstats, playergamelogs
)

# -------------------- Streamlit page & theme --------------------
st.set_page_config(page_title="Hot Shot Props • NBA Analytics", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {background:#000;color:#f4f4f4;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#000 0%,#111 100%)!important;}
h1,h2,h3,h4,h5 {color:#ff5555;font-weight:700;}
.hr{border:0;border-top:1px solid #222;margin:.6rem 0;}
[data-testid="stMetric"] {background:#111;border-radius:12px;padding:10px;border:1px solid #222;}
[data-testid="stMetric"] label{color:#ff7777;}
[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#fff;font-size:1.35em;}
.leader-card{display:flex;align-items:center;gap:14px;background:#0d0d0d;border:1px solid #222;
padding:10px;border-radius:10px;margin-bottom:8px;}
.leader-img img{width:55px;height:55px;border-radius:8px;}
.leader-info{display:flex;flex-direction:column;}
.leader-info a{color:#ffb4b4;text-decoration:none;font-weight:bold;}
.leader-stat{color:#ccc;font-size:0.9em;}
.hint{background:#0e0e0e;border:1px solid #222;border-radius:10px;padding:.75rem 1rem;color:#ddd;}
.small{font-size:.9rem;color:#bbb;}
</style>
""", unsafe_allow_html=True)

# -------------------- Season helper --------------------
def current_season():
    today = dt.date.today()
    y = today.year if today.month >= 10 else today.year - 1
    return f"{y}-{str(y+1)[-2:]}"
SEASON = current_season()

# -------------------- nba_api robustness --------------------
try:
    from nba_api.stats.library.http import NBAStatsHTTP
    NBAStatsHTTP._COMMON_HEADERS.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/123.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    })
    NBAStatsHTTP._session = None
except Exception:
    NBAStatsHTTP = None  # type: ignore

NBA_TIMEOUT = 5
NBA_RETRIES = 2
NBA_BACKOFF = 0.35

def _nba_call(endpoint_cls, **kwargs):
    last_err = None
    for i in range(NBA_RETRIES + 1):
        try:
            return endpoint_cls(timeout=NBA_TIMEOUT, **kwargs)
        except Exception as e:
            last_err = e
            if i < NBA_RETRIES:
                time.sleep(NBA_BACKOFF * (2**i) + 0.1*np.random.rand())
    if NBAStatsHTTP is not None:
        try:
            orig = getattr(NBAStatsHTTP, "_BASE_URL", "https://stats.nba.com/stats")
            setattr(NBAStatsHTTP, "_BASE_URL", "https://cdn.nba.com/stats")
            NBAStatsHTTP._session = None
            try:
                return endpoint_cls(timeout=NBA_TIMEOUT, **kwargs)
            finally:
                setattr(NBAStatsHTTP, "_BASE_URL", orig)
                NBAStatsHTTP._session = None
        except Exception as e2:
            last_err = e2
    raise RuntimeError(f"NBA data fetch failed after retries: {last_err}")

# -------------------- Favorites persistence --------------------
DEFAULT_TMP = "/tmp" if os.access("/", os.W_OK) else "."
FAV_PATH = os.path.join(DEFAULT_TMP, "favorites.json")

def load_favorites():
    try:
        if os.path.exists(FAV_PATH):
            with open(FAV_PATH, "r") as f: return json.load(f)
    except Exception:
        pass
    return []

def save_favorites(favs):
    try:
        with open(FAV_PATH, "w") as f: json.dump(sorted(set(favs)), f)
    except Exception:
        pass

if "favorites" not in st.session_state:
    st.session_state.favorites = load_favorites()

# -------------------- Active players --------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_active_maps():
    aps = players.get_active_players()
    id2name = {p["id"]: p["full_name"] for p in aps}
    name2id = {v: k for k, v in id2name.items()}
    names_sorted = sorted(name2id.keys())
    return aps, id2name, name2id, names_sorted

ACTIVE_PLAYERS, ID_TO_NAME, NAME_TO_ID, PLAYER_NAMES = get_active_maps()

# -------------------- Cached fetchers --------------------
@st.cache_data(ttl=900, show_spinner=False)
def get_league_leaders_df(season: str) -> pd.DataFrame:
    return _nba_call(leagueleaders.LeagueLeaders,
                     season=season, per_mode48="PerGame").get_data_frames()[0]

@st.cache_data(ttl=900, show_spinner=False)
def get_player_career_df(player_id: int) -> pd.DataFrame:
    return _nba_call(playercareerstats.PlayerCareerStats,
                     player_id=player_id).get_data_frames()[0]

@st.cache_data(ttl=900, show_spinner=False)
def get_player_gamelogs_df(player_id: int, season: str) -> pd.DataFrame:
    gl = _nba_call(playergamelogs.PlayerGameLogs,
                   player_id_nullable=player_id,
                   season_nullable=season,
                   season_type_nullable="Regular Season").get_data_frames()[0]
    if "GAME_DATE" in gl.columns:
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        gl = gl.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    return gl

# -------------------- ML (Ridge) --------------------
# Optional dependencies
try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    Ridge = None  # type: ignore

ML_STATS = ["PTS", "REB", "AST", "FG3M"]
ML_FEATS = ["PTS","REB","AST","FG3M","MIN","FGA","FG3A","FTA","OREB","DREB","TOV"]

@st.cache_data(ttl=60*60, show_spinner=False)  # 1 hour
def build_player_models(player_id: int, season: str):
    """
    Train small Ridge models per stat using this season only.
    Features: rolling 1/3/5 of core box-score cols.
    """
    if not SKLEARN_OK:
        return None  # trigger fallback
    gl = get_player_gamelogs_df(player_id, season)
    if gl is None or gl.empty or len(gl) < 8:
        return None
    df = gl.copy().sort_values("GAME_DATE")  # chronological for rolling
    # keep only numeric cols we care about
    use_cols = [c for c in ML_FEATS if c in df.columns]
    if not use_cols:
        return None
    # rolling features
    for c in use_cols:
        s = df[c].astype(float)
        df[f"{c}_r1"] = s.rolling(1).mean()
        df[f"{c}_r3"] = s.rolling(3).mean()
        df[f"{c}_r5"] = s.rolling(5).mean()
    # For each target stat, shift target by -1 (predict next game)
    models = {}
    feat_cols = [c for c in df.columns if any(s in c for s in [*_ for _ in ML_FEATS]) and ("_r" in c)]
    for target in ML_STATS:
        if target not in df.columns: 
            continue
        t = df[target].astype(float).shift(-1)
        X = df[feat_cols].iloc[:-1].fillna(method="bfill").fillna(0.0).values
        y = t.iloc[:-1].fillna(method="bfill").fillna(0.0).values
        if len(y) < 6: 
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Ridge(alpha=1.0).fit(Xtr, ytr)
        models[target] = {"model": model, "feat_cols": feat_cols}
    return models if models else None

def predict_next_game_ml(player_id: int, season: str):
    models = build_player_models(player_id, season)
    if not models:
        return None, False
    gl = get_player_gamelogs_df(player_id, season)
    if gl is None or gl.empty:
        return None, False
    df = gl.copy().sort_values("GAME_DATE")
    # last row features (most recent game)
    for c in [c for c in ML_FEATS if c in df.columns]:
        s = df[c].astype(float)
        df[f"{c}_r1"] = s.rolling(1).mean()
        df[f"{c}_r3"] = s.rolling(3).mean()
        df[f"{c}_r5"] = s.rolling(5).mean()
    last = df.iloc[-1:]
    preds = {}
    for stat, pack in models.items():
        X = last[pack["feat_cols"]].fillna(method="bfill").fillna(0.0).values
        preds[stat] = float(pack["model"].predict(X)[0])
    return preds, True

def predict_weighted(gamelogs: pd.DataFrame):
    if gamelogs is None or gamelogs.empty:
        return None
    n = min(10, len(gamelogs))
    w = np.arange(n, 0, -1)
    p = {}
    for s in ML_STATS:
        if s in gamelogs.columns:
            vals = gamelogs[s].head(n).astype(float).values
            p[s] = float(np.average(vals, weights=w)) if len(vals) else np.nan
        else:
            p[s] = np.nan
    return p

# -------------------- Sidebar --------------------
def go_home():
    st.session_state.pop("selected_player", None)
    try:
        st.query_params.clear()
    except Exception:
        pass
    st.rerun()

with st.sidebar:
    st.button("🏠 Home Screen", on_click=go_home, type="primary", key="home_btn")
    st.markdown("---")
    st.header("Search Player")
    search_name = st.selectbox("Player", PLAYER_NAMES, index=None, placeholder="Select player")
    st.markdown("### ⭐ Favorites")
    if not st.session_state["favorites"]:
        st.caption("No favorites yet.")
    for fav in list(st.session_state["favorites"]):
        c1, c2 = st.columns([4,1])
        if c1.button(fav, key=f"fav_{fav}"):
            st.session_state["selected_player"] = fav
            st.rerun()
        if c2.button("❌", key=f"rm_{fav}"):
            st.session_state["favorites"].remove(fav)
            save_favorites(st.session_state["favorites"])
            st.rerun()

# direct-link support
try:
    pid_param = st.query_params.get("player_id")
    if pid_param:
        if isinstance(pid_param, (list, tuple)): pid_param = pid_param[0]
        pid = int(pid_param)
        if pid in ID_TO_NAME:
            st.session_state["selected_player"] = ID_TO_NAME[pid]
except Exception:
    pass

# -------------------- Home --------------------
def show_home():
    st.title("🏀 NBA League Leaders")
    st.subheader(f"Season {SEASON}")
    st.markdown("<div class='hint'>💡 Use the sidebar to search any player, or click a leader to open their page.</div>", unsafe_allow_html=True)
    try:
        leaders = get_league_leaders_df(SEASON)
        for stat in ["PTS","REB","AST","STL","BLK","FG3M"]:
            top = leaders.sort_values(stat, ascending=False).iloc[0]
            player_id = int(top["PLAYER_ID"])
            href = f"?player_id={player_id}"
            st.markdown(f"""
            <div class='leader-card'>
              <div class='leader-img'>
                <img src='https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png'>
              </div>
              <div class='leader-info'>
                <a href='{href}'>{top["PLAYER"]}</a>
                <div>{top["TEAM"]}</div>
                <div class='leader-stat'>{stat}: <b>{round(float(top[stat]),2)}</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load league leaders: {e}")

if "selected_player" not in st.session_state and not search_name:
    show_home()
    st.stop()

# -------------------- Player Detail --------------------
selected = search_name or st.session_state.get("selected_player")
player_id = NAME_TO_ID.get(selected)
try:
    if player_id:
        st.query_params.update({"player_id": str(player_id)})
    else:
        st.query_params.clear()
except Exception:
    pass

st.title(f"📊 {selected}")

if st.button("⭐ Add to Favorites"):
    if selected not in st.session_state["favorites"]:
        st.session_state["favorites"].append(selected)
        save_favorites(st.session_state["favorites"])
        st.success(f"Added {selected} to favorites")

spinner = st.empty()
spinner.info("Loading player data…")

def load_player_fast(pid: int):
    def _career():  return get_player_career_df(pid)
    def _glogs():   return get_player_gamelogs_df(pid, SEASON)
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_career): "career", ex.submit(_glogs): "logs"}
        career_df, glogs = None, None
        for fut in as_completed(futs):
            kind = futs[fut]
            data = fut.result()
            if kind == "career": career_df = data
            else: glogs = data
    return career_df, glogs

if player_id is None:
    spinner.empty()
    st.error("Player not found.")
    st.stop()

try:
    career_df, gamelogs = load_player_fast(player_id)
except Exception as e:
    spinner.empty()
    st.error(f"Failed to load from nba_api (temporary block?): {e}")
    st.stop()

spinner.empty()

# Headshot
st.image(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png", width=230)

# Career small table
st.subheader("Career Averages (Per Game)")
if not career_df.empty:
    df = career_df.copy()
    if "GP" in df.columns:
        gp = df["GP"].replace(0, np.nan)
        for c in ("PTS","REB","AST"):
            if c in df.columns:
                df[c] = (df[c] / gp).round(2)
    cols = [c for c in ["SEASON_ID","TEAM_ABBREVIATION","PTS","REB","AST"] if c in df.columns]
    st.dataframe(df[cols].fillna(0), width='stretch')
else:
    st.info("No career/per-season data.")

# ---- Last game + last-5 averages
st.subheader("Last Game")
if isinstance(gamelogs, pd.DataFrame) and not gamelogs.empty:
    last = gamelogs.iloc[0]
    c = st.columns(4)
    for i, s in enumerate(["PTS","REB","AST","FG3M"]):
        v = last.get(s, np.nan)
        c[i].metric(s, "N/A" if pd.isna(v) else int(v))
else:
    st.info("No recent game found.")

st.subheader("Recent Form (Last 5)")
if isinstance(gamelogs, pd.DataFrame) and len(gamelogs) >= 5:
    avg = gamelogs.head(5).mean(numeric_only=True)
    c = st.columns(4)
    for i, s in enumerate(["PTS","REB","AST","FG3M"]):
        v = avg.get(s, np.nan)
        c[i].metric(s, "N/A" if pd.isna(v) else round(float(v),2))
else:
    st.info("Not enough games.")

# ---- Prediction (ML first; fallback weighted)
st.subheader("Predicted Next Game")
ml_preds, used_ml = predict_next_game_ml(player_id, SEASON)
if not ml_preds:
    ml_preds = predict_weighted(gamelogs)
    used_ml = False

if ml_preds:
    c = st.columns(4)
    for i, s in enumerate(["PTS","REB","AST","FG3M"]):
        v = ml_preds.get(s, np.nan)
        label = f"{s} {'(ML)' if used_ml else '(WA)'}"
        c[i].metric(label, "N/A" if pd.isna(v) else f"{v:.1f}")
    st.caption(("**(ML)** = Ridge regression on this season’s logs • "
                "**(WA)** = weighted average fallback"), help="ML uses rolling 1/3/5-game features.")
else:
    st.info("No data to generate a prediction.")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# -------------------- Expanders with bar charts --------------------
def chart_last10(df: pd.DataFrame, cols: list, title: str):
    if df is None or df.empty: 
        st.info("No recent games.")
        return
    keep = ["GAME_DATE"] + [c for c in cols if c in df.columns]
    sub = df[keep].head(10).copy()
    long = sub.melt("GAME_DATE", var_name="Stat", value_name="Value")
    ch = (alt.Chart(long)
          .mark_bar()
          .encode(x="Stat:N", y="mean(Value):Q", color="Stat:N")
          .properties(height=280, width=850))
    st.altair_chart(ch, theme=None)

with st.expander("📈 Scoring (PTS • usage)", expanded=False):
    chart_last10(gamelogs, ["PTS","MIN","FGA","FTA"], "Scoring")

with st.expander("🎯 Shooting Splits (FG/3PT/FT volume)", expanded=False):
    chart_last10(gamelogs, ["FGM","FGA","FG3M","FG3A","FTM","FTA"], "Shooting")

with st.expander("🏀 Rebounding", expanded=False):
    chart_last10(gamelogs, ["OREB","DREB","REB"], "Rebounding")

with st.expander("🧠 Playmaking / Control", expanded=False):
    chart_last10(gamelogs, ["AST","TOV"], "Playmaking")

with st.expander("🛡️ Defense / Discipline", expanded=False):
    chart_last10(gamelogs, ["STL","BLK","PF"], "Defense")

with st.expander("💥 3-Point Production", expanded=False):
    chart_last10(gamelogs, ["FG3M"], "3-Point")

st.markdown("---")
st.caption("Hot Shot Props • NBA Analytics Dashboard ©2025")
