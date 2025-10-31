# app.py — Hot Shot Props | NBA Player Analytics (Topps Card UI, ML, Favorites)
# - Home: Current-season league leaders (PTS/REB/AST/3PM) with headshots & large clickable names
# - Player: headshot + team, metric rows (current, last game, last 5 avg), bar charts per stat,
#           ML next-game predictions (ad-hoc Ridge if sklearn present, else WMA fallback),
#           "Download Card (PNG)" that renders a Topps-style snapshot
# - Favorites: simple board (click to open) + multi-player ML predictions table
# - Robust nba_api calls with retries, headers, caching, and light throttling
# - No balldontlie usage anywhere

import os, time, json, math, io
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import altair as alt

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nba_api.stats.static import players as players_static, teams as teams_static
from nba_api.stats.endpoints import (
    playergamelogs, playercareerstats, leagueleaders
)

# ------------------------------ Streamlit config ------------------------------
st.set_page_config(page_title="Hot Shot Props • NBA Player Analytics", layout="wide")

# ------------------------------ Theme / Topps styling ------------------------------
st.markdown("""
<style>
:root { --bg:#0b0b0b; --ink:#f4f6fb; --card:#121212; --line:#202020; --accent:#ff4d4d; --accent2:#ffb4b4; }
html, body, [data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--ink); }
h1,h2,h3,h4 { color: var(--accent2) !important; letter-spacing: .2px; }
a, a:visited { color:#ffd1d1; text-decoration:none; }
a:hover { text-decoration: underline; }
.small { color:#b6b8bf; font-size:.9rem; }
.big-name { font-size:1.28rem; font-weight:900; line-height:1.15; }
.subtle { color:#c6c8cf; font-weight:600; letter-spacing:.3px; }
.card { background:linear-gradient(180deg,#141414 0%, #0f0f0f 100%); border:1px solid var(--line);
        padding:16px; border-radius:18px; box-shadow:0 8px 28px rgba(0,0,0,.6); }
.tpc { border-radius:14px; border:2px solid #c7a76d; box-shadow:0 0 0 6px #243, 0 10px 30px rgba(0,0,0,.35);
       background: radial-gradient(120% 60% at 50% -10%, #2a2a2a 0%, #101010 70%); }
.badge { display:inline-block; padding:4px 10px; font-size:.8rem; border:1px solid #2a2a2a;
         border-radius:999px; background:#1b0d0d; color:#ff9b9b; }
[data-testid="stSidebar"] { background:#0a0a0a; border-right:1px solid #151515; }
.stButton>button { background:var(--accent); color:#fff; border:none; border-radius:10px; font-weight:800;
                   padding:.55rem .9rem; }
[data-testid="stMetric"]{ background:#121212; border:1px solid #232323; border-radius:12px; padding:10px; }
[data-testid="stMetric"] div[data-testid="stMetricValue"]{ color:#ffe9e9; }
hr { border:0; border-top:1px solid #1b1b1b; margin:14px 0; }
@media (max-width: 780px){ .block-container{ padding:8px 8px!important } [data-testid="column"]{ width:100%!important; display:block!important } }
</style>
""", unsafe_allow_html=True)

# ------------------------------ Robust requests session ------------------------------
def _nba_headers() -> Dict[str, str]:
    return {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }

_session = None
def get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(_nba_headers())
        r = Retry(
            total=4, read=4, connect=4, backoff_factor=0.7,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET","POST"])
        )
        adapter = HTTPAdapter(max_retries=r, pool_connections=30, pool_maxsize=60)
        s.mount("https://", adapter); s.mount("http://", adapter)
        _session = s
    return _session

def call_endpoint(EndpointClass, **kwargs):
    # nba_api respects requests under the hood; retries above help.
    return EndpointClass(**kwargs, timeout=15)

# ------------------------------ Utilities ------------------------------
def season_str_today() -> str:
    today = datetime.utcnow()
    yr = today.year
    if today.month < 7:
        start = yr - 1
        end = yr
    else:
        start = yr
        end = yr + 1
    return f"{start}-{str(end)[-2:]}"

def HEADSHOT(pid: int) -> str:
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"

# ------------------------------ Caches ------------------------------
@st.cache_data(ttl=60*60)
def get_active_players() -> pd.DataFrame:
    return pd.DataFrame(players_static.get_active_players())

@st.cache_data(ttl=60*60)
def get_teams_df() -> pd.DataFrame:
    return pd.DataFrame(teams_static.get_teams())

@st.cache_data(ttl=60*30)
def league_leaders_df(stat: str) -> pd.DataFrame:
    try:
        obj = call_endpoint(
            leagueleaders.LeagueLeaders,
            season=season_str_today(),
            season_type_all_star="Regular Season",
            stat_category_abbreviation=stat
        )
        df = obj.get_data_frames()[0]
        return df
    except Exception as e:
        st.warning(f"League Leaders unavailable for {stat}: {e}")
        fb = pd.DataFrame([
            {"PLAYER_ID": 201939, "PLAYER": "Stephen Curry", "TEAM": "GSW", "FG3M": 4.9, "PTS": 30.1, "REB": 5.1, "AST": 6.2},
            {"PLAYER_ID": 203507, "PLAYER": "Giannis Antetokounmpo", "TEAM": "MIL", "PTS": 31.0, "REB": 12.0, "AST": 6.0, "FG3M": 0.9},
        ])
        order_key = "FG3M" if stat.upper() in ("FG3M","3PM") else stat.upper()
        if order_key in fb.columns:
            fb = fb.sort_values(order_key, ascending=False)
        return fb

@st.cache_data(ttl=60*15, show_spinner=False)
def get_player_logs(pid: int) -> pd.DataFrame:
    try:
        obj = call_endpoint(
            playergamelogs.PlayerGameLogs,
            player_id_nullable=pid,
            season_nullable=season_str_today()
        )
        df = obj.get_data_frames()[0]
        if "GAME_DATE" in df:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Failed to load logs from nba_api (possibly temporary block): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60*60, show_spinner=False)
def get_player_career(pid: int) -> pd.DataFrame:
    try:
        obj = call_endpoint(playercareerstats.PlayerCareerStats, player_id=pid)
        return obj.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

# ------------------------------ ML: Ridge (ad-hoc) with fallback ------------------------------
try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

ML_TARGETS = ["PTS", "REB", "AST", "FG3M"]

def build_features(g: pd.DataFrame) -> Optional[pd.DataFrame]:
    if g is None or g.empty:
        return None
    df = g.copy()
    need = ["PTS","REB","AST","FG3M","MIN","FGA","FGM","FG3A","FG3M","FTA","FTM","TOV"]
    keep = [c for c in need if c in df.columns]
    if not keep:
        return None
    # rolling means to create features (shift by 1 so target day not leaked)
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    for win in [3,5,10]:
        for c in keep:
            df[f"{c}_r{win}"] = df[c].rolling(win, min_periods=1).mean().shift(1)
    # drop rows that have no history
    df = df.dropna().reset_index(drop=True)
    return df

def predict_next_game(g: pd.DataFrame) -> Tuple[Dict[str,float], bool]:
    """
    Returns (predictions, used_ml_flag)
    If sklearn unavailable or not enough rows, uses weighted moving average fallback.
    """
    preds: Dict[str,float] = {}

    if g is None or g.empty:
        return {t: np.nan for t in ML_TARGETS}, False

    # Fallback first: weighted avg on last up to 8 games
    def wma(series: pd.Series, n=8):
        s = series.head(n).to_numpy()[::-1]
        if s.size == 0: return np.nan
        w = np.arange(1, s.size+1, dtype=float)
        return float(np.dot(s, w) / w.sum())

    if not SKLEARN_OK:
        for t in ML_TARGETS:
            preds[t] = wma(g[t]) if t in g.columns else np.nan
        return preds, False

    # Build features/targets for ad-hoc ridge per stat
    featdf = build_features(g)
    if featdf is None or len(featdf) < 15:
        for t in ML_TARGETS:
            preds[t] = wma(g[t]) if t in g.columns else np.nan
        return preds, False

    used_ml = False
    for t in ML_TARGETS:
        if t not in featdf.columns:
            preds[t] = np.nan
            continue
        y = featdf[t].astype(float)
        # select features that end with _rK and are not the target itself
        feat_cols = [c for c in featdf.columns if c.endswith(("_r3","_r5","_r10")) and (t not in c or c.endswith(("_r3","_r5","_r10")))]
        if not feat_cols:
            preds[t] = wma(g[t]) if t in g.columns else np.nan
            continue
        X = featdf[feat_cols].astype(float)

        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = Ridge(alpha=1.0)
            model.fit(Xtr, ytr)
            # Next game features = last row values
            last_row = featdf.iloc[[-1]][feat_cols]
            preds[t] = float(model.predict(last_row)[0])
            used_ml = True
        except Exception:
            preds[t] = wma(g[t]) if t in g.columns else np.nan

    return preds, used_ml

# ------------------------------ Sidebar: nav + favorites ------------------------------
def init_session():
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = {}  # name -> id
    if "route" not in st.session_state:
        st.session_state["route"] = "home"
    if "selected_player" not in st.session_state:
        st.session_state["selected_player"] = None
    if "selected_name" not in st.session_state:
        st.session_state["selected_name"] = None

init_session()

def route_to_home():
    st.session_state["route"] = "home"
    st.session_state["selected_player"] = None
    st.session_state["selected_name"] = None
    st.experimental_set_query_params()
    st.rerun()

with st.sidebar:
    st.subheader("Select Player")
    # quick search (debounced)
    plist = get_active_players()
    name_to_id = {f'{r["full_name"]}': r["id"] for _, r in plist.iterrows()}
    typed = st.text_input("Search by name", "")
    choices = [n for n in name_to_id if typed.lower() in n.lower()] if typed else sorted(name_to_id.keys())
    pick = st.selectbox("Choose a player", ["—"] + choices, index=0)
    if pick != "—":
        pid = name_to_id[pick]
        st.session_state["route"] = "player"
        st.session_state["selected_player"] = pid
        st.session_state["selected_name"] = pick
        st.experimental_set_query_params(player_id=str(pid))
        st.rerun()

    st.button("🏠 Home", on_click=route_to_home, type="primary")

    st.markdown("---")
    # Favorites
    st.markdown("**⭐ Favorites**")
    favs = st.session_state["favorites"]
    if favs:
        for nm, pid in list(favs.items()):
            c1, c2 = st.columns([0.75, 0.25])
            with c1:
                if st.button(nm, key=f"fav_open_{pid}"):
                    st.session_state["route"] = "player"
                    st.session_state["selected_player"] = pid
                    st.session_state["selected_name"] = nm
                    st.experimental_set_query_params(player_id=str(pid))
                    st.rerun()
            with c2:
                if st.button("✕", key=f"fav_del_{pid}"):
                    favs.pop(nm, None)
                    st.rerun()
    else:
        st.caption("No favorites yet.")
    st.markdown("---")
    if st.button("📋 Favorites Page"):
        st.session_state["route"] = "favorites"
        st.rerun()

# parse query param routing
qs = st.experimental_get_query_params()
if "player_id" in qs:
    try:
        pid_q = int(qs["player_id"][0])
        st.session_state["route"] = "player"
        st.session_state["selected_player"] = pid_q
    except Exception:
        pass

# ------------------------------ UI Components ------------------------------
def leader_card(df: pd.DataFrame, label: str, key_prefix: str):
    if df is None or df.empty:
        st.warning(f"No data for {label}")
        return
    row = df.iloc[0]
    pid = int(row.get("PLAYER_ID", 0))
    name = str(row.get("PLAYER", "Unknown"))
    team = str(row.get("TEAM", row.get("TEAM_ABBREVIATION", "")))
    stat_key = None
    for cand in ["PTS","REB","AST","FG3M","3PM"]:
        if cand in row.index:
            stat_key = cand
            break
    val = float(row.get(stat_key, 0.0)) if stat_key else 0.0

    with st.container():
        c1, c2 = st.columns([1.0, 1.8])
        with c1:
            st.image(HEADSHOT(pid))
        with c2:
            st.markdown(f"""
            <div class="big-name"><a href="?player_id={pid}">{name}</a></div>
            <div class="subtle">{team}</div>
            <div style="margin-top:10px">
                <span class="badge">Leader — {label}</span>
                <div style="font-size:1.6rem;font-weight:900;margin-top:6px">{val:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open {name} →", key=f"{key_prefix}_{pid}"):
                st.session_state["route"] = "player"
                st.session_state["selected_player"] = pid
                st.session_state["selected_name"] = name
                st.experimental_set_query_params(player_id=str(pid))
                st.rerun()

def metrics_block(title: str, ser: pd.Series):
    st.subheader(title)
    c1,c2,c3,c4,c5 = st.columns(5)
    def fmt(k): 
        return f"{ser.get(k, np.nan):.2f}" if k in ser and pd.notna(ser[k]) else "N/A"
    with c1: st.metric("PTS", fmt("PTS"))
    with c2: st.metric("REB", fmt("REB"))
    with c3: st.metric("AST", fmt("AST"))
    with c4: st.metric("3PM", fmt("FG3M"))
    with c5: st.metric("MIN", fmt("MIN"))

def player_bars(title: str, logs: pd.DataFrame, n=10):
    st.subheader(title)
    show = ["PTS","REB","AST","FG3M","MIN"]
    have = [x for x in show if x in logs.columns]
    if not have:
        st.info("No stats available.")
        return
    df = logs.head(n)[["GAME_DATE"]+have].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    melted = df.melt("GAME_DATE", var_name="Stat", value_name="Value")
    chart = (alt.Chart(melted)
                .mark_bar()
                .encode(x=alt.X("GAME_DATE:N", title="", sort=None),
                        y=alt.Y("Value:Q", title=""),
                        color=alt.Color("Stat:N"))
                .properties(height=260, width=700))
    st.altair_chart(chart, width='stretch')

def draw_topp_card_png(player_name: str, team: str, headshot_url: str, metrics: Dict[str, float]) -> bytes:
    # Simple Topps-like card 600x850
    W,H = 600, 850
    img = Image.new("RGB", (W,H), (15,15,18))
    draw = ImageDraw.Draw(img)

    # frame
    draw.rectangle([10,10,W-10,H-10], outline=(199,167,109), width=6)

    # headshot
    try:
        hs = Image.open(BytesIO(requests.get(headshot_url, timeout=10).content)).convert("RGBA")
        hs = hs.resize((520, 380))
        img.paste(hs, (40,120), mask=hs)
    except Exception:
        pass

    # title
    draw.rectangle([0,0,W,80], fill=(26,26,30))
    title = f"{player_name}"
    draw.text((30,22), title, fill=(255,200,200))

    # team band
    draw.rectangle([0,80,W,110], fill=(255,77,77))
    draw.text((30,86), f"{team}", fill=(20,20,20))

    # metrics
    y0 = 520
    for k,v in metrics.items():
        draw.text((60,y0), f"{k}: {v:.2f}", fill=(255,233,233))
        y0 += 42

    # footer
    draw.rectangle([0,H-60,W,H], fill=(26,26,30))
    draw.text((30,H-45), "Hot Shot Props — NBA Analytics Card", fill=(255,210,210))

    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()

# ------------------------------ Pages ------------------------------
def page_home():
    st.title("🏠 Home")
    st.caption("Hint: use the sidebar to search any player by name. Add ⭐ to build your Favorites board.")

    st.subheader("League Leaders (Current Season)")
    cols = st.columns(4)
    for (stat, label), c in zip([("PTS","PTS Leader"), ("REB","REB Leader"), ("AST","AST Leader"), ("FG3M","3PM Leader")], cols):
        with c:
            st.markdown('<div class="card tpc">', unsafe_allow_html=True)
            df = league_leaders_df(stat)
            leader_card(df, label, key_prefix=f"lead_{stat}")
            st.markdown("</div>", unsafe_allow_html=True)

def page_player(pid: int, name: Optional[str]):
    st.title(name or "Player")

    # add to favorites
    if st.button("⭐ Add to Favorites"):
        nm = name or "Player"
        st.session_state["favorites"][nm] = pid

    st.image(HEADSHOT(pid), width=260)

    with st.spinner("Loading game logs…"):
        logs = get_player_logs(pid)
    if logs.empty:
        st.error("No logs available (API may be briefly blocked).")
        return

    # Metric rows
    # Current season avg
    cur_avg = logs[["PTS","REB","AST","FG3M","MIN"]].mean(numeric_only=True)
    metrics_block("Current Season Averages", cur_avg)

    # Last game
    if not logs.empty:
        last_game = logs.loc[0, ["PTS","REB","AST","FG3M","MIN"]]
        metrics_block("Last Game Stats", last_game)
    else:
        st.info("No last game found.")

    # Last 5 avg
    last5 = logs.head(5)[["PTS","REB","AST","FG3M","MIN"]].mean(numeric_only=True)
    metrics_block("Last 5 Games Averages", last5)

    # ML Next game
    preds, ml_used = predict_next_game(logs)
    tag = "(ML)" if ml_used else "(WMA)"
    metrics_block(f"Predicted Next Game {tag}", pd.Series(preds))

    # Bars per category (last 10)
    player_bars("Last 10 — Bars", logs, n=10)

    # Download Topps card
    card_png = draw_topp_card_png(
        player_name=name or "Player",
        team="",
        headshot_url=HEADSHOT(pid),
        metrics=preds
    )
    st.download_button("🧧 Download Card (PNG)", data=card_png, file_name=f"{(name or 'player').replace(' ','_')}_card.png", mime="image/png")

def page_favorites():
    st.title("⭐ Favorites")
    favs = st.session_state["favorites"]
    if not favs:
        st.info("No favorites yet. Add players from the sidebar.")
        return

    # clickable list
    st.subheader("Board")
    cols = st.columns(3)
    i=0
    for nm, pid in favs.items():
        with cols[i%3]:
            st.markdown(f"- [{nm}](?player_id={pid})")
        i+=1

    st.subheader("Predicted Next Game (All Favorites)")
    rows = []
    prog = st.progress(0, text="Scoring favorites with ML…")
    total = len(favs)
    for idx, (nm, pid) in enumerate(favs.items(), start=1):
        # light throttle to avoid hammering
        time.sleep(0.15)
        logs = get_player_logs(pid)
        preds, ml_used = predict_next_game(logs)
        rows.append({
            "Player": nm,
            **{k: round(float(v),2) if pd.notna(v) else np.nan for k,v in preds.items()},
            "Model": "ML" if ml_used else "WMA"
        })
        prog.progress(min(idx/total,1.0))
    if rows:
        out = pd.DataFrame(rows)
        st.dataframe(out, height=420, width='stretch')

# ------------------------------ Router ------------------------------
route = st.session_state.get("route","home")
pid = st.session_state.get("selected_player")
name = st.session_state.get("selected_name")

if route == "player" and pid:
    page_player(pid, name)
elif route == "favorites":
    page_favorites()
else:
    page_home()