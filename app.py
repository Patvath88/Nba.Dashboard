# Hot Shot Props — NBA Player Analytics (Topps Card Edition)
# Streamlit single file app: Home (league leaders), Player page, Favorites page
# Robust NBA Stats fetching with retries/backoff/caching. ML predictions (Ridge).
# Theme: black/red Topps card styling. Snapshot PNG export.

import os, time, re, io, json
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import streamlit as st
import streamlit.components.v1 as components

# ------------------- Page config & theme -------------------
st.set_page_config(page_title="Hot Shot Props — NBA Card View", layout="wide")

components.html(
    """<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">""",
    height=0
)

TOPPS_CSS = """
<style>
:root { --bg:#000; --panel:#0c0c0c; --ink:#f3f4f6; --line:#181818; --red:#ef4444; --pink:#ffb4b4; }
html, body, [data-testid="stAppViewContainer"] { background:var(--bg)!important; color:var(--ink)!important; }
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#000 0%,#0b0b0b 100%)!important; border-right:1px solid #111; }

h1,h2,h3,h4 { color:#ffb4b4!important; letter-spacing:.2px; }
.small { font-size:.9rem; color:#cbd5e1; }
.badge { display:inline-block; padding:3px 10px; border-radius:999px; border:1px solid #222; background:#130808; color:#fca5a5; }
.hr { border:0; border-top:1px solid var(--line); margin:10px 0 16px; }

/* Buttons */
.stButton>button { background:var(--red)!important; color:#fff!important; border:none!important; border-radius:10px!important;
  padding:.55rem .95rem!important; font-weight:700!important; }

/* Metrics */
[data-testid="stMetric"]{ background:#101010; border:1px solid #1a1a1a; border-radius:14px; padding:12px; }
[data-testid="stMetric"] label{ color:#fda4a4; }
[data-testid="stMetric"] div[data-testid="stMetricValue"]{ color:#ffe4e6; font-size:1.25rem; }

/* "Card" look */
.card {
  background: radial-gradient(1000px 300px at -10% -10%, #3f1d1d55 0%, transparent 60%) , #0e0e0e;
  border: 1px solid #2a2a2a; border-radius: 16px; padding: 14px; box-shadow: 0 14px 40px rgba(0,0,0,.55);
}

/* Leader tiles (bigger headshot) */
.leader { display:flex; align-items:center; gap:12px; padding:10px; border:1px solid #222; border-radius:14px; background:#0f0f0f; }
.leader img { width:92px; height:68px; object-fit:cover; border-radius:10px; border:1px solid #222; }
.leader .name a { font-size:1.15rem; font-weight:800; color:#ffb4b4; text-decoration:none; }
.leader .sub { font-size:.9rem; color:#a1a1aa; }

/* Topps frame around player headshot */
.cardframe { border:2px solid #e11d48; border-radius:16px; padding:8px; background:
  linear-gradient(135deg, #e11d48 0%, #7f1d1d 25%, #0f0f0f 60%); }

/* Mobile */
@media (max-width: 780px){
  .block-container { padding: 0.6rem 0.6rem!important; }
  [data-testid="stMetric"]{ margin-bottom: 6px!important; }
  .leader img { width:84px; height:62px; }
}
</style>
"""
st.markdown(TOPPS_CSS, unsafe_allow_html=True)

# ------------------- Robust HTTP layer -------------------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

NBA_HOST = "https://stats.nba.com"
HEADSHOT_URL = "https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"
TEAM_LOGO = "https://cdn.nba.com/logos/nba/{tid}/global/L/logo.png"

@st.cache_resource
def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5, connect=3, read=3,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    })
    return s

def _get_json(path: str, params: Dict[str, str], timeout: int = 15) -> dict:
    """GET NBA stats JSON with polite delay and backoff-friendly session."""
    time.sleep(0.8)  # be nice
    url = f"{NBA_HOST}{path}"
    r = _session().get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ------------------- Static player/team maps -------------------
from nba_api.stats.static import teams as teams_static, players as players_static
ALL_TEAMS = teams_static.get_teams()
ALL_PLAYERS = players_static.get_players()

TEAM_BY_ID = {t["id"]: t for t in ALL_TEAMS}
TEAM_BY_ABBR = {t["abbreviation"]: t for t in ALL_TEAMS}
PLAYER_BY_ID = {p["id"]: p for p in ALL_PLAYERS}
PLAYER_BY_NAME = {p["full_name"]: p for p in ALL_PLAYERS}

# ------------------- Cache: images -------------------
@st.cache_data(ttl=24*3600)
def fetch_headshot(pid: int) -> Optional[Image.Image]:
    try:
        r = _session().get(HEADSHOT_URL.format(pid=pid), timeout=12)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception:
        pass
    return None

@st.cache_data(ttl=24*3600)
def fetch_team_logo(tid: int) -> Optional[Image.Image]:
    try:
        r = _session().get(TEAM_LOGO.format(tid=tid), timeout=12)
        if r.status_code == 200:
            img = Image.open(io.BytesIO(r.content))
            return img.convert("RGBA") if img.mode != "RGBA" else img
    except Exception:
        pass
    return None

# ------------------- Data: League leaders (Home) -------------------
# Use raw endpoint to avoid library param mismatches. Per Game = PerMode=PerGame
@st.cache_data(ttl=1800)
def league_leader(season: str, stat: str) -> Optional[dict]:
    """
    Returns dict: {player_id, player, team_abbr, value} for top player by stat.
    stat in {"PTS","REB","AST","FG3M"}
    """
    try:
        js = _get_json(
            "/stats/leagueleaders",
            {
                "Season": season,
                "SeasonType": "Regular Season",
                "StatCategory": stat,
                "PerMode": "PerGame",
                "Scope": "S",
                "Active": "Y",
                "LeagueID": "00"
            }
        )
        headers = js["resultSet"]["headers"] if "resultSet" in js else js["resultSets"][0]["headers"]
        rows = js["resultSet"]["rows"] if "resultSet" in js else js["resultSets"][0]["rowSet"]
        df = pd.DataFrame(rows, columns=headers)
        # common columns: PLAYER_ID, PLAYER, TEAM, PTS, REB, AST, FG3M etc.
        df = df.sort_values(stat, ascending=False).reset_index(drop=True)
        top = df.iloc[0]
        return dict(
            player_id=int(top["PLAYER_ID"]),
            player=str(top["PLAYER"]),
            team_abbr=str(top["TEAM"]),
            value=float(top[stat])
        )
    except Exception as e:
        st.error(f"Leaders {stat} unavailable: {e}")
        return None

# ------------------- Data: Player gamelogs -------------------
@st.cache_data(ttl=3600)
def fetch_player_logs(player_id: int, seasons: Tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        try:
            js = _get_json(
                "/stats/playergamelogs",
                {
                    "PlayerID": str(player_id),
                    "Season": season,
                    "SeasonType": "Regular Season",
                    "MeasureType": "Base"
                },
                timeout=15
            )
            rs = js["resultSets"][0]
            df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
            if not df.empty:
                df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
                df["SEASON"] = season
                frames.append(df)
        except Exception as e:
            st.warning(f"Logs unavailable for {season}: {e}")
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.sort_values("GAME_DATE", ascending=False, inplace=True)
        return out
    return pd.DataFrame()

# ------------------- ML model (Ridge) -------------------
try:
    from sklearn.linear_model import Ridge
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

ML_STATS = ["PTS", "REB", "AST", "FG3M"]

def ml_predict_next(df: pd.DataFrame) -> Tuple[Dict[str, float], str]:
    """
    Train a small Ridge model per stat on rolling features and predict next-game stat.
    Returns (preds, tag) where tag is 'ML' or 'Fallback'.
    """
    if not SKLEARN_OK or df is None or df.empty:
        return {}, "Fallback"

    df = df.copy()
    df.sort_values("GAME_DATE", inplace=True)
    for c in ML_STATS:
        if c not in df.columns:
            df[c] = np.nan

    # rolling features
    for k in [5, 10, 20]:
        for c in ML_STATS:
            df[f"{c}_r{k}"] = df[c].rolling(k, min_periods=1).mean()

    # feature set
    feat_cols = [f"{c}_r{k}" for c in ML_STATS for k in [5,10,20]]

    preds = {}
    tag = "ML"
    ok_any = False
    for target in ML_STATS:
        avail = df.dropna(subset=feat_cols+[target]).copy()
        if len(avail) < 10:
            tag = "Fallback"
            continue
        X = avail[feat_cols].values
        y = avail[target].values
        try:
            model = Ridge(alpha=0.6)
            model.fit(X, y)
            last_row = df.iloc[[-1]][feat_cols].values
            preds[target] = float(np.clip(model.predict(last_row)[0], 0, 200))
            ok_any = True
        except Exception:
            tag = "Fallback"
    if not ok_any:
        return {}, "Fallback"
    return preds, tag

# ------------------- Utilities -------------------
def human_team_name(abbr: str) -> str:
    t = TEAM_BY_ABBR.get(abbr.upper())
    return t["full_name"] if t else abbr

def link_to_player(pid: int, label: str) -> str:
    # Generate a link that sets ?player_id=PID
    return f'<a href="?player_id={pid}" style="text-decoration:none;color:#ffb4b4;font-weight:800;">{label}</a>'

def add_query(**params):
    qp = dict(st.query_params)
    qp.update({k:str(v) for k,v in params.items()})
    st.query_params.clear()
    st.query_params.update(qp)

# ------------------- Sidebar: search + favorites + nav -------------------
def sidebar_ui():
    st.sidebar.header("Select Player")
    # Combined search (optional team filter)
    q = st.sidebar.text_input("Search player", placeholder="Type a player's name...")
    team_filter = st.sidebar.selectbox("Filter by team (optional)", ["All"] + sorted([t["full_name"] for t in ALL_TEAMS]))
    names = [p["full_name"] for p in ALL_PLAYERS]
    if q:
        names = [n for n in names if q.lower() in n.lower()]
    if team_filter != "All":
        abbr = None
        for t in ALL_TEAMS:
            if t["full_name"] == team_filter:
                abbr = t["abbreviation"]
                break
        if abbr:
            # Approx filter: players have no team in static, so show all; final selection still works
            pass
    sel = st.sidebar.selectbox("Pick", ["—"] + names)
    if sel and sel != "—":
        pid = PLAYER_BY_NAME[sel]["id"]
        add_query(player_id=pid)
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("⭐ Favorites")
    favs: Dict[str,int] = st.session_state.get("favorites", {})
    if favs:
        for name, pid in list(favs.items()):
            c1, c2 = st.sidebar.columns([0.8,0.2])
            with c1:
                st.sidebar.markdown(f"- [{name}](?player_id={pid})")
            with c2:
                if st.sidebar.button("✕", key=f"del_{pid}"):
                    favs.pop(name, None)
                    st.session_state["favorites"] = favs
                    st.rerun()
    else:
        st.sidebar.caption("No favorites yet.")

    st.sidebar.markdown("---")
    if st.sidebar.button("🏠 Home"):
        st.query_params.clear()
        st.rerun()
    if st.sidebar.button("📇 Favorites page"):
        add_query(view="favorites")
        st.rerun()

# ------------------- Views -------------------
def view_home():
    st.markdown("## 🏠 Home")
    st.caption("Hint: use the sidebar to search any player by name. Add ⭐ to build your Favorites board.")
    season = "2025-26"
    st.markdown("### League Leaders — 2025-26 (Per Game)")
    cols = st.columns(4)

    for idx, stat in enumerate(["PTS","REB","AST","FG3M"]):
        with cols[idx]:
            info = league_leader(season, stat)
            if not info:
                st.error(f"Leaders {stat} unavailable.")
                continue
            pid = info["player_id"]
            head = fetch_headshot(pid)
            team = human_team_name(info["team_abbr"])
            val = info["value"]
            if head is None:
                st.warning("Headshot unavailable.")
            else:
                st.markdown('<div class="cardframe">', unsafe_allow_html=True)
                st.image(head, caption=None)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="leader"><div class="info"><div class="name">{link_to_player(pid, info["player"])}</div>'
                f'<div class="sub">{team}</div></div></div>',
                unsafe_allow_html=True
            )
            st.metric(f"{stat} (Per Game)", f"{val:.2f}")

def _player_header(pid: int):
    p = PLAYER_BY_ID.get(pid, {})
    name = p.get("full_name", "Player")
    st.markdown(f"## 🧾 {name}")
    favs: Dict[str,int] = st.session_state.get("favorites", {})
    if name in favs:
        if st.button("★ In Favorites (remove)"):
            favs.pop(name, None)
            st.session_state["favorites"] = favs
            st.rerun()
    else:
        if st.button("⭐ Add to Favorites"):
            favs[name] = pid
            st.session_state["favorites"] = favs
            st.rerun()

def _metrics_row(title: str, series: pd.Series):
    st.markdown(f"#### {title}")
    c1,c2,c3,c4 = st.columns(4)
    for (lab, col), c in zip([("PTS","PTS"),("REB","REB"),("AST","AST"),("3PM","FG3M")],[c1,c2,c3,c4]):
        v = series.get(col, np.nan)
        if pd.isna(v): c.metric(lab, "N/A")
        else: c.metric(lab, f"{float(v):.2f}")
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

def _bar_block(df: pd.DataFrame, title: str, n: int):
    st.markdown(f"**{title}**")
    cols = ["GAME_DATE","MATCHUP","PTS","REB","AST","FG3M","MIN"]
    show = df.head(n)[[c for c in cols if c in df.columns]].copy()
    if "GAME_DATE" in show:
        show["GAME_DATE"] = pd.to_datetime(show["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    st.dataframe(show, height=260, width='stretch')

def view_player(pid: int):
    _player_header(pid)

    # media day headshot + overlay team logo if known (from most recent log)
    seasons = ("2025-26","2024-25")
    logs = fetch_player_logs(pid, seasons)
    head = fetch_headshot(pid)
    if head is not None:
        st.image(head, width=260)
    else:
        st.caption("Headshot not available.")

    if logs.empty:
        st.error("No career data available or temporarily blocked by NBA API. Try again shortly.")
        return

    # try detect latest team id by matching team abbr
    latest_abbr = str(logs.iloc[0]["TEAM_ABBREVIATION"]) if "TEAM_ABBREVIATION" in logs.columns and len(logs) else None
    latest_tid = TEAM_BY_ABBR.get(latest_abbr, {}).get("id") if latest_abbr else None
    if latest_tid:
        logo = fetch_team_logo(latest_tid)
        if logo is not None:
            st.image(logo, width=64)

    # current season subset
    cur = logs[logs["SEASON"]=="2025-26"].copy()
    last = cur.head(1)
    # averages
    def _avg(df): 
        if df.empty: 
            return pd.Series({})
        return df[["PTS","REB","AST","FG3M","MIN"]].mean(numeric_only=True)

    _metrics_row("Current Season Averages", _avg(cur))
    _metrics_row("Last Game", last[["PTS","REB","AST","FG3M","MIN"]].iloc[0] if not last.empty else pd.Series({}))
    _metrics_row("Last 5 Games Avg", _avg(cur.head(5)))
    _metrics_row("Last 10 Games Avg", _avg(cur.head(10)))

    # ML prediction forced
    preds, tag = ml_predict_next(cur if not cur.empty else logs)
    st.markdown("#### Predicted Next Game (ML)")
    c1,c2,c3,c4 = st.columns(4)
    for (lab, key), c in zip([("PTS","PTS"),("REB","REB"),("AST","AST"),("3PM","FG3M")],[c1,c2,c3,c4]):
        v = preds.get(key, None)
        c.metric(lab + (" (ML)" if tag=="ML" else " (Fallback)"), "N/A" if v is None else f"{v:.2f}")

    # expanders with tables
    with st.expander("Last 5"):
        _bar_block(cur, "Last 5 Games", 5)
    with st.expander("Last 10"):
        _bar_block(cur, "Last 10 Games", 10)
    with st.expander("Last 20"):
        _bar_block(cur, "Last 20 Games", 20)
    with st.expander("Current Season"):
        _bar_block(cur, "All Games (Current Season)", len(cur))
    with st.expander("Last Season"):
        prev = logs[logs["SEASON"]=="2024-25"]
        _bar_block(prev, "All Games (Last Season)", len(prev))

    # Snapshot: Topps-like PNG
    if st.button("⬇️ Save Card (PNG)"):
        img = build_card_image(pid, preds, tag, head, latest_tid)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button("Download PNG", data=buf.getvalue(), file_name="player_card.png", mime="image/png")

def build_card_image(pid:int, preds:Dict[str,float], tag:str, head:Image.Image, tid:Optional[int]) -> Image.Image:
    W,H = 720, 480
    card = Image.new("RGBA",(W,H),(10,10,10,255))
    draw = ImageDraw.Draw(card)
    # frame
    draw.rounded_rectangle([10,10,W-10,H-10], radius=16, outline=(225,29,72,255), width=3)
    # headshot
    if head is not None:
        head = head.resize((300,220))
        card.paste(head,(24,40),head)
    # team logo
    if tid:
        logo = fetch_team_logo(tid)
        if logo is not None:
            ratio = 100 / max(logo.size)
            L = logo.resize((int(logo.size[0]*ratio), int(logo.size[1]*ratio)))
            card.paste(L,(W-140,26),L)
    # text
    p = PLAYER_BY_ID.get(pid,{})
    name = p.get("full_name","Player")
    def txt(x,y,t,sz=28,fill=(255,180,180,255)):
        draw.text((x,y), str(t), fill=fill)
    txt(24, 12, "Hot Shot Props • Card", 24)
    txt(350, 70, name, 30)
    txt(350, 110, f"Predicted (ML):", 22)
    base_y = 140
    for i,(lab,key) in enumerate([("PTS","PTS"),("REB","REB"),("AST","AST"),("3PM","FG3M")]):
        val = preds.get(key, None)
        txt(350, base_y + i*40, f"{lab}: " + ("N/A" if val is None else f"{val:.2f}"), 24)
    txt(350, base_y + 4*40 + 10, "(Model: ML Ridge)" if tag=="ML" else "(Model: Fallback)", 18, fill=(200,200,200,255))
    return card

def view_favorites():
    st.markdown("## 📇 Favorites")
    favs: Dict[str,int] = st.session_state.get("favorites", {})
    if not favs:
        st.info("No favorites yet. Star players from the sidebar or player pages.")
        return
    rows = []
    for name, pid in favs.items():
        logs = fetch_player_logs(pid, ("2025-26","2024-25"))
        preds, tag = ml_predict_next(logs[logs["SEASON"]=="2025-26"] if not logs.empty else logs)
        rows.append({
            "Player": name,
            "Link": f"Open",
            "PTS (ML)": preds.get("PTS", np.nan),
            "REB (ML)": preds.get("REB", np.nan),
            "AST (ML)": preds.get("AST", np.nan),
            "3PM (ML)": preds.get("FG3M", np.nan),
            "Model": tag
        })
    df = pd.DataFrame(rows)
    # Make names clickable
    df_display = df.copy()
    df_display["Player"] = df_display["Player"].apply(lambda n: f"[{n}](?player_id={PLAYER_BY_NAME[n]['id']})")
    st.dataframe(df_display, use_container_width=True)

# ------------------- App router -------------------
if "favorites" not in st.session_state:
    st.session_state["favorites"] = {}

sidebar_ui()

qp = dict(st.query_params)
pid = int(qp.get("player_id")) if "player_id" in qp else None
view = qp.get("view") if "view" in qp else None

if pid:
    view_player(pid)
elif view == "favorites":
    view_favorites()
else:
    view_home()