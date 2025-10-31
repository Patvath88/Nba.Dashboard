# app.py — Hot Shot Props | NBA Player Analytics (Trading Card Edition)

import os, re, time, math, datetime as dt
from io import BytesIO
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import altair as alt

from nba_api.stats.static import players as nba_players, teams as nba_teams
from nba_api.stats.endpoints import (
    playergamelogs, playercareerstats, leagueleaders
)

# ============ CONFIG ============

st.set_page_config(
    page_title="Hot Shot Props • NBA Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# global style (black + “Topps” card vibes)
st.markdown("""
<style>
:root { --bg:#0a0a0a; --panel:#111; --ink:#f2f2f2; --muted:#bbb; --line:#1d1d1d; --accent:#ff4455; }
html, body, [data-testid="stAppViewContainer"]{ background:var(--bg)!important; color:var(--ink)!important; }
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0a0a0a, #0f0f0f); border-right:1px solid #121212; }
h1,h2,h3,h4{ color:#ffd7db !important; letter-spacing:.3px }
.stButton>button{ background:var(--accent) !important; color:#fff !important; border:none; border-radius:10px; font-weight:700; padding:.5rem .9rem }
.card{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:16px; }
.leader-wrap{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:14px; }
.leader-name a{ color:#ffd7db; text-decoration:none; font-size:1.1rem; font-weight:800; }
.leader-team{ color:#ffd7db99; font-size:.9rem; }
.leader-stat{ font-size:1.6rem; font-weight:900; color:#fff }
.hr{ border:0; border-top:1px solid var(--line); margin: 10px 0 16px }
.tag{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:.75rem;
      color:#e5e7eb; border:1px solid #333; background:#151515 }
.metric-row [data-testid="stMetric"]{ background:#0f0f0f; border:1px solid #181818; border-radius:14px; padding:12px }
.metric-row [data-testid="stMetric"] div[data-testid="stMetricValue"]{ color:#ffecef }
.topps-card{ background:linear-gradient(180deg,#141414,#0c0c0c); border:2px solid #cc3344;
             border-radius:22px; box-shadow:0 10px 30px rgba(0,0,0,.6); padding:18px }
.help{ color:#aaa; font-size:.9rem }
a { color:#ffd7db; }
</style>
""", unsafe_allow_html=True)

# ============ UTILITIES & CACHING ============

NBA_TIMEOUT = 12  # reduce timeout to avoid long stalls
API_SLEEP   = 0.12  # gentle throttle
THIS_SEASON = dt.date.today().year + 1 if dt.date.today().month >= 8 else dt.date.today().year
# Convert to 'YYYY-YY' format (e.g., 2025-26)
def season_str(year:int=None) -> str:
    if year is None: year = THIS_SEASON
    return f"{year-1}-{str(year)[-2:]}"

def pct(x: float) -> str:
    try: return f"{100.0*float(x):.0f}%"
    except: return "—"

def headshot_url(player_id: int) -> str:
    # CDN sizes: 1040x760 tends to exist; fallbacks will be handled by requests
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

@st.cache_data(ttl=3600)
def get_active_players() -> pd.DataFrame:
    return pd.DataFrame(nba_players.get_active_players())

@st.cache_data(ttl=3600)
def get_all_teams() -> pd.DataFrame:
    return pd.DataFrame(nba_teams.get_teams())

# Robust League Leaders across nba_api versions (per-game)
@st.cache_data(ttl=1800)
def get_league_leaders(stat: str, season: str, season_type: str = "Regular Season") -> Optional[pd.DataFrame]:
    stat = stat.upper()
    attempts = [
        {"per_mode": "PerGame"},
        {"per_mode48": "PerGame"},
        {}  # default
    ]
    err = None
    for kw in attempts:
        try:
            obj = leagueleaders.LeagueLeaders(
                season=season,
                season_type_all_star=season_type,
                stat_category_abbreviation=stat,
                timeout=NBA_TIMEOUT,
                **kw
            )
            df = obj.get_data_frames()[0]
            if not df.empty and "PLAYER_ID" in df.columns:
                return df
        except Exception as e:
            err = e
    st.warning(f"Leaders for {stat} unavailable: {err}")
    return None

# fetch player gamelogs for selected seasons (fast path: current + last)
@st.cache_data(ttl=900)
def fetch_player_logs(player_id: int, seasons: Tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        try:
            time.sleep(API_SLEEP)
            gl = playergamelogs.PlayerGameLogs(
                player_id_nullable=player_id,
                season_nullable=s,
                timeout=NBA_TIMEOUT
            ).get_data_frames()[0]
            if not gl.empty:
                gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
                gl["SEASON"] = s
                frames.append(gl)
        except Exception as e:
            st.warning(f"Logs unavailable for {s}: {e}")
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
        return df
    return pd.DataFrame()

# ============ ML MODEL (Ridge) ============

try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SK_ML = True
except Exception:
    SK_ML = False

ML_TARGETS = ["PTS","REB","AST","FG3M"]

def build_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df.empty: return None
    base = df.copy()
    want = ["PTS","REB","AST","FG3M","MIN","FGA","FGM","FG3A","FTA","OREB","DREB","TOV","BLK","STL"]
    for c in want:
        if c not in base.columns: base[c]=np.nan
        # rolling means to encode recent form
        base[f"{c}_r5"]  = base[c].rolling(5,  min_periods=1).mean()
        base[f"{c}_r10"] = base[c].rolling(10, min_periods=1).mean()
        base[f"{c}_r20"] = base[c].rolling(20, min_periods=1).mean()
    # drop rows with too many NaNs
    base = base.dropna(subset=["PTS","REB","AST","FG3M"], how="all")
    return base

@st.cache_data(ttl=1800)
def train_ridge_models(player_id: int, df: pd.DataFrame) -> Dict[str, Pipeline]:
    models = {}
    if not SK_ML or df is None or df.empty:
        return models
    feats = [c for c in df.columns if any(s in c for s in ["_r5","_r10","_r20"]) and ("_%" not in c)]
    if len(feats) < 6:
        return models
    for target in ML_TARGETS:
        if target not in df.columns: 
            continue
        X = df[feats].fillna(method="ffill").fillna(0.0)
        y = df[target].fillna(0.0)
        if len(X) < 12:  # need at least ~12 games to generalize a little
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([("scaler", StandardScaler()), ("mdl", Ridge(alpha=1.0))])
        pipe.fit(Xtr, ytr)
        models[target] = pipe
    return models

def predict_next(models: Dict[str, Pipeline], df: pd.DataFrame) -> Tuple[Dict[str,float], bool]:
    out, used_ml = {}, False
    if models and not df.empty:
        feats = [c for c in df.columns if any(s in c for s in ["_r5","_r10","_r20"]) and ("_%" not in c)]
        Xlast = df[feats].tail(1).fillna(method="ffill").fillna(0.0)
        for t in ML_TARGETS:
            if t in models:
                try:
                    pred = float(models[t].predict(Xlast)[0])
                    out[t] = round(pred, 2)
                    used_ml = True
                except Exception:
                    pass
    # fallback: weighted moving average if ML not available for some targets
    for t in ML_TARGETS:
        if t not in out:
            s = df[t].rolling(5,min_periods=1).mean().iloc[-1]*0.5 + \
                df[t].rolling(10,min_periods=1).mean().iloc[-1]*0.3 + \
                df[t].rolling(20,min_periods=1).mean().iloc[-1]*0.2
            out[t] = round(float(s),2)
    return out, used_ml

# ============ PNG CARD EXPORT ============

def make_trading_card_png(name:str, team:str, headshot:Image.Image, preds:Dict[str,float]) -> BytesIO:
    W,H = 720, 1024
    img = Image.new("RGB",(W,H),(12,12,12))
    draw = ImageDraw.Draw(img)
    # border
    draw.rounded_rectangle((10,10,W-10,H-10), radius=28, outline=(204,51,68), width=6)
    # name plate
    draw.rectangle((30,30,W-30,110), fill=(30,30,30))
    try:
        f_big  = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
        f_med  = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        f_stat = ImageFont.truetype("DejaVuSans.ttf", 26)
    except:
        f_big=f_med=f_stat=None
    draw.text((40,42), name, fill=(255,215,220), font=f_big)
    draw.text((40,82), team, fill=(220,170,175), font=f_med)
    # headshot
    if headshot:
        hs = headshot.copy().resize((W-60, int((W-60)*0.7)))
        img.paste(hs, (30,130))
    # stats
    y0 = 130 + int((W-60)*0.7) + 30
    draw.text((40,y0), "Next Game (ML):", fill=(255,215,220), font=f_med)
    y = y0 + 42
    for k in ["PTS","REB","AST","FG3M"]:
        val = f"{preds.get(k,'—')}"
        draw.text((60,y), f"{k}: {val}", fill=(245,240,240), font=f_stat)
        y += 36
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf

# ============ SIDEBAR (Search + Favorites + Home) ============

def go_home():
    st.session_state.pop("selected_player", None)
    st.query_params.clear()
    st.rerun()

def sidebar_ui():
    with st.sidebar:
        st.header("Select Player")
        # search + team filter
        all_players = get_active_players()
        all_teams   = get_all_teams()

        team_names = ["(All teams)"] + sorted(all_teams["full_name"].tolist())
        team_choice = st.selectbox("Filter by team", team_names, index=0)
        df = all_players.copy()
        if team_choice != "(All teams)":
            # map team to ids by matching player current team via gamelogs is heavy; keep as text filter
            team_abbr = [t["abbreviation"] for t in nba_teams.get_teams() if t["full_name"]==team_choice][0]
            # heuristic: players often include 'team' in id mapping via separate endpoint, skip heavy filter here

        name = st.text_input("Search by name", "")
        if name:
            patt = re.compile(re.escape(name), re.I)
            df = df[df["full_name"].str.contains(patt)]
        # pick player
        selected = st.selectbox("Players", [""] + sorted(df["full_name"].tolist()))
        if selected:
            pid = int(df[df["full_name"]==selected]["id"].iloc[0])
            st.session_state.selected_player = (pid, selected)
            st.query_params["player_id"] = str(pid)
            st.rerun()

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        # favorites
        favs = st.session_state.get("favorites", {})
        st.subheader("⭐ Favorites")
        if favs:
            for name, pid in list(favs.items()):
                cols = st.columns([0.8,0.2])
                with cols[0]:
                    st.button(name, key=f"fav_{pid}", on_click=lambda p=pid,n=name: select_favorite(p,n))
                with cols[1]:
                    if st.button("✕", key=f"del_{pid}"):
                        favs.pop(name, None)
                        st.session_state.favorites = favs
                        st.rerun()
        else:
            st.caption("No favorites yet.")
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.button("🏠 Home", on_click=go_home)

def select_favorite(pid:int, name:str):
    st.session_state.selected_player = (pid, name)
    st.query_params["player_id"] = str(pid)
    st.rerun()

# ============ HOME PAGE (Leaders with big headshots) ============

def render_home():
    st.title("🏠 Home")
    st.caption("Hint: use the sidebar to search any player by name. Add ⭐ to build your Favorites board.")
    season = season_str()
    st.subheader(f"League Leaders — {season} (Per Game)")
    cols = st.columns(4)
    groups = [("PTS","Points"),("REB","Rebounds"),("AST","Assists"),("FG3M","3-Point Makes")]

    for col, (abbr,label) in zip(cols, groups):
        with col:
            df = get_league_leaders(abbr, season)
            if df is None or df.empty:
                st.error(f"Leaders {abbr} unavailable.")
                continue
            # ensure sorted
            sort_col = abbr if abbr in df.columns else df.columns[5]
            row = df.sort_values(sort_col, ascending=False).iloc[0]
            pid = int(row.get("PLAYER_ID", 0))
            pname = row.get("PLAYER", row.get("PLAYER_NAME","Unknown"))
            team  = row.get("TEAM", row.get("TEAM_ABBREVIATION",""))
            value = row.get(abbr, row.get("PTS", 0.0))
            # headshot
            img = None
            try:
                r = requests.get(headshot_url(pid), timeout=6)
                if r.ok:
                    img = Image.open(BytesIO(r.content)).convert("RGB")
            except Exception:
                pass

            st.markdown("<div class='leader-wrap'>", unsafe_allow_html=True)
            if img:
                st.image(img, caption=None, width=260)
            st.markdown(f"<div class='leader-name'><a href='?player_id={pid}'>{pname}</a></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='leader-team'>{team}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='leader-stat'>{label}: {value:.2f}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ============ PLAYER PAGE ============

def render_player(pid:int, pname:str):
    # Add to favorites button and banner
    if "favorites" not in st.session_state:
        st.session_state.favorites = {}
    st.title(pname)
    cols = st.columns([0.3,0.7])
    with cols[0]:
        if st.button("⭐ Add to Favorites"):
            st.session_state.favorites[pname] = pid
            st.success("Added to Favorites.")
    with cols[1]:
        ph = st.empty()
        ph.info("Preparing metrics...")

    # fast seasons (current + last)
    seasons = (season_str(), season_str(THIS_SEASON-1))
    logs = fetch_player_logs(pid, seasons)
    if logs.empty:
        ph.empty()
        st.error("Failed to load from nba_api (likely a temporary block). Try again shortly.")
        return
    ph.empty()

    # headshot
    hs_img = None
    try:
        r = requests.get(headshot_url(pid), timeout=6)
        if r.ok: hs_img = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception: pass
    if hs_img:
        st.image(hs_img, width=340)

    # metrics rows
    st.subheader("Predicted Next Game (ML)")
    feats = build_features(logs)
    models = train_ridge_models(pid, feats) if feats is not None else {}
    preds, used_ml = predict_next(models, feats if feats is not None else logs)
    label = "ML model" if used_ml else "WMA fallback"
    with st.container():
        row = st.container()
        with row:
            st.caption(f"Source: {label}")
            mcols = st.columns(4)
            for i,k in enumerate(["PTS","REB","AST","FG3M"]):
                with mcols[i]:
                    st.metric(k, f"{preds.get(k,'—')}")

    # expanders: L5 / L10 / L20 / Current / Last
    def bar_panel(title: str, df: pd.DataFrame):
        st.markdown(f"#### {title}")
        show = df[["GAME_DATE","PTS","REB","AST","FG3M"]].copy()
        show["GAME_DATE"] = show["GAME_DATE"].dt.strftime("%Y-%m-%d")
        long = show.melt("GAME_DATE", var_name="Stat", value_name="Value")
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("Stat:N"),
            y=alt.Y("Value:Q"),
            color=alt.Color("Stat:N"),
            column=alt.Column("GAME_DATE:N", title="Game Date")
        ).properties(height=180)
        st.altair_chart(chart, use_container_width=False)

    with st.expander("Last 5"):
        if len(logs)>=5: bar_panel("Last 5", logs.head(5))
        else: st.info("Not enough games.")

    with st.expander("Last 10"):
        if len(logs)>=10: bar_panel("Last 10", logs.head(10))
        else: st.info("Not enough games.")

    with st.expander("Last 20"):
        if len(logs)>=20: bar_panel("Last 20", logs.head(20))
        else: st.info("Not enough games.")

    with st.expander("Current Season"):
        cur = logs[logs["SEASON"]==seasons[0]]
        if not cur.empty: bar_panel(f"{seasons[0]}", cur)
        else: st.info("No current season data yet.")

    with st.expander("Last Season"):
        prev = logs[logs["SEASON"]==seasons[1]]
        if not prev.empty: bar_panel(f"{seasons[1]}", prev)
        else: st.info("No last season data.")

    # Download trading card
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("Download Trading Card (PNG)")
    team = ""
    try:
        team = logs["TEAM_ABBREVIATION"].iloc[0]
    except Exception:
        pass
    buf = make_trading_card_png(pname, team, hs_img if hs_img else Image.new("RGB",(600,420),(20,20,20)), preds)
    st.download_button("⬇️ Download PNG Card", data=buf, file_name=f"{pname.replace(' ','_')}_card.png", mime="image/png")

# ============ FAVORITES PAGE ============

def render_favorites():
    st.title("⭐ Favorites")
    favs = st.session_state.get("favorites", {})
    if not favs:
        st.info("No favorites yet. Add some from a player page.")
        return
    rows = []
    for name, pid in favs.items():
        logs = fetch_player_logs(pid, (season_str(),))
        if logs.empty:
            rows.append((name, pid, "—","—","—","—","No data"))
            continue
        feats = build_features(logs)
        models = train_ridge_models(pid, feats) if feats is not None else {}
        preds, used_ml = predict_next(models, feats if feats is not None else logs)
        rows.append((name, pid, preds["PTS"], preds["REB"], preds["AST"], preds["FG3M"], "ML" if used_ml else "WMA"))
    df = pd.DataFrame(rows, columns=["Player","PlayerID","PTS","REB","AST","3PM","Model"])
    # clickable player names
    for i in df.index:
        pid = int(df.at[i,"PlayerID"])
        df.at[i,"Player"] = f"[{df.at[i,'Player']}](?player_id={pid})"
    st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)

# ============ ROUTER ============

def main():
    if "favorites" not in st.session_state:
        st.session_state.favorites = {}
    sidebar_ui()

    q = st.query_params
    view = q.get("view", ["home"])[0]
    pid  = q.get("player_id", [None])[0]

    if pid:
        # resolve name for title / favorites
        ap = get_active_players()
        pname = ap.loc[ap["id"]==int(pid), "full_name"]
        pname = pname.iloc[0] if not pname.empty else "Player"
        render_player(int(pid), pname)
    elif view == "favorites":
        render_favorites()
    else:
        render_home()

if __name__ == "__main__":
    main()