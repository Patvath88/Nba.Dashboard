# app.py — Hot Shot Props | NBA Player Analytics (single-file, multi-view)
# - Home: THIS-SEASON League Leaders (PTS/REB/AST/3PM/MIN) with links to player pages
# - Player: Team→Player fast search, favorites, metrics, bar charts, ML-lite prediction
# - Favorites: list with remove (x) + batch ML predictions table, names link to player pages
# - Snapshot: Save Card (PNG) — basketball-card styled with headshot + metrics
# - Performance: nba_api only, current-season-only logs, caching, short timeouts

import os, io, json, time, datetime as dt
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image, ImageDraw, ImageFont

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playercareerstats, playergamelog, leagueleaders
)

# ---------------------------- Page config & CSS ----------------------------
st.set_page_config(page_title="Hot Shot Props — NBA Player Analytics", layout="wide")
st.markdown("""
<style>
:root { --bg:#000; --panel:#0b0b0b; --ink:#f3f4f6; --line:#171717; --red:#ef4444; }
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink)!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#000 0%,#0b0b0b 100%)!important;border-right:1px solid #111;}
h1,h2,h3{color:#ffb4b4!important}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px}
.badge{display:inline-block;padding:4px 10px;border:1px solid var(--line);border-radius:999px;background:#140606;color:#fca5a5}
[data-testid="stMetric"]{background:#0e0e0e;border:1px solid #181818;border-radius:12px;padding:12px}
.stButton>button{background:var(--red)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:700!important}
.stDownloadButton>button{background:#1f2937!important;border:1px solid #374151!important}
.leader-card{background:#0e0e0e;border:1px solid #181818;border-radius:12px;padding:10px;display:flex;gap:12px;align-items:center}
.leader-img{width:56px;height:56px;border-radius:10px;border:1px solid #222;overflow:hidden}
.leader-name a{color:#ffb4b4;text-decoration:none;font-weight:700}
.fav-chip{display:flex;align-items:center;justify-content:space-between;background:#121212;border:1px solid #1e1e1e;border-radius:10px;padding:8px 10px;margin-bottom:8px}
.fav-x{background:transparent;border:1px solid #2a2a2a;color:#ddd;border-radius:7px;padding:2px 8px;cursor:pointer}
.small-hint{color:#a1a1aa;font-size:.92rem}
@media (max-width:780px){ .block-container{padding:0.6rem !important} [data-testid="stMetric"]{margin-bottom:10px} }
</style>
""", unsafe_allow_html=True)

# ---------------------------- Globals ----------------------------
DEFAULT_TMP_DIR = "/tmp" if os.access("/", os.W_OK) else "."
FAV_PATH = os.path.join(DEFAULT_TMP_DIR, "favorites.json")

NBA_TIMEOUT = 9     # tight timeouts → fewer long stalls
API_SLEEP   = 0.10  # polite pacing

STATS_COLS = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']
SHOW_KEYS  = ['PTS','REB','AST','FG3M','MIN']
PRED_KEYS  = ['PTS','REB','AST','FG3M']

# ---------------------------- Favorites persistence (dict) ----------------------------
def load_favorites() -> Dict[str, int]:
    if os.path.exists(FAV_PATH):
        try:
            with open(FAV_PATH, "r") as f:
                d = json.load(f)
                return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}

def save_favorites(favs: Dict[str,int]) -> None:
    try:
        with open(FAV_PATH, "w") as w:
            json.dump(favs, w, indent=2)
    except Exception:
        pass

if "favorites" not in st.session_state:
    st.session_state.favorites = load_favorites()

# ---------------------------- Cached data helpers ----------------------------
@st.cache_data(ttl=60*60, show_spinner=False)
def teams_df() -> pd.DataFrame:
    return pd.DataFrame(teams.get_teams())

@st.cache_data(ttl=60*60, show_spinner=False)
def players_df() -> pd.DataFrame:
    return pd.DataFrame(players.get_active_players())

def current_season() -> str:
    today = dt.date.today()
    yr = today.year
    start = yr if today.month >= 7 else yr - 1
    return f"{start}-{str(start+1)[2:]}"

@st.cache_data(ttl=60*15, show_spinner=True)
def fetch_leader(stat: str) -> Optional[pd.Series]:
    try:
        df = leagueleaders.LeagueLeaders(
            season=current_season(),
            per_mode48="PerGame",
            stat_category_abbreviation=stat,
            timeout=NBA_TIMEOUT
        ).get_data_frames()[0]
        return df.iloc[0]
    except Exception:
        return None

@st.cache_data(ttl=60*20, show_spinner=True)
def fetch_player_season(player_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Career table + current-season gamelog (sorted desc)."""
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=NBA_TIMEOUT).get_data_frames()[0]
    except Exception:
        career = pd.DataFrame()
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=current_season(), timeout=NBA_TIMEOUT).get_data_frames()[0]
        if 'GAME_DATE' in gl.columns:
            gl['GAME_DATE'] = pd.to_datetime(gl['GAME_DATE'])
            gl = gl.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
    except Exception:
        gl = pd.DataFrame()
    return career, gl

# ---------------------------- Utilities ----------------------------
def player_headshot(player_id: int) -> str:
    # nba cdn 1040x760; Streamlit auto-resizes
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

def route_to(view: str, **params):
    st.session_state["view"] = view
    for k,v in params.items():
        st.session_state[k] = v

def weighted_next_prediction(gl: pd.DataFrame) -> Dict[str, float]:
    """Weighted average of L5/L10/L20 + season mean (current season only)."""
    if gl is None or gl.empty:
        return {}
    gl = gl.copy()
    # normalize stat cols by names used in gamelog
    colmap = {
        'PTS':'PTS','REB':'REB','AST':'AST',
        'FG3M':'FG3M','MIN':'MIN'
    }
    preds = {}
    for stat in PRED_KEYS:
        c = colmap[stat]
        if c not in gl.columns:
            continue
        try:
            s5  = gl[c].head(5).mean()
            s10 = gl[c].head(10).mean()
            s20 = gl[c].head(20).mean()
            season = gl[c].mean()
            val = 0.4*s5 + 0.3*s10 + 0.2*s20 + 0.1*season
            preds[stat] = round(float(val), 2)
        except Exception:
            continue
    # minutes might be handy for display
    if 'MIN' in gl.columns:
        preds['MIN'] = round(float(gl['MIN'].head(5).mean()), 2)
    return preds

def bar_block(df_long: pd.DataFrame, x_name: str, y_name: str, title: str):
    chart = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(x=alt.X(f"{x_name}:N", sort=None), y=f"{y_name}:Q")
        .properties(title=title)
    )
    st.altair_chart(chart, use_container_width=True)

# ---------------------------- Sidebar (search + favorites + nav) ----------------------------
def sidebar_controls() -> Tuple[Optional[int], str]:
    st.sidebar.markdown("### Select Player")
    # view switch
    view = st.sidebar.radio("View", ["Home","Player","Favorites"], index={"Home":0,"Player":1,"Favorites":2}[st.session_state.get("view","Home")])
    st.session_state["view"] = view

    # Search
    tdf = teams_df().sort_values("full_name")
    team_names = ["All teams"] + tdf["full_name"].tolist()
    sel_team = st.sidebar.selectbox("Filter by team", team_names, index=0)

    pdf = players_df()
    if sel_team != "All teams":
        # map team id->abbr from nba_api is not directly in players list; we rely on user selecting player by name (global).
        pass
    # player list (sorted)
    name_list = sorted([p["full_name"] for _,p in pdf.iterrows()])
    selected_name = st.sidebar.selectbox("Player", ["(none)"] + name_list, index=0)
    sel_id = None
    if selected_name != "(none)":
        try:
            sel_id = int(pdf[pdf["full_name"]==selected_name].iloc[0]["id"])
        except Exception:
            sel_id = None

    # favorites list under search
    st.sidebar.markdown("### ⭐ Favorites")
    favs: Dict[str,int] = st.session_state.favorites
    if not favs:
        st.sidebar.caption("No favorites yet.")
    else:
        for name, pid in list(favs.items()):
            cols = st.sidebar.columns([0.75,0.25])
            if cols[0].button(name, key=f"fav_nav_{pid}"):
                route_to("Player", selected_player_id=pid, selected_player_name=name)
            if cols[1].button("❌", key=f"fav_rm_{pid}"):
                favs.pop(name, None)
                save_favorites(favs)
                st.rerun()

    # home hint
    st.sidebar.markdown("---")
    st.sidebar.button("🏠 Home", on_click=lambda: route_to("Home"))

    return sel_id, selected_name

# ---------------------------- Home View (League Leaders) ----------------------------
def view_home():
    st.title("Hot Shot Props — NBA Player Analytics")
    st.caption("**Hint:** Use the sidebar to search for any player. Save ⭐ favorites and view batch predictions on the Favorites tab.")

    cols = st.columns(5)
    stats = ["PTS","REB","AST","FG3M","MIN"]
    leaders = []
    with st.spinner("Loading league leaders..."):
        for s in stats:
            leaders.append((s, fetch_leader(s)))
            time.sleep(API_SLEEP)

    for i,(stat,row) in enumerate(leaders):
        with cols[i]:
            if row is None:
                st.error(f"Leader for {stat} unavailable.")
                continue
            name = f"{row['PLAYER']} ({row['TEAM']})"
            val  = row[stat]
            # resolve player id to link to Player view
            pdf = players_df()
            pid = None
            try:
                pid = int(pdf[pdf["full_name"]==row["PLAYER"]].iloc[0]["id"])
            except Exception:
                pass
            st.metric(f"{stat} Leader", f"{val:.2f}")
            if pid:
                if st.button("Open", key=f"ldr_{stat}_{pid}"):
                    route_to("Player", selected_player_id=pid, selected_player_name=row["PLAYER"])
                    st.rerun()
            st.caption(name)

# ---------------------------- Player View ----------------------------
def view_player(player_id: int, player_name: Optional[str]=None):
    if not player_id:
        st.info("Pick a player from the sidebar to load their dashboard.")
        return
    if not player_name:
        try:
            player_name = players_df().set_index("id").loc[player_id]["full_name"]
        except Exception:
            player_name = "Player"

    st.header(player_name)

    # Add / Remove favorite
    favs: Dict[str,int] = st.session_state.favorites
    is_fav = player_name in favs
    fav_col, _ = st.columns([0.3,0.7])
    if is_fav:
        if fav_col.button("⭐ Remove from Favorites"):
            favs.pop(player_name, None)
            save_favorites(favs)
            st.rerun()
    else:
        if fav_col.button("⭐ Add to Favorites"):
            favs[player_name] = player_id
            save_favorites(favs)
            st.rerun()

    # Fetch
    progress = st.progress(0.0)
    progress.progress(0.05, text="Fetching current season data...")
    career, gl = fetch_player_season(player_id)
    progress.progress(0.35, text="Preparing metrics...")

    # Headshot
    st.image(player_headshot(player_id), width=320)

    # Metrics rows
    if gl is None or gl.empty:
        st.error(f"Failed to load from nba_api (likely a temporary block). Try again in a minute.")
        return

    # Current season per-game average (from gamelog)
    cs = {}
    for k in SHOW_KEYS:
        if k in gl.columns:
            cs[k] = round(float(gl[k].mean()),2)
    mcols = st.columns(len(SHOW_KEYS))
    for i,k in enumerate(SHOW_KEYS):
        with mcols[i]:
            st.metric(f"{k} (Season Avg)", f"{cs.get(k,'N/A')}")

    progress.progress(0.55, text="Computing ML prediction...")

    # Last game stats (if available)
    lg = gl.head(1).copy()
    lg_vals = {}
    for k in SHOW_KEYS:
        if k in lg.columns:
            try:
                lg_vals[k] = round(float(lg.iloc[0][k]),2)
            except Exception:
                pass
    st.subheader("Last Game Stats")
    cols = st.columns(len(SHOW_KEYS))
    for i,k in enumerate(SHOW_KEYS):
        with cols[i]:
            st.metric(k, f"{lg_vals.get(k,'N/A')}")

    # Last 5 games averages
    st.subheader("Form (Last 5)")
    f5_vals = {}
    for k in SHOW_KEYS:
        if k in gl.columns:
            f5_vals[k] = round(float(gl[k].head(5).mean()),2)
    cols = st.columns(len(SHOW_KEYS))
    for i,k in enumerate(SHOW_KEYS):
        with cols[i]:
            st.metric(k, f"{f5_vals.get(k,'N/A')}")

    # Prediction (ML-lite)
    preds = weighted_next_prediction(gl)
    st.subheader("Predicted Next Game (ML)")
    cols = st.columns(len(SHOW_KEYS))
    for i,k in enumerate(SHOW_KEYS):
        with cols[i]:
            val = preds.get(k, "N/A")
            lab = f"{k} (ML)"
            st.metric(lab, f"{val}")

    progress.progress(0.7, text="Building bar charts...")

    # Bar graphs for each stat category (last 10)
    st.subheader("Breakdown (Last 10 Games)")
    last10 = gl.head(10)[["GAME_DATE","PTS","REB","AST","FG3M","MIN"]].copy()
    last10 = last10.rename(columns={"GAME_DATE":"Game"})
    last10["Game"] = last10["Game"].dt.strftime("%m-%d")
    long = last10.melt(id_vars=["Game"], var_name="Stat", value_name="Value")
    bar_block(long, "Game", "Value", f"{player_name} — Last 10 Games")

    progress.progress(0.85, text="Rendering snapshot card...")

    # Snapshot (Basketball Card PNG)
    if st.button("🖼️ Save Card (PNG)"):
        png_bytes = render_card_png(player_name, player_id, cs, lg_vals, f5_vals, preds)
        st.download_button("⬇️ Download Card", data=png_bytes, file_name=f"{player_name.replace(' ','_')}_card.png", mime="image/png")

    progress.progress(1.0, text="Done.")

# ---------------------------- Favorites View ----------------------------
def view_favorites():
    st.title("⭐ Favorites — Batch ML Predictions")
    favs: Dict[str,int] = st.session_state.favorites
    if not favs:
        st.info("No favorites yet. Add players from the Player view.")
        return

    rows = []
    with st.spinner("Fetching data & computing predictions..."):
        for name, pid in favs.items():
            time.sleep(API_SLEEP)
            _, gl = fetch_player_season(pid)
            preds = weighted_next_prediction(gl)
            row = {"Player": name, "Link": f"[open](/?view=Player&pid={pid})"}
            for k in SHOW_KEYS:
                row[k] = preds.get(k, np.nan)
            rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.markdown("#### Manage")
    for name, pid in list(favs.items()):
        cols = st.columns([0.75,0.25])
        with cols[0]:
            if st.button(name, key=f"fav_go_{pid}"):
                route_to("Player", selected_player_id=pid, selected_player_name=name)
                st.rerun()
        with cols[1]:
            if st.button("❌", key=f"fav_del_{pid}"):
                favs.pop(name, None)
                save_favorites(favs)
                st.rerun()

# ---------------------------- Card renderer (Pillow) ----------------------------
def render_card_png(name: str, pid: int, season: Dict, last: Dict, form5: Dict, preds: Dict) -> bytes:
    W, H = 1000, 600
    img = Image.new("RGB", (W,H), (12,12,12))
    draw = ImageDraw.Draw(img)

    # Try to load a safe system font
    try:
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 46)
        font_med = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font_big = ImageFont.load_default()
        font_med = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Frame
    draw.rounded_rectangle([(10,10),(W-10,H-10)], radius=26, outline=(239,68,68), width=4)

    # Headshot
    try:
        hs = Image.open(io.BytesIO(st.session_state.get(f"hs_{pid}") or b""))
    except Exception:
        # fetch if not cached
        try:
            import requests
            r = requests.get(player_headshot(pid), timeout=6)
            hs = Image.open(io.BytesIO(r.content))
            st.session_state[f"hs_{pid}"] = r.content
        except Exception:
            hs = Image.new("RGB", (600,400), (30,30,30))
    hs = hs.resize((440,320))
    img.paste(hs, (30,60))

    # Title
    draw.text((30,20), f"Hot Shot Props — Player Card", fill=(255,180,180), font=font_med)
    draw.text((500,65), name, fill=(255,255,255), font=font_big)
    draw.text((500,115), current_season(), fill=(200,200,200), font=font_small)

    # Stat blocks (season / last / form / pred)
    def block(x, y, title, d):
        draw.text((x,y), title, fill=(255,180,180), font=font_med)
        y += 36
        line = ",  ".join([f"{k}: {d.get(k,'N/A')}" for k in ['PTS','REB','AST','FG3M','MIN']])
        draw.text((x,y), line, fill=(230,230,230), font=font_small)

    block(500,160,"Season Avg", season or {})
    block(500,210,"Last Game", last or {})
    block(500,260,"Last 5 Avg", form5 or {})
    block(500,310,"Predicted Next (ML)", preds or {})

    # Save
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

# ---------------------------- Routing & App ----------------------------
def parse_query() -> Tuple[str, Optional[int]]:
    params = st.query_params
    view = params.get("view", [st.session_state.get("view","Home")])[0] if hasattr(params,"get") else st.session_state.get("view","Home")
    pid  = params.get("pid", [None])[0] if hasattr(params,"get") else None
    if pid is not None:
        try: pid = int(pid)
        except: pid = None
    return view, pid

def main():
    sel_id, sel_name = sidebar_controls()
    view, pid_from_qs = parse_query()

    # explicit navigation from sidebar
    if sel_id and view != "Player":
        route_to("Player", selected_player_id=sel_id, selected_player_name=sel_name)
        view = "Player"; pid_from_qs = sel_id

    if view == "Home":
        view_home()
    elif view == "Favorites":
        view_favorites()
    else:  # Player
        pid = pid_from_qs or st.session_state.get("selected_player_id") or sel_id
        name = st.session_state.get("selected_player_name") or sel_name
        # lock into player view state
        if pid:
            st.session_state["selected_player_id"] = pid
            if name: st.session_state["selected_player_name"] = name
        view_player(pid, name)

# Headshot URL helper used in render_card_png
def player_headshot(player_id: int) -> str:
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

if __name__ == "__main__":
    main()