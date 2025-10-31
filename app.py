# app.py — Hot Shot Props | NBA Player Trading Cards (Topps-style)
# - NBA-only data via nba_api (no balldontlie)
# - Robust request layer (rate-limit friendly, retries, jitter, cache)
# - Home: League Leaders (PTS/REB/AST/3PM) with large headshots + buttons
# - Player "card" view: season avg, last game, last-5 avg, ML prediction, bar charts
# - Favorites page: list + batch predicted next-game stats for favorited players
# - Download Trading Card PNG (Topps-style frame)
# - Mobile-friendly + card theme

import os, time, random, threading
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# ---------- Optional ML deps ----------
try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    import joblib
    SKLEARN_OK = True
except Exception:
    Ridge = None
    train_test_split = None
    joblib = None
    SKLEARN_OK = False

# ---------- Streamlit page ----------
st.set_page_config(
    page_title="Hot Shot Props — NBA Player Trading Cards",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Topps-like theme (CSS) ----------
st.markdown("""
<style>
:root{
  --bg:#0a0a0a; --ink:#f5f5f5; --muted:#bdbdbd; --line:#1a1a1a;
  --accent:#ff3355; --accent2:#1e90ff; --gold:#ffd166;
  --card:#101216; --card-grad:linear-gradient(135deg,#0d0f13 0%,#12151b 50%,#0f1116 100%);
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink)!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a0a0a 0,#0c0c0c 100%)!important;border-right:1px solid var(--line);}
.card{
  border-radius:18px; border:2px solid #2b2e36; background:var(--card-grad);
  box-shadow:0 18px 60px rgba(0,0,0,.55), inset 0 0 0 2px rgba(255,255,255,.03);
  padding:16px;
}
.card-title{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu; font-weight:900; letter-spacing:.3px;}
.topps-frame{
  position:relative; padding:12px; border-radius:18px;
  background:linear-gradient(135deg,#1f2330,#171b25);
  border:3px solid #3b4150; box-shadow:inset 0 0 0 3px rgba(255,255,255,.05), 0 14px 32px rgba(0,0,0,.45);
}
.topps-frame:before{
  content:""; position:absolute; inset:-3px; border-radius:18px;
  padding:2px; background:linear-gradient(135deg,#ff3355,#ff9d00,#1e90ff);
  -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude;
}
.big-name{font-weight:900; font-size:clamp(22px,3vw,34px); line-height:1.05;}
.subtle{color:var(--muted); font-size:.95rem;}
.badge{display:inline-block; padding:.25rem .6rem; border-radius:999px; border:1px solid #394150; background:#151823; color:#e6e7ea; font-size:.8rem;}
.stat-metric{
  background:#131722; border:1px solid #2a3040; border-radius:14px; padding:12px;
}
.stat-metric h4{margin:0 0 6px 0; font-size:.95rem; color:#cbd5e1;}
.stat-metric .v{font-weight:800; font-size:1.35rem; color:#fff;}
.btn-primary button{background:var(--accent)!important;border:none;border-radius:12px;color:white;font-weight:800;}
.btn-alt button{background:#1e90ff!important;border:none;border-radius:12px;color:white;font-weight:800;}
hr{border:0;border-top:1px solid var(--line); margin:12px 0;}
@media (max-width: 820px){
  .stButton>button{width:100%}
}
</style>
""", unsafe_allow_html=True)

# ---------- NBA request layer (robust) ----------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import players as players_static, teams as teams_static
from nba_api.stats.endpoints import leagueleaders, playercareerstats, playergamelogs

NBA_TIMEOUT        = 12     # seconds
NBA_MAX_RETRIES    = 6
NBA_MIN_DELAY      = 0.18   # base inter-call delay
NBA_BACKOFF        = 1.7
NBA_JITTER         = (0.00, 0.25)
NBA_BURST_CAP      = 10
NBA_REFILL_RATE    = 5
HEADSHOT = lambda pid: f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"

_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36",
]
def _nba_headers():
    return {
        "User-Agent": random.choice(_UA_POOL),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
    }

_session_lock = threading.Lock()
_SESSION = None
def _get_session():
    global _SESSION
    with _session_lock:
        if _SESSION is None:
            s = requests.Session()
            adapter = HTTPAdapter(max_retries=Retry(total=0))
            s.mount("https://", adapter); s.mount("http://", adapter)
            _SESSION = s
        return _SESSION

class _Bucket:
    def __init__(self, cap=NBA_BURST_CAP, refill_per_sec=NBA_REFILL_RATE):
        self.cap=cap; self.tokens=cap; self.refill=refill_per_sec
        self.t=time.monotonic(); self.m=threading.Lock()
    def take(self):
        with self.m:
            now=time.monotonic()
            self.tokens=min(self.cap, self.tokens+(now-self.t)*self.refill)
            self.t=now
            if self.tokens<1:
                time.sleep((1-self.tokens)/self.refill)
                self.tokens=0; self.t=time.monotonic()
            else:
                self.tokens-=1
_BUCKET = _Bucket()

def call_nba_endpoint(endpoint_cls, **kwargs):
    sess = _get_session()
    last_err = None
    for attempt in range(1, NBA_MAX_RETRIES+1):
        _BUCKET.take()
        time.sleep(NBA_MIN_DELAY + random.uniform(*NBA_JITTER))
        try:
            obj = endpoint_cls(headers=_nba_headers(), timeout=NBA_TIMEOUT, session=sess, **kwargs)
            _ = obj.get_dict()
            return obj
        except Exception as e:
            last_err = e
            sleep = min((NBA_MIN_DELAY * (NBA_BACKOFF ** (attempt-1))) + random.uniform(*NBA_JITTER), 8.0)
            time.sleep(sleep)
    raise RuntimeError(f"NBA endpoint failed after {NBA_MAX_RETRIES} tries: {last_err}")

@st.cache_data(ttl=60*30)
def get_league_leaders_this_season(cat="PTS"):
    season = SeasonAll.current_season.value  # e.g., "2024-25"
    obj = call_nba_endpoint(
        leagueleaders.LeagueLeaders,
        season=season,
        season_type_all_star="Regular Season",
        stat_category_abbreviation=cat
    )
    return obj.get_data_frames()[0]

@st.cache_data(ttl=60*30, show_spinner=False)
def safe_player_career(player_id:int) -> pd.DataFrame:
    obj = call_nba_endpoint(playercareerstats.PlayerCareerStats, player_id=player_id)
    return obj.get_data_frames()[0]

@st.cache_data(ttl=60*30, show_spinner=False)
def safe_player_gamelogs(player_id:int, season:str|None=None) -> pd.DataFrame:
    kwargs={"player_id_nullable":player_id}
    if season: kwargs["season_nullable"]=season
    obj = call_nba_endpoint(playergamelogs.PlayerGameLogs, **kwargs)
    return obj.get_data_frames()[0]

# ---------- Static helpers ----------
@st.cache_data(ttl=24*60*60)
def get_active_players() -> pd.DataFrame:
    # nba_api static returns list[dict]
    return pd.DataFrame(players_static.get_active_players())

@st.cache_data(ttl=24*60*60)
def get_teams_df() -> pd.DataFrame:
    return pd.DataFrame(teams_static.get_teams())

TEAMS = get_teams_df()
TEAM_BY_ID = {t["id"]: t for _,t in TEAMS.iterrows()} if not TEAMS.empty else {}
TEAM_BY_ABBR = {t["abbreviation"]: t for _,t in TEAMS.iterrows()} if not TEAMS.empty else {}

STAT_COLS = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']
PREDICT_TARGETS = ['PTS','REB','AST','FG3M']

# ---------- ML prediction (Ridge; fallback = weighted avg) ----------
def _weighted_next(vals: pd.Series) -> float:
    # 40/30/20/10 (last5/10/20/season) style fallback
    w = [0.4, 0.3, 0.2, 0.1]
    return float(np.dot(vals.values, w))

def predict_next_game(logs: pd.DataFrame) -> dict:
    """
    logs: player game logs (all seasons). Must include targets columns.
    """
    if logs is None or logs.empty:
        return {}
    # sort by date descending
    if "GAME_DATE" in logs.columns:
        logs = logs.copy()
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
        logs = logs.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    # Feature creation: rolling means
    feats = pd.DataFrame(index=logs.index)
    for c in STAT_COLS:
        if c in logs.columns:
            feats[f"{c}_r5"]  = logs[c].rolling(5,  min_periods=1).mean()
            feats[f"{c}_r10"] = logs[c].rolling(10, min_periods=1).mean()
            feats[f"{c}_r20"] = logs[c].rolling(20, min_periods=1).mean()
    feats = feats.fillna(0)

    preds = {}
    for tgt in PREDICT_TARGETS:
        if tgt not in logs.columns: 
            continue
        y = logs[tgt].fillna(0).values
        X = feats.values
        if SKLEARN_OK and len(logs) >= 14 and X.shape[0] == len(y):
            try:
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
                model = Ridge(alpha=1.0).fit(Xtr, ytr)
                # predict next game using latest row of feats
                v = float(model.predict(feats.iloc[[0]].values)[0])
                preds[tgt] = round(v, 2)
                continue
            except Exception:
                pass
        # fallback weighted average
        series_vals = []
        for w_n in [5, 10, 20]:
            series_vals.append(logs[tgt].head(w_n).mean())
        # "season" average over all logs
        series_vals.append(logs[tgt].mean())
        preds[tgt] = round(_weighted_next(pd.Series(series_vals).fillna(series_vals[-1])), 2)
    return preds

# ---------- Sidebar ----------
def sidebar_ui():
    st.sidebar.markdown("### 🔎 Search player")
    plist = get_active_players()
    plist["label"] = plist["full_name"]
    # optional team filter
    team_abbrs = ["(All teams)"] + sorted(list(TEAMS["abbreviation"])) if not TEAMS.empty else ["(All teams)"]
    sel_team = st.sidebar.selectbox("Team filter", team_abbrs, index=0)
    if sel_team != "(All teams)":
        # If you want stricter mapping, you can query by team in logs;
        # here we leave the global player list since nba_api static doesn't map roster live reliably.
        pass
    search = st.sidebar.text_input("Type a player's name...", "")
    options = plist if search.strip()=="" else plist[plist["label"].str.contains(search, case=False, na=False)]
    names = options["label"].tolist()
    pid_map = dict(zip(options["label"], options["id"]))
    chosen = st.sidebar.selectbox("Select", names, index=None, placeholder="e.g. Jalen Brunson")

    # favorites in session
    favs = st.session_state.get("favorites", {})
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = {}

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⭐ Favorites")
    if favs:
        # each favorite line with remove 'x'
        for nm, pid in list(favs.items()):
            cols = st.sidebar.columns([0.75,0.25])
            with cols[0]:
                if st.sidebar.button(nm, key=f"fav_btn_{pid}"):
                    st.session_state["selected_player"] = pid
                    st.session_state["selected_name"] = nm
                    st.session_state["tab"] = "Player"
                    st.rerun()
            with cols[1]:
                if st.sidebar.button("✕", key=f"rm_{pid}"):
                    del st.session_state["favorites"][nm]
                    st.rerun()
    else:
        st.sidebar.caption("No favorites yet.")

    st.sidebar.markdown("---")
    # nav tabs
    tab = st.sidebar.radio("Navigate", ["Home","Player","Favorites"], index=0, horizontal=False)
    st.session_state["tab"] = tab

    if chosen:
        st.session_state["selected_player"] = pid_map[chosen]
        st.session_state["selected_name"] = chosen
        st.session_state["tab"] = "Player"
        st.rerun()

# ---------- Home: League leaders ----------
def _leader_card(df: pd.DataFrame, label: str):
    if df is None or df.empty:
        st.warning(f"No leaders for {label}.")
        return
    row = df.iloc[0]
    pid = int(row.get("PLAYER_ID", 0))
    name = str(row.get("PLAYER", "Unknown"))
    team = str(row.get("TEAM", row.get("TEAM_ABBREVIATION","")))
    value = None
    for k in ["PTS","REB","AST","FG3M","3PM"]:
        if k in row:
            value = row[k]; break
    c1,c2 = st.columns([1.1,1.8])
    with c1:
        st.image(HEADSHOT(pid), width=None)
    with c2:
        st.markdown(f"""
        <div class="big-name">{name}</div>
        <div class="subtle">{team}</div>
        <div style="margin-top:8px;font-size:1.05rem;">
          <span class="badge">Leader — {label}</span><br/>
          <span style="font-size:1.45rem;font-weight:900">{value:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Open {name} →", key=f"open_leader_{label}_{pid}"):
            st.session_state["selected_player"] = pid
            st.session_state["selected_name"] = name
            st.session_state["tab"] = "Player"
            st.rerun()

def home_screen():
    st.markdown('<div class="card"><div class="card-title" style="font-size:1.6rem">🏠 Home</div><div class="subtle">Hint: use the sidebar to search any player by name. Add ⭐ to build your Favorites board.</div></div>', unsafe_allow_html=True)
    st.markdown("### League Leaders (Current Season)")
    cols = st.columns(4)
    mapping = [("PTS","PTS"), ("REB","REB"), ("AST","AST"), ("3PM","FG3M")]
    for i,(label,cat) in enumerate(mapping):
        with cols[i]:
            try:
                df = get_league_leaders_this_season(cat if cat!="3PM" else "FG3M")
                st.markdown('<div class="topps-frame">', unsafe_allow_html=True)
                _leader_card(df, label)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Leaders {label} unavailable: {e}")

# ---------- Player Card ----------
def row_metrics(title: str, data: dict):
    st.markdown(f"#### {title}")
    c = st.columns(4)
    keys = ["PTS","REB","AST","FG3M"]
    for i,k in enumerate(keys):
        with c[i]:
            v = data.get(k, "N/A")
            vtxt = f"{v:.2f}" if isinstance(v,(float,int)) else str(v)
            st.markdown(f"""
            <div class="stat-metric">
              <h4>{k}</h4>
              <div class="v">{vtxt}</div>
            </div>
            """, unsafe_allow_html=True)

def bar_block(df: pd.DataFrame, title: str, cols: list[str]):
    st.markdown(f"#### {title}")
    if df is None or df.empty:
        st.info("No data.")
        return
    # make long
    show = df.copy()
    show = show[cols]
    long = show.melt(var_name="Stat", value_name="Value")
    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X("Stat:N", axis=alt.Axis(labelColor="#e5e7eb", title=None)),
        y=alt.Y("Value:Q", axis=alt.Axis(labelColor="#e5e7eb")),
        tooltip=["Stat","Value"]
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)

def player_card(pid: int, pname: str):
    # Headshot + name/top bar
    st.markdown('<div class="topps-frame">', unsafe_allow_html=True)
    c1,c2 = st.columns([1.1,2.0])
    with c1:
        st.image(HEADSHOT(pid))
    with c2:
        st.markdown(f'<div class="big-name">{pname}</div>', unsafe_allow_html=True)
        favs = st.session_state.get("favorites", {})
        is_fav = pname in favs
        colA, colB = st.columns([0.35,0.65])
        with colA:
            if not is_fav:
                if st.button("⭐ Add to Favorites"):
                    st.session_state["favorites"][pname] = pid
                    st.rerun()
            else:
                st.success("In Favorites")
        with colB:
            st.caption("Preparing metrics...")

    st.markdown('</div>', unsafe_allow_html=True)

    # Fetch data
    try:
        career = safe_player_career(pid)
        logs = safe_player_gamelogs(pid)
    except Exception as e:
        st.error(f"Failed to load from nba_api (likely a temporary block). Try again in a minute.\n\n{e}")
        return

    # Current season rows from career
    if career is None or career.empty:
        st.info("No career data available.")
        return
    # Take last row (latest season)
    cur_row = career.iloc[-1]
    # season per-game averages
    season_avg = {}
    gp = max(float(cur_row.get("GP", 0) or 0), 1.0)
    for k in STAT_COLS:
        if k in career.columns:
            season_avg[k] = float(cur_row[k]) / gp
    row_metrics("Current Season Averages", {k:season_avg.get(k,"N/A") for k in ["PTS","REB","AST","FG3M"]})

    # Last game
    last_game_stats = {}
    if logs is not None and not logs.empty:
        lg = logs.sort_values("GAME_DATE", ascending=False).iloc[0]
        for k in ["PTS","REB","AST","FG3M","MIN"]:
            if k in logs.columns:
                last_game_stats[k] = float(lg[k])
    row_metrics("Last Game", last_game_stats)

    # Last 5 avg
    last5 = {}
    if logs is not None and not logs.empty:
        head5 = logs.sort_values("GAME_DATE", ascending=False).head(5)
        for k in ["PTS","REB","AST","FG3M"]:
            if k in head5.columns:
                last5[k] = float(head5[k].mean())
    row_metrics("Last 5 Games Averages", last5)

    # ML prediction
    preds = predict_next_game(logs if logs is not None else pd.DataFrame())
    row_metrics("Predicted Next Game (ML)", preds)

    # Bar graphs for core stats (season averages)
    core_cols = ["PTS","REB","AST","FG3M","MIN"]
    s_for_chart = pd.DataFrame([ {k:season_avg.get(k,0.0) for k in core_cols} ])
    bar_block(s_for_chart, "Season Averages — Bars", core_cols)

    # Last 10-game bar breakdown (sum/mean)
    if logs is not None and not logs.empty:
        last10 = logs.sort_values("GAME_DATE", ascending=False).head(10)
        last10_avg = {k: float(last10[k].mean()) for k in core_cols if k in last10.columns}
        last10_df = pd.DataFrame([last10_avg])
        bar_block(last10_df, "Last 10 Games — Average Bars", list(last10_avg.keys()))

    # Download Trading Card PNG
    make_trading_card(pid, pname, season_avg, preds)

def make_trading_card(pid:int, name:str, season_avg:dict, preds:dict):
    """
    Creates a Topps-like PNG card with headshot + few key stats & download button.
    """
    # Basic canvas
    W,H = 780, 1080
    img = Image.new("RGB", (W,H), (16,18,22))
    draw = ImageDraw.Draw(img)

    # Load headshot
    try:
        r = requests.get(HEADSHOT(pid), timeout=10)
        face = Image.open(BytesIO(r.content)).convert("RGB").resize((W-80, int((W-80)/1.37)))
    except Exception:
        face = Image.new("RGB",(W-80,int((W-80)/1.37)), (25,27,33))
    img.paste(face, (40, 120))

    # Frame & header
    draw.rounded_rectangle((20,20,W-20,H-20), radius=26, outline=(60,65,82), width=6)
    draw.rounded_rectangle((26,26,W-26,H-26), radius=22, outline=(255,51,85), width=3)

    # Fonts (system fallbacks)
    try:
        # If you have a custom bold TTF, place it in repo and load it here
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 52)
        stat_font  = ImageFont.truetype("DejaVuSans.ttf", 36)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 26)
    except Exception:
        title_font = ImageFont.load_default()
        stat_font  = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Player name
    draw.text((40, 40), name, fill=(255,255,255), font=title_font)

    # Bottom stat strips (season avg + ML preds)
    y0 = H - 290
    draw.rounded_rectangle((40, y0, W-40, y0+90), radius=16, fill=(30,33,44), outline=(64,71,90))
    draw.text((56,y0+18), "Season Avg", fill=(255,209,102), font=small_font)
    s_keys = ["PTS","REB","AST","FG3M"]
    s_vals = "  |  ".join([f"{k}: {season_avg.get(k,0):.1f}" for k in s_keys])
    draw.text((56,y0+52), s_vals, fill=(235,236,240), font=stat_font)

    y1 = y0 + 110
    draw.rounded_rectangle((40, y1, W-40, y1+90), radius=16, fill=(30,33,44), outline=(64,71,90))
    draw.text((56,y1+18), "Predicted Next (ML)", fill=(102,204,255), font=small_font)
    p_vals = "  |  ".join([f"{k}: {preds.get(k,'N/A')}" for k in s_keys])
    draw.text((56,y1+52), p_vals, fill=(235,236,240), font=stat_font)

    # Save to memory and offer download
    bio = BytesIO(); img.save(bio, format="PNG"); bio.seek(0)
    st.download_button("🖼️ Download Trading Card PNG", bio, file_name=f"{name.replace(' ','_')}_card.png", type="primary")

# ---------- Favorites page ----------
def favorites_page():
    st.markdown("### ⭐ Favorites — Next Game Predictions (ML)")
    favs = st.session_state.get("favorites", {})
    if not favs:
        st.info("No favorites yet.")
        return

    rows = []
    prog = st.progress(0.0, "Scouting favorites…")
    total = len(favs)
    for i,(name,pid) in enumerate(favs.items(), start=1):
        try:
            logs = safe_player_gamelogs(pid)
            preds = predict_next_game(logs)
            rows.append({"Player":name, "Player ID":pid, **{k:preds.get(k,"N/A") for k in PREDICT_TARGETS}})
        except Exception as e:
            rows.append({"Player":name, "Player ID":pid, **{k:"—" for k in PREDICT_TARGETS}})
        prog.progress(i/total, f"Processed {i}/{total}")
    prog.empty()

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    # links to open each
    for _,r in df.iterrows():
        cols = st.columns([0.55,0.45])
        with cols[0]:
            st.markdown(f"**{r['Player']}**")
        with cols[1]:
            if st.button("Open", key=f"fav_open_{r['Player ID']}"):
                st.session_state["selected_player"] = int(r["Player ID"])
                st.session_state["selected_name"] = r["Player"]
                st.session_state["tab"] = "Player"
                st.rerun()

# ---------- App orchestration ----------
if "favorites" not in st.session_state:
    st.session_state["favorites"] = {}
if "selected_player" not in st.session_state:
    st.session_state["selected_player"] = None
if "selected_name" not in st.session_state:
    st.session_state["selected_name"] = None
if "tab" not in st.session_state:
    st.session_state["tab"] = "Home"

sidebar_ui()

tab = st.session_state.get("tab","Home")
if tab == "Home":
    home_screen()
elif tab == "Player":
    pid = st.session_state.get("selected_player")
    pname = st.session_state.get("selected_name","")
    if pid is None:
        st.info("Pick a player from the sidebar to load their card.")
    else:
        player_card(pid, pname or "Player")
elif tab == "Favorites":
    favorites_page()