# app.py — Hot Shot Props | NBA Player Analytics (Topps Card Edition)
# Patched: anti-rate-limit caching, stable leaders, ML-first predictions, Favorites page,
# trading-card theme, player expanders (L5/L10/L20/Season/Last Season), PNG card export.

import os, re, json, time, random, math, io
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import streamlit.components.v1 as components

# -------- nba_api (stable endpoints) --------
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import (
    playergamelogs,        # season game logs (stable)
    leaguedashplayerstats  # used to build leaders Per Game
)

# -------- Optional ML (force ML when present) --------
try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ===========================
# Page, Theme, Globals
# ===========================
st.set_page_config(page_title="Hot Shot Props — NBA Card View", layout="wide")

# Hide multipage nav if present
st.markdown("""
<style>
[data-testid="stSidebarNav"], [data-testid="stSidebarNavItems"], [data-testid="stSidebarHeader"]{
  display:none !important;
}
</style>
""", unsafe_allow_html=True)

# Mobile viewport
components.html("""
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover">
""", height=0)

# --- Topps card theme ---
st.markdown("""
<style>
:root{
  --bg:#0a0a0a; --panel:#101014; --ink:#f4f4f5; --muted:#b4b4b7; --accent:#ff3b3b;
  --line:#1c1c21; --gold:#f5d36e; --green:#22c55e; --red:#ef4444;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink)!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a0a0a 0%,#141418 100%)!important;border-right:1px solid var(--line);}
.card{background:linear-gradient(160deg,#13131a 0%,#0e0e14 100%);border:1px solid #23232b;border-radius:18px;padding:18px;box-shadow:0 18px 60px rgba(0,0,0,.55);}
.card-header{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;border:1px solid var(--line);background:#15151c;color:#f9d3d3;font-size:.8rem}
.badge.ok{color:#bbf7d0;border-color:#065f46;background:#032f22}
.badge.warn{color:#fde68a;border-color:#7c2d12;background:#3b0a0a}
.hr{border:0;border-top:1px solid var(--line);margin:.8rem 0;}
.topps{border:3px solid var(--gold); border-radius:22px; box-shadow:0 0 0 6px #1c1500 inset, 0 16px 60px rgba(0,0,0,.65);}
.leader{display:flex;gap:12px;align-items:center;background:#0f0f13;border:1px solid #22252b;border-radius:14px;padding:10px;margin-bottom:10px}
.leader img{width:70px;height:70px;border-radius:12px;border:1px solid #2a2f37;object-fit:cover;background:#111}
.leader .name{font-weight:800;font-size:1.05rem}
a { color:#f7b0b0; text-decoration:none; }
a:hover { text-decoration:underline; }
.stButton>button{background:var(--accent)!important;color:#fff!important;border:none!important;border-radius:12px!important;padding:.5rem .9rem!important;font-weight:700}
[data-testid="stMetric"]{background:#0e0e12;border:1px solid #1d1d24;border-radius:14px;padding:12px}
[data-testid="stMetric"] label{color:#f9b1b1}
[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#fff;font-size:1.35rem}
@media (max-width:780px){ .block-container{padding:0.6rem!important} [data-testid="column"]{width:100%!important;display:block!important} }
</style>
""", unsafe_allow_html=True)

# Common constants
SEASON = "2025-26"                  # Leaderboard season
SEASON_TYPE = "Regular Season"
PREDICT_STATS = ["PTS", "REB", "AST", "FG3M"]
CACHE_DIR = Path("nba_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ===========================
# Robust NBA session + caching
# ===========================
def _nba_headers() -> Dict[str, str]:
    return {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

_SESSION = None
def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update(_nba_headers())
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        retry = Retry(
            total=3, connect=3, read=3, backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"])
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=40)
        s.mount("https://", adapter); s.mount("http://", adapter)
        _SESSION = s
    return _SESSION


@st.cache_data(ttl=24*3600)
def fetch_player_logs(player_id: int, seasons: Tuple[str, ...]) -> pd.DataFrame:
    """Anti-rate-limit fetch with disk cache + random delays + CDN fallback."""
    s = _session()
    frames = []

    for season in seasons:
        cache_file = CACHE_DIR / f"{player_id}_{season}.parquet"

        # 1) Load from disk cache if present
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    frames.append(df)
                    continue
            except Exception:
                pass

        # 2) Primary endpoint (stats.nba.com) with polite delay
        try:
            time.sleep(random.uniform(0.5, 2.0))
            resp = s.get(
                "https://stats.nba.com/stats/playergamelogs",
                params={"PlayerID": str(player_id),
                        "Season": season,
                        "SeasonType": SEASON_TYPE,
                        "MeasureType": "Base"},
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()["resultSets"][0]
            df = pd.DataFrame(data["rowSet"], columns=data["headers"])
            if not df.empty:
                df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
                df["SEASON"] = season
                df.to_parquet(cache_file, index=False)
                frames.append(df)
                continue
        except Exception:
            # warn once per call; keep quiet for UX
            pass

        # 3) CDN fallback — best-effort (structure varies; may be empty)
        try:
            cdn_url = f"https://cdn.nba.com/static/json/liveData/playerDetails/NBAPlayerDetails_{player_id}.json"
            r = s.get(cdn_url, timeout=10)
            if r.ok:
                j = r.json()
                # This fallback may not contain logs; keep as placeholder
                df = pd.DataFrame()
                df.to_parquet(cache_file, index=False)
        except Exception:
            pass

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.sort_values("GAME_DATE", ascending=False, inplace=True)
        return out

    return pd.DataFrame()


@st.cache_data(ttl=3*3600)
def fetch_leaderboard(season: str) -> pd.DataFrame:
    """Use LeagueDashPlayerStats (stable) for PerGame leaders."""
    res = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=SEASON_TYPE,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame"
    )
    df = res.get_data_frames()[0]
    return df


@st.cache_data(ttl=24*3600)
def all_players_df() -> pd.DataFrame:
    ppl = pd.DataFrame(nba_players.get_players())
    ppl["full_name_lower"] = ppl["full_name"].str.lower()
    return ppl.sort_values("full_name")


# ===========================
# Helpers: images & cards
# ===========================
def headshot_url(player_id: int) -> str:
    # nba cdn headshots
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

def team_logo_url(team_abbr: str) -> str:
    return f"https://cdn.nba.com/logos/nba/{team_abbr}/primary/L/logo.svg"  # svg (not used on card image)

def download_image(url: str, size: Tuple[int,int]=(420,300)) -> Image.Image:
    try:
        r = _session().get(url, timeout=10)
        if r.ok:
            im = Image.open(io.BytesIO(r.content)).convert("RGBA")
            return im.resize(size)
    except Exception:
        pass
    # fallback empty
    im = Image.new("RGBA", size, (16,16,20,255))
    return im

def make_png_card(title: str, player_img: Image.Image, metrics: Dict[str, float]) -> bytes:
    """Render a trading-card PNG with player image + 4 metrics."""
    W,H = 980, 620
    card = Image.new("RGBA", (W,H), (10,10,14,255))
    draw = ImageDraw.Draw(card)

    # Topps frame
    draw.rectangle([12,12,W-12,H-12], outline=(245,211,110,255), width=6)
    draw.rectangle([26,26,W-26,H-26], outline=(28,22,0,255), width=8)

    # Title
    try:
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 28)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((40,36), title, fill=(255,240,240,255), font=font_big)

    # Player image
    img = player_img.resize((460,340))
    card.paste(img, (40,100))

    # Metrics panel
    y0 = 100
    x0 = 540
    draw.text((x0, y0-8), "Predicted Next Game (ML)", fill=(255,178,178,255), font=font_small)
    y = y0+28
    for k,v in metrics.items():
        draw.rectangle([x0-8,y-6,x0+340,y+32], outline=(36,36,46,255), width=2)
        draw.text((x0, y), f"{k}", fill=(245,245,255,255), font=font_small)
        draw.text((x0+220, y), f"{v:.2f}", fill=(105,220,140,255), font=font_small)
        y += 48

    bio = io.BytesIO()
    card.save(bio, format="PNG")
    return bio.getvalue()


# ===========================
# Features & ML
# ===========================
def add_roll_feats(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("GAME_DATE")
    s = df[stat].astype(float)
    df[f"{stat}_lag1"] = s.shift(1)
    df[f"{stat}_ema3"] = s.ewm(span=3, adjust=False).mean().shift(1)
    df[f"{stat}_ema5"] = s.ewm(span=5, adjust=False).mean().shift(1)
    df[f"{stat}_ma5"]  = s.rolling(5).mean().shift(1)
    df[f"{stat}_ma10"] = s.rolling(10).mean().shift(1)
    df[f"{stat}_ma20"] = s.rolling(20).mean().shift(1)
    return df

def train_ridge_for_stat(df: pd.DataFrame, stat: str) -> Optional[Ridge]:
    if not SKLEARN_OK:
        return None
    use = add_roll_feats(df, stat)
    feat_cols = [c for c in use.columns if any(t in c for t in [f"{stat}_lag1", f"{stat}_ema", f"{stat}_ma"])]
    use = use.dropna(subset=feat_cols+[stat])
    if len(use) < 30:
        return None
    X = use[feat_cols].values
    y = use[stat].values
    try:
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        model = Ridge(alpha=0.5).fit(Xtr,ytr)
        return model
    except Exception:
        return None

def predict_next(df: pd.DataFrame) -> Dict[str,float]:
    """Try ML first; fallback to Weighted Moving Average (w=5,3,2,1)."""
    out = {}
    # Prepare WMA weights
    def wma(arr):
        w = np.array([5,3,2,1])[:len(arr)][::-1]
        return float(np.average(arr[-len(w):], weights=w))
    for stat in PREDICT_STATS:
        try:
            m = train_ridge_for_stat(df, stat)
            if m is not None:
                feats = add_roll_feats(df, stat).iloc[-1:]
                feat_cols = [c for c in feats.columns if any(t in c for t in [f"{stat}_lag1", f"{stat}_ema", f"{stat}_ma"])]
                x = feats[feat_cols].fillna(method="ffill").fillna(0.0).values
                pred = float(m.predict(x)[0])
                out[stat] = max(0.0, pred)  # ML
            else:
                # fallback WMA on last 10 (using stat col)
                arr = df[stat].astype(float).tail(10).values
                out[stat] = wma(arr) if len(arr) else float("nan")
        except Exception:
            arr = df[stat].astype(float).tail(10).values
            out[stat] = wma(arr) if len(arr) else float("nan")
    return out


# ===========================
# Sidebar: search + favorites
# ===========================
def init_state():
    st.session_state.setdefault("favorites", {})  # {player_id: name}

init_state()

def sidebar_ui() -> Optional[int]:
    with st.sidebar:
        st.markdown("### 🔎 Search player")
        ppl = all_players_df()
        name = st.text_input("Type a player's name…", value="", label_visibility="collapsed")
        player_id = None
        if name.strip():
            sub = ppl[ppl["full_name_lower"].str.contains(name.strip().lower())].head(20)
            options = [f'{row.full_name}  (id:{row.id})' for _, row in sub.iterrows()]
            sel = st.selectbox("Results", ["-- pick --"]+options, index=0, label_visibility="collapsed")
            if sel != "-- pick --":
                m = re.search(r"id:(\d+)\)$", sel)
                if m: player_id = int(m.group(1))

        st.markdown("---")
        st.markdown("### ⭐ Favorites")
        favs: Dict[str,str] = st.session_state["favorites"]
        if not favs:
            st.caption("No favorites yet.")
        else:
            for pid, nm in list(favs.items()):
                c1, c2 = st.columns([0.8, 0.2])
                with c1:
                    st.link_button(nm, url=f"?player_id={pid}", use_container_width=True)
                with c2:
                    if st.button("✕", key=f"del_{pid}"):
                        favs.pop(pid, None)
                        st.session_state["favorites"] = favs
                        st.rerun()
        st.markdown("---")
        st.link_button("🏠 Home", url=".", use_container_width=True)
        st.link_button("📌 Favorites page", url="?page=favorites", use_container_width=True)
    return player_id


# ===========================
# Home page (leaders)
# ===========================
def render_home():
    st.markdown("## 🏠 Home")
    st.caption("Hint: use the sidebar to search any player by name. Add ⭐ to build your Favorites board.")

    df = fetch_leaderboard(SEASON)

    cols = st.columns(4)
    leaders = {
        "PTS": ("Points", 0),
        "REB": ("Rebounds", 1),
        "AST": ("Assists", 2),
        "FG3M": ("3-Pointers", 3),
    }
    for stat, (label, i) in leaders.items():
        with cols[i]:
            try:
                top = df.sort_values(stat, ascending=False).iloc[0]
                pid = int(top["PLAYER_ID"])
                name = str(top["PLAYER_NAME"])
                team = str(top["TEAM_ABBREVIATION"])
                val = float(top[stat])

                st.markdown(f"#### {label}")
                with st.container(border=True):
                    # Big leader tile
                    st.markdown(f"""
<div class="leader">
  <img src="{headshot_url(pid)}" />
  <div>
    <div class="name"><a href="?player_id={pid}">{name}</a></div>
    <div style="color:#b9bbbe">{team}</div>
    <div style="margin-top:6px">Avg: <b>{val:.2f}</b> ({SEASON})</div>
  </div>
</div>
""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Leaders {label} unavailable: {e}")


# ===========================
# Player page
# ===========================
def render_section_bars(title: str, df: pd.DataFrame):
    with st.expander(title, expanded=False):
        ccols = st.columns(4)
        for j, k in enumerate(PREDICT_STATS):
            with ccols[j]:
                try:
                    long = df[["GAME_DATE", k]].tail(10).rename(columns={k:"Value"})
                    long = long.sort_values("GAME_DATE")
                    st.bar_chart(long, x="GAME_DATE", y="Value", width='stretch', height=220)
                except Exception:
                    st.empty()

def render_player(player_id: int):
    ppl = all_players_df()
    row = ppl[ppl["id"]==player_id]
    name = row["full_name"].iloc[0] if not row.empty else "Player"

    st.markdown(f"## {name}")
    if str(player_id) not in st.session_state["favorites"]:
        if st.button("⭐ Add to Favorites"):
            st.session_state["favorites"][str(player_id)] = name
            st.experimental_rerun()

    # Seasons to pull (current + last)
    seasons = (SEASON, "2024-25")
    logs = fetch_player_logs(player_id, seasons)
    if logs.empty:
        st.error("Failed to load from nba_api (likely a temporary block). Try again shortly.")
        return

    # Ensure numeric columns
    for c in PREDICT_STATS + ["MIN"]:
        if c in logs.columns:
            logs[c] = pd.to_numeric(logs[c], errors="coerce")

    # Header image
    img = download_image(headshot_url(player_id), size=(520,360))
    st.image(img, caption="", use_container_width=False)

    # Metrics rows
    st.markdown("### Current Season Averages")
    now = logs[logs["SEASON"]==SEASON]
    if now.empty:
        st.warning(f"Logs unavailable for {SEASON}.")
    else:
        c1,c2,c3,c4,c5 = st.columns(5)
        for c, col in zip(["PTS","REB","AST","FG3M","MIN"], [c1,c2,c3,c4,c5]):
            with col:
                st.metric(c, f'{now[c].mean():.2f}')

    st.markdown("### Last Game")
    last = logs.iloc[0:1]
    c1,c2,c3,c4,c5 = st.columns(5)
    for c, col in zip(["PTS","REB","AST","FG3M","MIN"], [c1,c2,c3,c4,c5]):
        with col:
            val = float(last[c].values[0]) if c in last.columns and not last.empty else float("nan")
            st.metric(c, f'{val:.2f}' if not math.isnan(val) else "N/A")

    st.markdown("### Last 5 Games Averages")
    l5 = logs.head(5)
    c1,c2,c3,c4,c5 = st.columns(5)
    for c, col in zip(["PTS","REB","AST","FG3M","MIN"], [c1,c2,c3,c4,c5]):
        with col:
            st.metric(c, f'{l5[c].mean():.2f}')

    # ML prediction (forced when sklearn is available)
    st.markdown("### Predicted Next Game (ML)")
    pred = predict_next(now if not now.empty else logs)
    c1,c2,c3,c4 = st.columns(4)
    for (k,v), col in zip(pred.items(), [c1,c2,c3,c4]):
        with col:
            st.metric(f"{k} (ML)", f"{v:.2f}")

    # Expanders (form sections)
    render_section_bars("📈 Form (Last 5)", logs.head(5))
    render_section_bars("📈 Form (Last 10)", logs.head(10))
    render_section_bars("📈 Form (Last 20)", logs.head(20))
    if not now.empty:
        render_section_bars(f"📈 Season {SEASON}", now)
    last_season = logs[logs["SEASON"]=="2024-25"]
    if not last_season.empty:
        render_section_bars("📈 Last Season", last_season)

    # Download PNG (card)
    png = make_png_card(f"{name} — Hot Shot Props", img, pred)
    st.download_button("⬇️ Download as PNG card", data=png, file_name=f"{name.replace(' ','_')}_card.png", mime="image/png")


# ===========================
# Favorites page
# ===========================
def render_favorites_page():
    st.markdown("## 📌 Favorites — ML Projections")
    favs: Dict[str,str] = st.session_state["favorites"]
    if not favs:
        st.caption("No favorites saved yet.")
        return

    rows = []
    for pid, nm in favs.items():
        try:
            logs = fetch_player_logs(int(pid), (SEASON,))
            if logs.empty:
                rows.append({"PLAYER": nm, **{k: np.nan for k in PREDICT_STATS}})
            else:
                for c in PREDICT_STATS + ["MIN"]:
                    if c in logs.columns:
                        logs[c] = pd.to_numeric(logs[c], errors="coerce")
                pred = predict_next(logs)
                rows.append({"PLAYER": nm, **pred, "LINK": f"?player_id={pid}"})
        except Exception:
            rows.append({"PLAYER": nm, **{k: np.nan for k in PREDICT_STATS}})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.caption("Tip: click a player's name in the sidebar to open their page.")


# ===========================
# Router
# ===========================
def main():
    pid = None
    q = st.query_params
    page = q.get("page", ["home"])[0]
    if "player_id" in q:
        try:
            pid = int(q.get("player_id")[0])
            page = "player"
        except Exception:
            pid = None

    _ = sidebar_ui()

    if page == "player" and pid:
        render_player(pid)
    elif page == "favorites":
        render_favorites_page()
    else:
        render_home()

if __name__ == "__main__":
    main()