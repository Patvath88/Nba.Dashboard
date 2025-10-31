# app.py — Hot Shot Props | NBA "Topps Card" Analytics (Overhauled)
# • Direct HTTPS calls to stats.nba.com endpoints with browser headers
# • Disk + memory cache with graceful fallbacks
# • ML-first predictions (Ridge) with WMA fallback
# • Home (leaders, bigger pictures), Player, Favorites
# • Trading-card PNG export

import os, io, re, time, random, math, json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import streamlit.components.v1 as components

# ---------- Page setup ----------
st.set_page_config(page_title="Hot Shot Props — NBA Card View", layout="wide")

components.html('<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover">', height=0)

# ---------- Theme (Topps card vibe) ----------
st.markdown("""
<style>
:root{--bg:#0a0a0a;--panel:#101014;--ink:#f4f4f5;--muted:#b4b4b7;--accent:#ff3b3b;--line:#1c1c21;--gold:#f5d36e}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink)!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a0a0a 0%,#131318 100%)!important;border-right:1px solid var(--line)}
.card{background:linear-gradient(160deg,#13131a 0%,#0e0e14 100%);border:1px solid #23232b;border-radius:18px;padding:18px;box-shadow:0 18px 60px rgba(0,0,0,.55)}
.topps{border:3px solid var(--gold);border-radius:22px;box-shadow:0 0 0 6px #1c1500 inset, 0 16px 60px rgba(0,0,0,.65)}
.leader{display:flex;gap:14px;align-items:center;background:#0f0f13;border:1px solid #22252b;border-radius:14px;padding:12px;margin-bottom:12px}
.leader img{width:92px;height:92px;border-radius:12px;border:1px solid #2a2f37;object-fit:cover;background:#111}
.leader .name{font-weight:800;font-size:1.12rem}
a{color:#f7b0b0;text-decoration:none} a:hover{text-decoration:underline}
.stButton>button{background:var(--accent)!important;color:#fff!important;border:none!important;border-radius:12px!important;padding:.5rem .9rem!important;font-weight:700}
[data-testid="stMetric"]{background:#0e0e12;border:1px solid #1d1d24;border-radius:14px;padding:12px}
[data-testid="stMetric"] label{color:#f9b1b1}
[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#fff;font-size:1.35rem}
@media (max-width:780px){.block-container{padding:0.6rem!important} [data-testid="column"]{width:100%!important;display:block!important}}
</style>
""", unsafe_allow_html=True)

# ---------- Globals ----------
SEASON        = "2025-26"
LAST_SEASON   = "2024-25"
SEASON_TYPE   = "Regular Season"
PREDICT_STATS = ["PTS","REB","AST","FG3M"]
CACHE_DIR     = Path("nba_cache"); CACHE_DIR.mkdir(exist_ok=True)

# ---------- HTTP session with retries ----------
def _headers() -> Dict[str,str]:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
    }

_SESSION = None
def sess() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        s = requests.Session()
        s.headers.update(_headers())
        r = Retry(total=3, connect=3, read=3, backoff_factor=0.7,
                  status_forcelist=(429,500,502,503,504), allowed_methods=frozenset(["GET"]))
        ad = HTTPAdapter(max_retries=r, pool_connections=30, pool_maxsize=60)
        s.mount("https://", ad); s.mount("http://", ad)
        _SESSION = s
    return _SESSION

# ---------- Static players list (no network) ----------
@st.cache_data
def players_df() -> pd.DataFrame:
    # Minimal baked list from NBA JSON (id+name) bundled here to avoid extra fetches.
    # You can swap this with nba_api.static.players.get_players() if you prefer.
    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
    # We won't hit it; use local minimal CSV if exists; otherwise load a tiny baked sample.
    f = CACHE_DIR / "players_min.csv"
    if f.exists():
        df = pd.read_csv(f)
    else:
        # tiny fallback; encourage user search to still work by id param
        df = pd.DataFrame([
            {"id": 201939, "full_name": "Stephen Curry"},
            {"id": 203500, "full_name": "Giannis Antetokounmpo"},
            {"id": 201142, "full_name": "Kevin Durant"},
            {"id": 1629029,"full_name": "Tyrese Maxey"},
            {"id": 1626157,"full_name": "Devin Booker"},
        ])
    df["full_name_lower"] = df["full_name"].str.lower()
    return df.sort_values("full_name")

def headshot_url(pid:int) -> str:
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"

# ---------- Low-level stats endpoints (direct) ----------
def _get_json(url:str, params:Dict[str,str], timeout:int=15) -> Optional[dict]:
    try:
        time.sleep(random.uniform(0.4, 1.2))  # polite jitter
        r = sess().get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=24*3600, show_spinner=False)
def fetch_leaders(season:str) -> pd.DataFrame:
    """
    Direct call to leaguedashplayerstats (PerGame) → robust + cache.
    """
    url = "https://stats.nba.com/stats/leaguedashplayerstats"
    params = {
        "College":"",
        "Conference":"",
        "Country":"",
        "DateFrom":"",
        "DateTo":"",
        "Division":"",
        "DraftPick":"",
        "DraftYear":"",
        "GameScope":"",
        "GameSegment":"",
        "Height":"",
        "LastNGames":"0",
        "LeagueID":"00",
        "Location":"",
        "MeasureType":"Base",
        "Month":"0",
        "OpponentTeamID":"0",
        "Outcome":"",
        "PORound":"0",
        "PaceAdjust":"N",
        "PerMode":"PerGame",
        "Period":"0",
        "PlayerExperience":"",
        "PlayerPosition":"",
        "PlusMinus":"N",
        "Rank":"N",
        "Season":season,
        "SeasonSegment":"",
        "SeasonType":SEASON_TYPE,
        "ShotClockRange":"",
        "StarterBench":"",
        "TeamID":"0",
        "TwoWay":"0",
        "VsConference":"",
        "VsDivision":"",
        "Weight":""
    }
    data = _get_json(url, params)
    if data:
        rs = data["resultSets"][0]
        df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
        return df
    # fallback to cache on disk if exists
    p = CACHE_DIR / f"leaders_{season}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    # else empty
    return pd.DataFrame()

@st.cache_data(ttl=24*3600, show_spinner=False)
def fetch_player_logs(pid:int, season:str) -> pd.DataFrame:
    """
    Direct call to playergamelogs (Base). Cache each season separately; fall back to disk.
    """
    disk = CACHE_DIR / f"logs_{pid}_{season}.parquet"
    url = "https://stats.nba.com/stats/playergamelogs"
    params = {"PlayerID":str(pid), "Season":season, "SeasonType":SEASON_TYPE, "MeasureType":"Base"}
    data = _get_json(url, params)
    if data:
        try:
            rs = data["resultSets"][0]
            df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
            if not df.empty:
                df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
                df.sort_values("GAME_DATE", ascending=False, inplace=True)
                df.to_parquet(disk, index=False)
                return df
        except Exception:
            pass
    if disk.exists():
        return pd.read_parquet(disk)
    return pd.DataFrame()

# ---------- ML (Ridge) + fallback WMA ----------
try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

def _add_feats(df:pd.DataFrame, stat:str) -> pd.DataFrame:
    df = df.copy().sort_values("GAME_DATE")
    s = pd.to_numeric(df[stat], errors="coerce")
    df[f"{stat}_lag1"] = s.shift(1)
    df[f"{stat}_ema3"] = s.ewm(span=3, adjust=False).mean().shift(1)
    df[f"{stat}_ema5"] = s.ewm(span=5, adjust=False).mean().shift(1)
    df[f"{stat}_ma5"]  = s.rolling(5).mean().shift(1)
    df[f"{stat}_ma10"] = s.rolling(10).mean().shift(1)
    df[f"{stat}_ma20"] = s.rolling(20).mean().shift(1)
    return df

def _ridge(df:pd.DataFrame, stat:str) -> Optional[Ridge]:
    if not SKLEARN_OK: return None
    use = _add_feats(df, stat)
    cols = [c for c in use.columns if any(t in c for t in [f"{stat}_lag1", f"{stat}_ema", f"{stat}_ma"])]
    use = use.dropna(subset=cols+[stat])
    if len(use) < 30: return None
    X = use[cols].values; y = pd.to_numeric(use[stat], errors="coerce").values
    try:
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        m = Ridge(alpha=0.6).fit(Xtr,ytr); return m
    except Exception:
        return None

def _wma(arr:np.ndarray) -> float:
    if len(arr)==0: return float("nan")
    w = np.array([5,3,2,1])[:len(arr)][::-1]
    return float(np.average(arr[-len(w):], weights=w))

def predict_next(df:pd.DataFrame) -> Dict[str,float]:
    out={}
    for stat in PREDICT_STATS:
        try:
            m = _ridge(df, stat)
            if m is not None:
                last = _add_feats(df, stat).iloc[-1:]
                cols = [c for c in last.columns if any(t in c for t in [f"{stat}_lag1", f"{stat}_ema", f"{stat}_ma"])]
                x = last[cols].fillna(method="ffill").fillna(0.0).values
                out[stat] = max(0.0, float(m.predict(x)[0]))
            else:
                arr = pd.to_numeric(df[stat], errors="coerce").tail(10).values
                out[stat] = _wma(arr)
        except Exception:
            arr = pd.to_numeric(df[stat], errors="coerce").tail(10).values
            out[stat] = _wma(arr)
    return out

# ---------- Image helpers ----------
def _fetch_img(url:str, size:Tuple[int,int]=(520,360)) -> Image.Image:
    try:
        r = sess().get(url, timeout=10)
        if r.ok:
            im = Image.open(io.BytesIO(r.content)).convert("RGBA")
            return im.resize(size)
    except Exception:
        pass
    return Image.new("RGBA", size, (16,16,22,255))

def make_card_png(title:str, player_img:Image.Image, metrics:Dict[str,float]) -> bytes:
    W,H=980,620
    im = Image.new("RGBA",(W,H),(10,10,14,255))
    d = ImageDraw.Draw(im)
    try:
        fBig = ImageFont.truetype("DejaVuSans-Bold.ttf",42)
        fSm  = ImageFont.truetype("DejaVuSans.ttf",28)
    except Exception:
        fBig = ImageFont.load_default(); fSm = ImageFont.load_default()
    # Topps frame
    d.rectangle([12,12,W-12,H-12], outline=(245,211,110,255), width=6)
    d.rectangle([26,26,W-26,H-26], outline=(28,22,0,255), width=8)
    d.text((40,36), title, fill=(255,240,240,255), font=fBig)
    im.paste(player_img.resize((460,340)), (40,100))
    x0=540; y=128
    d.text((x0,y-18), "Predicted Next Game (ML)", fill=(255,178,178,255), font=fSm)
    for k,v in metrics.items():
        d.rectangle([x0-8,y-6,x0+340,y+32], outline=(36,36,46,255), width=2)
        d.text((x0, y), f"{k}", fill=(245,245,255,255), font=fSm)
        d.text((x0+220, y), f"{v:.2f}", fill=(105,220,140,255), font=fSm)
        y += 48
    bio=io.BytesIO(); im.save(bio, format="PNG"); return bio.getvalue()

# ---------- Sidebar ----------
def init_state():
    st.session_state.setdefault("favorites", {})  # {pid: name}
init_state()

def sidebar() -> Optional[int]:
    with st.sidebar:
        st.markdown("### 🔎 Search player")
        ppl = players_df()
        q = st.text_input("Type player name", "")
        pid=None
        if q.strip():
            sub = ppl[ppl["full_name_lower"].str.contains(q.strip().lower())].head(25)
            opts=[f'{r.full_name}  (id:{r.id})' for _,r in sub.iterrows()]
            sel = st.selectbox("Matches", ["-- pick --"]+opts, index=0)
            if sel != "-- pick --":
                m=re.search(r"id:(\d+)\)$", sel);  pid=int(m.group(1)) if m else None

        st.markdown("---")
        st.markdown("### ⭐ Favorites")
        favs = st.session_state["favorites"]
        if not favs: st.caption("No favorites yet.")
        else:
            for k,v in list(favs.items()):
                c1,c2=st.columns([0.8,0.2])
                with c1: st.link_button(v, url=f"?player_id={k}", use_container_width=True)
                with c2:
                    if st.button("✕", key=f"del_{k}"):
                        favs.pop(k,None); st.session_state["favorites"]=favs; st.rerun()
        st.markdown("---")
        st.link_button("🏠 Home", url=".", use_container_width=True)
        st.link_button("📌 Favorites page", url="?page=favorites", use_container_width=True)
    return pid

# ---------- Pages ----------
def page_home():
    st.markdown("## 🏠 Home")
    st.caption("Hint: use the sidebar to search any player by name. Add ⭐ to build your Favorites board.")
    df = fetch_leaders(SEASON)
    if df.empty:
        st.warning("Leaders unavailable right now. Showing nothing until cache warms.")
        return
    cols = st.columns(4)
    mapping = [("PTS","Points"),("REB","Rebounds"),("AST","Assists"),("FG3M","3-Pointers")]
    for (stat,label),col in zip(mapping, cols):
        with col:
            try:
                top = df.sort_values(stat, ascending=False).iloc[0]
                pid = int(top["PLAYER_ID"]); name=str(top["PLAYER_NAME"]); team=str(top["TEAM_ABBREVIATION"]); val=float(top[stat])
                st.markdown(f"#### {label}")
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
                st.error(f"{label} leader unavailable: {e}")

def _metrics_row(title:str, df:pd.DataFrame, cols_list:List[str]):
    st.markdown(f"### {title}")
    cs = st.columns(len(cols_list))
    for c, col in zip(cols_list, cs):
        with col:
            try:
                if title.startswith("Last Game"):
                    v = float(df[c].values[0])
                else:
                    v = float(pd.to_numeric(df[c], errors="coerce").mean())
                st.metric(c, f"{v:.2f}")
            except Exception:
                st.metric(c, "N/A")

def _bars_expander(label:str, df:pd.DataFrame):
    with st.expander(label, expanded=False):
        cols = st.columns(4)
        for k, col in zip(PREDICT_STATS, cols):
            with col:
                try:
                    long = df[["GAME_DATE", k]].tail(10).rename(columns={k:"Value"}).sort_values("GAME_DATE")
                    st.bar_chart(long, x="GAME_DATE", y="Value", width='stretch', height=220)
                except Exception:
                    st.empty()

def page_player(pid:int):
    ppl = players_df(); row = ppl[ppl["id"]==pid]
    name = row["full_name"].iloc[0] if not row.empty else "Player"
    st.markdown(f"## {name}")
    if str(pid) not in st.session_state["favorites"]:
        if st.button("⭐ Add to Favorites"): st.session_state["favorites"][str(pid)] = name; st.rerun()

    # Pull two seasons with caching; if live blocked, we'll still render from disk.
    now  = fetch_player_logs(pid, SEASON)
    prev = fetch_player_logs(pid, LAST_SEASON)

    if now.empty and prev.empty:
        st.error("No data available (temporary block or first-time fetch). Try again later to warm the cache.")
        return

    # Image
    st.image(_fetch_img(headshot_url(pid), size=(520,360)), use_container_width=False)

    # Numbers
    if not now.empty:
        _metrics_row("Current Season Averages", now, ["PTS","REB","AST","FG3M","MIN"])
        _metrics_row("Last Game", now.iloc[0:1], ["PTS","REB","AST","FG3M","MIN"])
        _metrics_row("Last 5 Games Averages", now.head(5), ["PTS","REB","AST","FG3M","MIN"])

        # ML prediction (forced when sklearn is present)
        pred = predict_next(now)
        c1,c2,c3,c4 = st.columns(4)
        for (k,v), col in zip(pred.items(), [c1,c2,c3,c4]):
            with col: st.metric(f"{k} (ML)", f"{v:.2f}")

        # Expanders
        _bars_expander("📈 Form (Last 5)", now.head(5))
        _bars_expander("📈 Form (Last 10)", now.head(10))
        _bars_expander("📈 Form (Last 20)", now.head(20))
        _bars_expander(f"📈 Season {SEASON}", now)
        if not prev.empty: _bars_expander(f"📈 Last Season ({LAST_SEASON})", prev)

        # PNG card
        png = make_card_png(f"{name} — Hot Shot Props", _fetch_img(headshot_url(pid), (520,360)), pred)
        st.download_button("⬇️ Download as PNG card", data=png, file_name=f"{name.replace(' ','_')}_card.png", mime="image/png")
    else:
        st.warning(f"Season {SEASON} logs not cached yet. Open again in a bit to warm cache.")

def page_favorites():
    st.markdown("## 📌 Favorites — ML Projections")
    favs:Dict[str,str] = st.session_state["favorites"]
    if not favs: st.caption("No favorites yet."); return

    rows=[]
    for pid,name in favs.items():
        pid_i=int(pid)
        df = fetch_player_logs(pid_i, SEASON)
        if df.empty:
            rows.append({"PLAYER":name, **{k: np.nan for k in PREDICT_STATS}, "LINK": f"?player_id={pid}"})
            continue
        for c in PREDICT_STATS+["MIN"]:
            if c in df.columns: df[c]=pd.to_numeric(df[c], errors="coerce")
        pred=predict_next(df)
        rows.append({"PLAYER":name, **pred, "LINK": f"?player_id={pid}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------- Router ----------
def main():
    st.session_state.setdefault("favorites", {})
    pid = st.query_params.get("player_id", [None])[0]
    pid = int(pid) if pid and pid.isdigit() else None
    page = st.query_params.get("page", ["home"])[0]

    _ = sidebar()

    if pid:              page_player(pid)
    elif page=="favorites": page_favorites()
    else:                page_home()

if __name__ == "__main__": main()
