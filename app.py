# app.py — Hot Shot Props | NBA Player Analytics (2025–26 Per-Game Leaders)
# - Home shows 2025–26 PER-GAME leaders (PTS/REB/AST/3PM) with headshots + links
# - Player page: headshot, season averages, last-5 form, WMA prediction (labeled), last-10 table
# - Robust caching and short retries to reduce timeouts
# - Uses width='stretch' (no deprecated use_container_width)
# - No balldontlie anywhere

import os
import time
import math
import pandas as pd
import streamlit as st
from typing import Optional, Tuple

# nba_api
from nba_api.stats.static import players as static_players, teams as static_teams
from nba_api.stats.endpoints import leagueleaders, playergamelogs

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

# -------------- Caching helpers ---------------
@st.cache_data(ttl=60*30, show_spinner=False)
def _players_index():
    # Map id->player & name->id
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
    # CDN path commonly used by nba.com
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
                per_mode_simple="PerGame",        # <--- Averages (NOT totals)
                timeout=10
            )
            df = obj.get_data_frames()[0]
            key = "FG3M" if stat.upper() in ("FG3M", "3PM") else stat.upper()
            if key in df.columns:
                df = df.sort_values(key, ascending=False).reset_index(drop=True)
            return df
        except Exception as e:
            last_exc = e
            time.sleep(0.75)
    raise last_exc

def leader_card(df: pd.DataFrame, label: str, key_prefix: str):
    if df is None or df.empty:
        st.error(f"{label} unavailable.")
        return
    row = df.iloc[0].to_dict()
    name = row.get("PLAYER", "Unknown")
    team = row.get("TEAM", "")
    # Map to player_id
    by_id, _ = _players_index()
    pid = None
    for pid_cand, pdata in by_id.items():
        if pdata.get("full_name", "").lower() == name.lower():
            pid = pid_cand
            break

    # Figure stat value
    stat_val = None
    for k in ("PTS","REB","AST","FG3M","3PM"):
        if k in df.columns and k in row:
            stat_val = row[k]
    stat_str = f"{stat_val:.2f}" if isinstance(stat_val,(int,float)) and not math.isnan(stat_val) else "—"

    # Render
    col_img, col_text = st.columns([1, 4], gap="small")
    with col_img:
        st.markdown('<div class="leader-img">', unsafe_allow_html=True)
        if pid:
            # render as fixed width image (avoid deprecated use_container_width)
            st.image(headshot_url(pid), width=64)
        else:
            st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_text:
        # Name → player page link if resolvable
        if pid:
            st.markdown(f"""<div class="leader-name">
                <a href="?player_id={pid}">{name}</a>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="leader-name">{name}</div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="leader-meta">{team}</div>""", unsafe_allow_html=True)
        st.metric(label=label, value=stat_str)

# -------------- Player logs + WMA -------------
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
            time.sleep(0.75)
    raise last_exc

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

# -------------- Routing -----------------------
def page_home():
    st.title("🏠 Home")
    st.caption("Hint: use the sidebar to search any player by name. Add ⭐ to build your Favorites board.")

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
                leader_card(df, label, key_prefix=f"lead_{stat}")
            except Exception as e:
                st.error(f"Leaders {stat} unavailable: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

def page_player(player_id: int):
    by_id, _ = _players_index()
    pmeta = by_id.get(player_id, {})
    pname = pmeta.get("full_name", f"Player {player_id}")
    st.header(pname)

    # Headshot + metrics
    c1, c2 = st.columns([1,3])
    with c1:
        st.image(headshot_url(player_id), width=260)
    with c2:
        with st.spinner("Preparing metrics..."):
            try:
                logs = player_logs(player_id)
            except Exception as e:
                st.error(f"Failed to load from nba_api (likely a temporary block). Try again in a minute.\n\n{e}")
                return

        if logs is None or logs.empty:
            st.info("No games found this season.")
            return

        # Current season per-game averages
        want = ["PTS","REB","AST","FG3M","MIN","FGA","FG3A","FTA","TOV"]
        present = [w for w in want if w in logs.columns]
        season_avg = logs[present].mean(numeric_only=True)

        st.subheader("Current Season Averages")
        mc = st.columns(4)
        def _met(c, k):
            v = season_avg.get(k)
            v = f"{v:.2f}" if isinstance(v, (int,float)) and not math.isnan(v) else "—"
            c.metric(k, v)

        _met(mc[0],"PTS"); _met(mc[1],"REB"); _met(mc[2],"AST"); _met(mc[3],"FG3M")

        # Last 5 averages
        st.subheader("Recent Form (Last 5)")
        last5 = logs.head(5)[present].mean(numeric_only=True)
        mc2 = st.columns(4)
        for i,k in enumerate(("PTS","REB","AST","FG3M")):
            v = last5.get(k)
            s = f"{v:.2f}" if isinstance(v,(int,float)) and not math.isnan(v) else "—"
            mc2[i].metric(k, s)

        # WMA prediction (labeled)
        st.subheader("Predicted Next Game (WMA)")
        pred = predict_wma(logs)
        mc3 = st.columns(4)
        for i,k in enumerate(("PTS","REB","AST","FG3M")):
            v = pred.get(k)
            s = f"{v:.2f} (WMA)" if isinstance(v,(int,float)) and not math.isnan(v) else "—"
            mc3[i].metric(k, s)

        # Table: last 10 games
        st.subheader("Breakdown (Last 10 Games)")
        show_cols = ["GAME_DATE","MATCHUP","WL","PTS","REB","AST","FG3M","MIN"]
        show_cols = [c for c in show_cols if c in logs.columns]
        st.dataframe(logs.head(10)[show_cols], width='stretch')

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

# -------------- Main --------------------------
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