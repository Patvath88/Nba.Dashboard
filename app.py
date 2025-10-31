# app.py — Hot Shot Props | NBA Player Analytics (Favorites fix + stable)
import os, json, time, datetime as dt
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import requests

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playercareerstats, playergamelogs

# ---------------------------- Page config & theme ----------------------------
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
.leader-name a{color:#ffb4b4;text-decoration:none;font-weight:700}
.fav-chip{display:flex;align-items:center;justify-content:space-between;background:#121212;border:1px solid #1e1e1e;border-radius:10px;padding:8px 10px;margin-bottom:8px}
.fav-x{background:transparent;border:1px solid #2a2a2a;color:#ddd;border-radius:7px;padding:2px 8px;cursor:pointer}
@media (max-width:780px){
  .block-container{padding:0.6rem !important}
  [data-testid="stMetric"]{margin-bottom:10px}
}
</style>
""", unsafe_allow_html=True)

# ---------------------------- Globals ----------------------------
DEFAULT_TMP_DIR = "/tmp" if os.access("/", os.W_OK) else "."
FAV_PATH = os.path.join(DEFAULT_TMP_DIR, "favorites.json")

STATS_COLS = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']
PREDICT_TARGETS = ['PTS','REB','AST','FG3M']

NBA_TIMEOUT = 12  # keep under 12s per your environment
API_SLEEP = 0.15  # kinder pacing across multiple season calls

# ---------------------------- Helpers: favorites ----------------------------
def load_favorites() -> Dict[str, int]:
    """Always return a dict of {player_name: player_id}. Migrates old list -> {}."""
    if os.path.exists(FAV_PATH):
        try:
            with open(FAV_PATH, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                # if it was [] or anything else, toss and start fresh as dict
                return {}
        except Exception:
            return {}
    return {}

def save_favorites(favs: Dict[str, int]) -> None:
    try:
        with open(FAV_PATH, "w") as f:
            json.dump(favs, f, indent=2)
    except Exception:
        pass

# Ensure session favorites is a dict even if file was a list previously
if "favorites" not in st.session_state:
    favs = load_favorites()
    if not isinstance(favs, dict):
        favs = {}
    st.session_state.favorites = favs

# ---------------------------- Cached data fetchers ----------------------------
@st.cache_data(show_spinner=False, ttl=60*20)
def get_active_players_df() -> pd.DataFrame:
    plist = players.get_active_players()
    return pd.DataFrame(plist)

@st.cache_data(show_spinner=False, ttl=60*120)
def get_teams_df() -> pd.DataFrame:
    tlist = teams.get_teams()
    return pd.DataFrame(tlist)

@st.cache_data(show_spinner=True, ttl=60*10)
def fetch_player_career(player_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (career_by_season_df, full_career_gamelogs_df)."""
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=NBA_TIMEOUT)
        career_df = career.get_data_frames()[0]
    except Exception as e:
        st.error(f"Failed to load career: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # gather logs across seasons (respect short sleep + timeout)
    logs_all = []
    seasons = career_df['SEASON_ID'].tolist() if 'SEASON_ID' in career_df.columns else []
    for sid in seasons:
        try:
            gl = playergamelogs.PlayerGameLogs(player_id_nullable=player_id,
                                               season_nullable=sid,
                                               timeout=NBA_TIMEOUT).get_data_frames()[0]
            logs_all.append(gl)
            time.sleep(API_SLEEP)
        except Exception:
            # keep going; one miss shouldn't kill the page
            continue

    logs_df = pd.concat(logs_all, ignore_index=True) if logs_all else pd.DataFrame()
    if 'GAME_DATE' in logs_df.columns:
        logs_df['GAME_DATE'] = pd.to_datetime(logs_df['GAME_DATE'])
        logs_df = logs_df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
    return career_df, logs_df

# ---------------------------- Prediction (lightweight ML-style) ----------------------------
def predict_next_from_logs(logs_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if logs_df is None or logs_df.empty:
        return None
    df = logs_df.copy()
    # rolling means with fallback
    out = {}
    for stat in PREDICT_TARGETS:
        if stat not in df.columns:
            continue
        s = df[stat].astype(float)
        r5  = s.rolling(5,  min_periods=1).mean().iloc[:1].mean()   # last row after sort desc is row 0
        r10 = s.rolling(10, min_periods=1).mean().iloc[:1].mean()
        r20 = s.rolling(20, min_periods=1).mean().iloc[:1].mean()
        # blended
        pred = 0.4 * r5 + 0.3 * r10 + 0.2 * r20 + 0.1 * s.mean()
        out[stat] = round(float(pred), 2)
    return out if out else None

# ---------------------------- UI helpers ----------------------------
def player_headshot_url(player_id: int) -> str:
    # NBA media day headshot pattern
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

def simple_metric_row(title: str, values: Dict[str, Optional[float]]):
    st.subheader(title)
    c1, c2, c3, c4, c5 = st.columns(5)
    cells = [c1, c2, c3, c4, c5]
    keys = ["PTS", "REB", "AST", "FG3M", "MIN"]
    for i, k in enumerate(keys):
        v = values.get(k, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            cells[i].metric(k, "N/A")
        else:
            cells[i].metric(k, f"{float(v):.2f}")

def bar_block(df_row_like: Dict[str, float], title: str):
    # build long df for bars
    long = pd.DataFrame([
        {"Stat": k, "Value": float(v)} for k, v in df_row_like.items()
        if k in ["PTS","REB","AST","FG3M","MIN"]
    ])
    if long.empty:
        st.info("No data for charts.")
        return
    st.write(f"**{title}**")
    ch = alt.Chart(long).mark_bar().encode(
        x=alt.X("Stat:N", sort=None),
        y=alt.Y("Value:Q")
    )
    st.altair_chart(ch, use_container_width=True)

# ---------------------------- Sidebar ----------------------------
ap = get_active_players_df()
tm = get_teams_df()

with st.sidebar:
    st.header("Select Player")
    # unified search: by name or by team abbreviation
    q = st.text_input("Search player (name) or filter by team (e.g., BOS):", "")
    # favorites list (clean & minimal)
    st.markdown("### ⭐ Favorites")
    if isinstance(st.session_state.favorites, dict) and st.session_state.favorites:
        for name, pid in list(st.session_state.favorites.items()):
            cols = st.columns([0.75, 0.25])
            with cols[0]:
                st.write(f"• {name}")
            with cols[1]:
                if st.button("✕", key=f"fav_{pid}"):
                    st.session_state.favorites.pop(name, None)
                    save_favorites(st.session_state.favorites)
                    st.rerun()
    else:
        st.caption("No favorites yet.")

def resolve_player_selection(query: str) -> Optional[Tuple[str, int]]:
    df = ap.copy()
    if not query:
        return None
    query = query.strip()
    # team filter
    if len(query) in (2,3) and query.isalpha():
        abbr = query.upper()
        # show first player match on that team for quick load
        if "team_abbreviation" in df.columns:
            tmask = df["team_abbreviation"].str.upper().eq(abbr)
            if tmask.any():
                row = df[tmask].iloc[0]
                return row["full_name"], int(row["id"])
    # name search
    mask = df["full_name"].str.contains(query, case=False, regex=False)
    if mask.any():
        row = df[mask].iloc[0]
        return row["full_name"], int(row["id"])
    return None

# ---------------------------- Main ----------------------------
st.title("Hot Shot Props — NBA Player Analytics")
st.caption("💡 Use the sidebar to search any player (name) or filter by team (e.g., **BOS**).")

sel = resolve_player_selection(q)

if not sel:
    st.info("Pick a player from the sidebar to load their dashboard.")
else:
    player_name, player_id = sel
    st.header(player_name)

    colA, colB = st.columns([0.25, 0.75])
    with colA:
        # headshot
        st.image(player_headshot_url(player_id), caption="Media Day Headshot", use_container_width=True)
        # favorite toggle
        if st.button("⭐ Add to Favorites"):
            st.session_state.favorites[player_name] = int(player_id)
            save_favorites(st.session_state.favorites)
            st.success("Added to favorites.")
    with colB:
        # fetch data
        with st.spinner("Loading player data…"):
            career_df, logs_df = fetch_player_career(player_id)

        if career_df.empty:
            st.error("No career data available for this player (or API timeout).")
        else:
            # Current season (last season row in career_df is most recent)
            cur_row = career_df.iloc[-1].to_dict()
            # compute per-game for career_df row
            cur_vals = {}
            gp = float(cur_row.get("GP", 0) or 0)
            for k in ["PTS","REB","AST","FG3M","MIN"]:
                tot = float(cur_row.get(k, 0) or 0)
                cur_vals[k] = (tot / gp) if gp > 0 else np.nan

            simple_metric_row("Current Season Averages", cur_vals)
            bar_block(cur_vals, "Current Season (bars)")

            # Last game stats
            last_vals = {}
            if not logs_df.empty:
                lg = logs_df.iloc[0]
                for k in ["PTS","REB","AST","FG3M","MIN"]:
                    last_vals[k] = float(lg.get(k, np.nan))
                simple_metric_row("Last Game Stats", last_vals)
                bar_block(last_vals, "Last Game (bars)")
            else:
                st.info("No game logs found for last game metrics.")

            # Last 5 avg
            avg5 = {}
            if not logs_df.empty:
                sub = logs_df.head(5)
                for k in ["PTS","REB","AST","FG3M","MIN"]:
                    avg5[k] = float(sub[k].mean()) if k in sub.columns else np.nan
                simple_metric_row("Last 5 Games Averages", avg5)
                bar_block(avg5, "Last 5 (bars)")

            # Prediction (ML-lite)
            pred = predict_next_from_logs(logs_df)
            if pred:
                # include minutes if present in logs
                if "MIN" not in pred and "MIN" in logs_df.columns:
                    pred["MIN"] = round(float(logs_df["MIN"].head(10).mean()), 2)
                st.subheader("Predicted Next Game (ML)")
                # ensure the keys we show exist
                show_pred = {k: pred.get(k, np.nan) for k in ["PTS","REB","AST","FG3M","MIN"]}
                simple_metric_row("Predicted Next Game (ML)", show_pred)
                bar_block(show_pred, "Predicted (bars)")
            else:
                st.info("Prediction not available (insufficient logs).")

    # ---- Download snapshot (simple PNG of the metric tables as CSV text) ----
    st.divider()
    st.subheader("Download Snapshot")
    snap_df = pd.DataFrame({
        "Metric": ["Current PTS","Current REB","Current AST","Current 3PM","Current MIN"],
        "Value": [cur_vals.get("PTS"),cur_vals.get("REB"),cur_vals.get("AST"),cur_vals.get("FG3M"),cur_vals.get("MIN")]
    })
    csv = snap_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Snapshot CSV", csv, file_name=f"{player_name.replace(' ','_')}_snapshot.csv", mime="text/csv")