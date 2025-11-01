# app.py — Hot Shot Props | NBA Player Projections (Streamlit + nba_api)
# - Search any player
# - Live stats (recent logs, season/last season)
# - Progressive "regression-proof" projection model (PTS/REB/AST/3PM) for next game
# - Caching + retry logic for nba_api calls
# - Clean, mobile-friendly UI with charts

import os
import time
import math
import json
import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# nba_api
from nba_api.stats.static import players as static_players, teams as static_teams
from nba_api.stats.endpoints import (
    playergamelogs, playercareerstats, commonplayerinfo,
    leaguegamelog, scoreboardv2, leaguedashteamstats
)
from nba_api.stats.library.parameters import SeasonAll

# --------------------------- Streamlit Page Config ---------------------------
st.set_page_config(
    page_title="NBA Live Player Projections — Hot Shot Props",
    page_icon="🏀",
    layout="wide",
)

alt.themes.enable("none")
st.markdown(
    """
    <style>
      .small {font-size: 0.8rem; color: #888;}
      .ok {color:#16a34a;font-weight:600}
      .warn {color:#b45309;font-weight:600}
      .bad {color:#dc2626;font-weight:600}
      .metric-card {padding:.75rem 1rem;border:1px solid #e5e7eb;border-radius:10px;background:#fff}
      .pill {padding:.15rem .5rem;border:1px solid #e5e7eb;border-radius:999px;background:#f9fafb;font-size:.75rem}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Utilities ---------------------------

HEADERS = {
    # nba.com headers needed to avoid 403s/blocks
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
}

def _retry_nba_api(func, *args, tries=3, sleep=1.25, **kwargs):
    last_exc = None
    for i in range(tries):
        try:
            return func(*args, headers=HEADERS, **kwargs)
        except Exception as e:
            last_exc = e
            time.sleep(sleep * (i + 1))
    if last_exc:
        raise last_exc

@st.cache_data(show_spinner=False, ttl=60*30)  # 30 mins
def get_players_df() -> pd.DataFrame:
    plist = static_players.get_players()
    df = pd.DataFrame(plist)
    return df

@st.cache_data(show_spinner=False, ttl=60*30)
def get_teams_df() -> pd.DataFrame:
    tlist = static_teams.get_teams()
    return pd.DataFrame(tlist)

def to_season_string(date: dt.date) -> str:
    # NBA seasons like "2024-25"
    year = date.year
    # Season typically starts Oct — if before August, it's prior season end
    if date.month >= 8:
        start = year
        end = (year + 1) % 100
    else:
        start = year - 1
        end = year % 100
    return f"{start}-{end:02d}"

@st.cache_data(show_spinner=False, ttl=60*10)
def get_player_info(player_id: int) -> Dict:
    res = _retry_nba_api(commonplayerinfo.CommonPlayerInfo, player_id=player_id)
    df = res.get_data_frames()[0]
    return df.iloc[0].to_dict()

@st.cache_data(show_spinner=False, ttl=60*10)
def get_player_game_logs(player_id: int, season: str, last_n_games: Optional[int]=None) -> pd.DataFrame:
    gl = _retry_nba_api(
        playergamelogs.PlayerGameLogs,
        player_id_nullable=str(player_id),
        season_nullable=season,
        season_type_nullable="Regular Season",
    )
    df = gl.get_data_frames()[0]
    if last_n_games:
        return df.sort_values("GAME_DATE", ascending=False).head(last_n_games).reset_index(drop=True)
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False, ttl=60*10)
def get_player_last_season_logs(player_id: int, current_season: str) -> pd.DataFrame:
    # current_season like "2024-25" -> last season "2023-24"
    base = int(current_season.split("-")[0])
    last_season = f"{base-1}-{(base-1+1)%100:02d}"
    return get_player_game_logs(player_id, last_season)

@st.cache_data(show_spinner=False, ttl=60*10)
def get_league_team_defense(season: str) -> pd.DataFrame:
    # Opponent defensive proxy: Team defensive rating + opponent-allowed stats
    # Using leaguedashteamstats with Opponent stats split
    dash = _retry_nba_api(
        leaguedashteamstats.LeagueDashTeamStats,
        season=season,
        measure_type_detailed_defense="Opponent",
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
    )
    df = dash.get_data_frames()[0]
    # Keep helpful opponent-allow per game cols
    cols_keep = ["TEAM_ID", "TEAM_NAME", "OPP_PTS", "OPP_FG3M", "OPP_AST", "OPP_REB"]
    return df[cols_keep].rename(columns={
        "OPP_PTS":"OPP_ALLOW_PTS",
        "OPP_FG3M":"OPP_ALLOW_3PM",
        "OPP_AST":"OPP_ALLOW_AST",
        "OPP_REB":"OPP_ALLOW_REB"
    })

@st.cache_data(show_spinner=False, ttl=60*10)
def get_scoreboard(target_date: dt.date) -> pd.DataFrame:
    sb = _retry_nba_api(scoreboardv2.ScoreboardV2, game_date=target_date.strftime("%Y-%m-%d"))
    games = sb.get_data_frames()[0]
    return games

def find_next_opponent_team_id(player_info: Dict, season: str) -> Optional[int]:
    """Try to find the player's next opponent in the next 7 days via Scoreboard."""
    team_id = player_info.get("TEAM_ID")
    if not team_id or pd.isna(team_id):
        return None
    today = dt.date.today()
    for d in range(0, 8):
        day = today + dt.timedelta(days=d)
        try:
            df = get_scoreboard(day)
        except Exception:
            continue
        # Games dataframe has columns: "GAME_ID","GAMECODE","HOME_TEAM_ID","VISITOR_TEAM_ID", ...
        if df.empty:
            continue
        # If player's team is home or away, opponent is the other side
        mask = (df["HOME_TEAM_ID"] == team_id) | (df["VISITOR_TEAM_ID"] == team_id)
        if mask.any():
            row = df[mask].iloc[0]
            if row["HOME_TEAM_ID"] == team_id:
                return int(row["VISITOR_TEAM_ID"])
            else:
                return int(row["HOME_TEAM_ID"])
    return None

def h2h_subset(df_logs: pd.DataFrame, opp_team_id: Optional[int]) -> pd.DataFrame:
    if opp_team_id is None or df_logs is None or df_logs.empty:
        return pd.DataFrame()
    # GAME_MATCHUP sample: "LAL @ GSW" — extract opponent by name not id; map id->abbrev
    # We'll approximate: use TEAM_ABBREVIATION and OPPONENT_TEAM_ID if present
    cols = [c.upper() for c in df_logs.columns]
    # nba_api PlayerGameLogs already includes OPPONENT_TEAM_ID
    if "OPPONENT_TEAM_ID" in df_logs.columns:
        return df_logs[df_logs["OPPONENT_TEAM_ID"] == opp_team_id].copy()
    return pd.DataFrame()  # fallback if schema changes

def safe_mean(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else 0.0

def var_aware_weight(n: int, variance: float, cap_low=1e-6) -> float:
    # Larger n and lower variance -> more weight
    v = max(variance, cap_low)
    return n / (v + 1.0)

def james_stein_shrink(player_mean: float, league_mean: float, n: int, var: float) -> float:
    # Classic shrinkage factor toward league mean to reduce overfit for small samples
    # k ~ variance / (variance + n)
    var = max(var, 1e-6)
    k = var / (var + max(n, 1))
    return (1 - k) * player_mean + k * league_mean

def recent_windows(df: pd.DataFrame, windows=(5,10,20)) -> Dict[int, Dict[str, float]]:
    out = {}
    for w in windows:
        sub = df.head(w)  # df is sorted DESC by GAME_DATE
        if sub.empty:
            out[w] = {"PTS":0,"REB":0,"AST":0,"FG3M":0,"N":0,"VAR_PTS":1,"VAR_REB":1,"VAR_AST":1,"VAR_3PM":1}
        else:
            out[w] = {
                "PTS": safe_mean(sub["PTS"]),
                "REB": safe_mean(sub["REB"]),
                "AST": safe_mean(sub["AST"]),
                "FG3M": safe_mean(sub["FG3M"]),
                "N": len(sub),
                "VAR_PTS": float(np.var(pd.to_numeric(sub["PTS"], errors="coerce").dropna(), ddof=1)) if len(sub)>1 else 1.0,
                "VAR_REB": float(np.var(pd.to_numeric(sub["REB"], errors="coerce").dropna(), ddof=1)) if len(sub)>1 else 1.0,
                "VAR_AST": float(np.var(pd.to_numeric(sub["AST"], errors="coerce").dropna(), ddof=1)) if len(sub)>1 else 1.0,
                "VAR_3PM": float(np.var(pd.to_numeric(sub["FG3M"], errors="coerce").dropna(), ddof=1)) if len(sub)>1 else 1.0,
            }
    return out

def league_baselines(season_team_opp: pd.DataFrame) -> Dict[str, float]:
    # League means of opponent-allowed categories (proxy for neutral baseline)
    if season_team_opp is None or season_team_opp.empty:
        return {"PTS": 112.0, "REB": 43.0, "AST": 25.0, "FG3M": 12.0}  # generic fallbacks
    return {
        "PTS": safe_mean(season_team_opp["OPP_ALLOW_PTS"]),
        "REB": safe_mean(season_team_opp["OPP_ALLOW_REB"]),
        "AST": safe_mean(season_team_opp["OPP_ALLOW_AST"]),
        "FG3M": safe_mean(season_team_opp["OPP_ALLOW_3PM"]),
    }

def opponent_adjustment(opp_row: Optional[pd.Series]) -> Dict[str, float]:
    # Convert opponent allowed per game into an adjustment multiplier around ~1.0
    # Lower allowed -> tougher -> <1 multiplier; higher allowed -> easier -> >1
    if opp_row is None or len(opp_row)==0:
        return {"PTS":1.0,"REB":1.0,"AST":1.0,"FG3M":1.0}
    # Calibrate around league-ish anchors
    anchors = {"PTS":112.0,"REB":43.0,"AST":25.0,"FG3M":12.0}
    adj = {}
    for k, col in [("PTS","OPP_ALLOW_PTS"),("REB","OPP_ALLOW_REB"),("AST","OPP_ALLOW_AST"),("FG3M","OPP_ALLOW_3PM")]:
        val = float(opp_row[col]) if col in opp_row else anchors[k]
        adj[k] = max(0.85, min(1.15, val / anchors[k]))  # clamp effect ±15%
    return adj

def ensemble_projection(
    df_cur: pd.DataFrame,
    df_last_season: pd.DataFrame,
    df_h2h: pd.DataFrame,
    season_opp_df: pd.DataFrame,
    opp_team_id: Optional[int],
) -> Dict[str, float]:
    """
    Progressive, regression-resistant blend:
    - Recency windows: L5/L10/L20 (variance-aware)
    - Current season average
    - Last season average
    - H2H vs upcoming opponent (small weight unless sample > 3)
    - James-Stein shrinkage toward league baselines
    - Opponent defensive adjustment (opponent-allowed stats)
    """
    # Sort cur df once
    cur = df_cur.copy()
    if cur.empty:
        return {"PTS":0,"REB":0,"AST":0,"FG3M":0}

    # Windows
    wins = recent_windows(cur, (5,10,20))

    # Season means
    cur_pts = safe_mean(cur["PTS"]); cur_reb = safe_mean(cur["REB"])
    cur_ast = safe_mean(cur["AST"]); cur_3pm = safe_mean(cur["FG3M"])
    cur_var = {
        "PTS": float(np.var(pd.to_numeric(cur["PTS"], errors="coerce").dropna(), ddof=1)) if len(cur)>1 else 1.0,
        "REB": float(np.var(pd.to_numeric(cur["REB"], errors="coerce").dropna(), ddof=1)) if len(cur)>1 else 1.0,
        "AST": float(np.var(pd.to_numeric(cur["AST"], errors="coerce").dropna(), ddof=1)) if len(cur)>1 else 1.0,
        "FG3M": float(np.var(pd.to_numeric(cur["FG3M"], errors="coerce").dropna(), ddof=1)) if len(cur)>1 else 1.0,
    }

    # Last season mean
    if df_last_season is not None and not df_last_season.empty:
        ls_pts = safe_mean(df_last_season["PTS"]); ls_reb = safe_mean(df_last_season["REB"])
        ls_ast = safe_mean(df_last_season["AST"]); ls_3pm = safe_mean(df_last_season["FG3M"])
        n_ls = len(df_last_season)
        ls_var = {
            "PTS": float(np.var(pd.to_numeric(df_last_season["PTS"], errors="coerce").dropna(), ddof=1)) if n_ls>1 else 1.0,
            "REB": float(np.var(pd.to_numeric(df_last_season["REB"], errors="coerce").dropna(), ddof=1)) if n_ls>1 else 1.0,
            "AST": float(np.var(pd.to_numeric(df_last_season["AST"], errors="coerce").dropna(), ddof=1)) if n_ls>1 else 1.0,
            "FG3M": float(np.var(pd.to_numeric(df_last_season["FG3M"], errors="coerce").dropna(), ddof=1)) if n_ls>1 else 1.0,
        }
    else:
        ls_pts=ls_reb=ls_ast=ls_3pm=0.0
        n_ls=0
        ls_var={"PTS":1,"REB":1,"AST":1,"FG3M":1}

    # H2H (very small weight unless >= 4)
    if df_h2h is not None and not df_h2h.empty:
        h_pts = safe_mean(df_h2h["PTS"]); h_reb = safe_mean(df_h2h["REB"])
        h_ast = safe_mean(df_h2h["AST"]); h_3pm = safe_mean(df_h2h["FG3M"])
        n_h = len(df_h2h)
        h_var = {
            "PTS": float(np.var(pd.to_numeric(df_h2h["PTS"], errors="coerce").dropna(), ddof=1)) if n_h>1 else 1.0,
            "REB": float(np.var(pd.to_numeric(df_h2h["REB"], errors="coerce").dropna(), ddof=1)) if n_h>1 else 1.0,
            "AST": float(np.var(pd.to_numeric(df_h2h["AST"], errors="coerce").dropna(), ddof=1)) if n_h>1 else 1.0,
            "FG3M": float(np.var(pd.to_numeric(df_h2h["FG3M"], errors="coerce").dropna(), ddof=1)) if n_h>1 else 1.0,
        }
    else:
        h_pts=h_reb=h_ast=h_3pm=0.0
        n_h=0
        h_var={"PTS":1,"REB":1,"AST":1,"FG3M":1}

    # League baselines & opponent adj
    league_means = league_baselines(season_opp_df)

    opp_row = None
    if opp_team_id is not None and season_opp_df is not None and not season_opp_df.empty:
        row = season_opp_df[season_opp_df["TEAM_ID"] == opp_team_id]
        opp_row = row.iloc[0] if not row.empty else None
    opp_adj = opponent_adjustment(opp_row)

    out = {}
    for k, col in [("PTS","PTS"),("REB","REB"),("AST","AST"),("FG3M","FG3M")]:
        # Gather candidates
        cands = []
        # Recency first: L5 highest base weight, then L10, L20 with slight decay
        for w, base_w in [(5, 1.0), (10, 0.7), (20, 0.5)]:
            mean_k = wins[w][k]
            n = wins[w]["N"]
            var_k = wins[w][f"VAR_{'3PM' if k=='FG3M' else k}"] if f"VAR_{'3PM' if k=='FG3M' else k}" in wins[w] else 1.0
            wgt = base_w * var_aware_weight(n, var_k)
            if n > 0:
                cands.append((mean_k, wgt))

        # Current season
        var_kc = cur_var[k if k!="FG3M" else "FG3M"]
        cands.append(((cur_pts if k=="PTS" else cur_reb if k=="REB" else cur_ast if k=="AST" else cur_3pm),
                      0.6 * var_aware_weight(len(cur), var_kc)))

        # Last season (smaller weight)
        if n_ls > 0:
            var_kls = ls_var[k if k!="FG3M" else "FG3M"]
            cands.append(((ls_pts if k=="PTS" else ls_reb if k=="REB" else ls_ast if k=="AST" else ls_3pm),
                          0.35 * var_aware_weight(n_ls, var_kls)))

        # H2H tiny unless >=4
        if n_h >= 1:
            base = 0.15 if n_h < 4 else 0.35
            var_kh = h_var[k if k!="FG3M" else "FG3M"]
            cands.append(((h_pts if k=="PTS" else h_reb if k=="REB" else h_ast if k=="AST" else h_3pm),
                          base * var_aware_weight(n_h, var_kh)))

        # Weighted blend
        if not cands:
            blended = 0.0
        else:
            num = sum(m * w for (m, w) in cands)
            den = sum(w for (_, w) in cands)
            blended = num / den if den > 0 else 0.0

        # James-Stein shrinkage to league mean (stabilizes extremes)
        # Use approximate variance from current season (fallback 1.0)
        var_for_shrink = max(var_kc, 1e-6)
        n_eff = min(int(sum(w for (_, w) in cands)), 82)  # cap
        shrunk = james_stein_shrink(blended, league_means[k if k!="FG3M" else "FG3M"], n_eff, var_for_shrink)

        # Opponent adjustment
        adj = shrunk * opp_adj[k if k!="FG3M" else "FG3M"]

        # Final sensible rounding
        out[k if k!="FG3M" else "3PM"] = float(np.round(adj, 2))

    return out

def result_pill(val: float, good: Tuple[float,float]) -> str:
    lo, hi = good
    if val >= hi:
        cls = "ok"
    elif val >= lo:
        cls = "warn"
    else:
        cls = "bad"
    return f'<span class="pill {cls}">{val:.2f}</span>'

# --------------------------- Sidebar ---------------------------

st.sidebar.title("🏀 NBA Live Dashboard")
st.sidebar.caption("Pick a player, get live stats + next-game projections.")

players_df = get_players_df()
teams_df = get_teams_df()

# Player search
name_query = st.sidebar.text_input("Search player", placeholder="e.g., LeBron James")
if name_query:
    cand = players_df[players_df["full_name"].str.contains(name_query, case=False, na=False)]
else:
    cand = players_df

player = st.sidebar.selectbox(
    "Select player",
    sorted(cand["full_name"].tolist()),
    index=0 if not cand.empty else None
)

# --------------------------- Main ---------------------------

if not player:
    st.info("Search and select a player in the sidebar to begin.")
    st.stop()

player_row = players_df[players_df["full_name"] == player].iloc[0]
player_id = int(player_row["id"])

colA, colB = st.columns([1, 2], vertical_alignment="center")
with colA:
    try:
        info = get_player_info(player_id)
        team_name = info.get("TEAM_NAME", "—")
        jersey = info.get("JERSEY", "—")
        pos = info.get("POSITION", "—")
    except Exception:
        team_name = "—"; jersey = "—"; pos = "—"

    st.subheader(player)
    st.caption(f"{team_name} • #{jersey} • {pos}")

with colB:
    st.markdown("### Next-Game Projection (progressive, regression-proof)")

# Current season string
today = dt.date.today()
season_str = to_season_string(today)

# Load logs
with st.spinner("Fetching recent logs..."):
    try:
        logs_cur = get_player_game_logs(player_id, season_str)
    except Exception as e:
        st.error(f"Could not fetch game logs for {player}: {e}")
        st.stop()

with st.spinner("Fetching last season & league context..."):
    try:
        logs_last = get_player_last_season_logs(player_id, season_str)
    except Exception:
        logs_last = pd.DataFrame()

    try:
        opp_df = get_league_team_defense(season_str)
    except Exception:
        opp_df = pd.DataFrame()

# Opponent for next game
with st.spinner("Finding next opponent..."):
    try:
        pinfo = get_player_info(player_id)
        opp_team_id = find_next_opponent_team_id(pinfo, season_str)
        if opp_team_id:
            opp_name = teams_df.loc[teams_df["id"]==opp_team_id, "full_name"]
            opp_name = opp_name.iloc[0] if not opp_name.empty else "Upcoming Opponent"
        else:
            opp_name = "Upcoming Opponent"
    except Exception:
        opp_team_id = None
        opp_name = "Upcoming Opponent"

# H2H slice from current + last season combined
combined = pd.concat([logs_cur, logs_last], ignore_index=True) if not logs_last.empty else logs_cur.copy()
df_h2h = h2h_subset(combined, opp_team_id)

# Projection
projs = ensemble_projection(logs_cur, logs_last, df_h2h, opp_df, opp_team_id)

# Display Projections
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f"""<div class="metric-card"><div class="small">Points</div>
<h2 style="margin:.25rem 0">{projs.get('PTS',0):.2f}</h2>
<div class="small">{opp_name}</div></div>""", unsafe_allow_html=True)
m2.markdown(f"""<div class="metric-card"><div class="small">Rebounds</div>
<h2 style="margin:.25rem 0">{projs.get('REB',0):.2f}</h2>
<div class="small">{opp_name}</div></div>""", unsafe_allow_html=True)
m3.markdown(f"""<div class="metric-card"><div class="small">Assists</div>
<h2 style="margin:.25rem 0">{projs.get('AST',0):.2f}</h2>
<div class="small">{opp_name}</div></div>""", unsafe_allow_html=True)
m4.markdown(f"""<div class="metric-card"><div class="small">3PM</div>
<h2 style="margin:.25rem 0">{projs.get('3PM',0):.2f}</h2>
<div class="small">{opp_name}</div></div>""", unsafe_allow_html=True)

st.divider()

# --------------------------- Recent Form Sections ---------------------------

st.markdown("## Recent Form & Splits")

def show_form_section(title: str, df: pd.DataFrame):
    if df is None or df.empty:
        with st.expander(title, expanded=False):
            st.write("No data.")
        return
    with st.expander(title, expanded=False):
        # Table
        show_cols = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","FG3M","FGA","FGM","PLUS_MINUS"]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

        # Chart (PTS/REB/AST/3PM)
        melt_cols = [c for c in ["PTS","REB","AST","FG3M"] if c in df.columns]
        if melt_cols:
            tidy = df[["GAME_DATE"] + melt_cols].copy()
            tidy["GAME_DATE"] = pd.to_datetime(tidy["GAME_DATE"])
            tidy = tidy.sort_values("GAME_DATE")
            tidy = tidy.melt("GAME_DATE", var_name="Stat", value_name="Value")
            ch = (
                alt.Chart(tidy)
                .mark_line(point=True)
                .encode(
                    x=alt.X("GAME_DATE:T", title="Game Date"),
                    y=alt.Y("Value:Q"),
                    tooltip=["GAME_DATE:T","Stat","Value:Q"]
                )
                .properties(height=220)
            )
            st.altair_chart(ch, use_container_width=True)

# L5 / L10 / L20 (from current season logs sorted desc)
for n in [5, 10, 20]:
    sub = logs_cur.head(n).copy()
    show_form_section(f"Last {n} (current season)", sub)

# Current season block
show_form_section(f"Current Season ({season_str})", logs_cur)

# Last season block
if not logs_last.empty:
    last_season_str = f"{int(season_str.split('-')[0])-1}-{int(season_str.split('-')[0])%100:02d}"
    show_form_section(f"Last Season ({last_season_str})", logs_last)

# Head-to-head block
if df_h2h is not None and not df_h2h.empty:
    opp_team_name = teams_df.loc[teams_df["id"]==opp_team_id, "full_name"]
    opp_team_name = opp_team_name.iloc[0] if not opp_team_name.empty else "Opponent"
    show_form_section(f"Head-to-Head vs {opp_team_name}", df_h2h)

# --------------------------- Notes ---------------------------
with st.expander("Model Notes", expanded=False):
    st.markdown(
        """
        **Projection recipe (designed to be “regression-proof”):**
        - Recency windows **L5 / L10 / L20**, each **variance-aware weighted** (more weight for stable, bigger samples).
        - Adds **current season** + **last season** means (smaller weight for last season).
        - **Head-to-head** gets a tiny weight unless 4+ games of sample.
        - Performs **James–Stein shrinkage** toward league baselines (reduces extreme over/under-shoot for small samples).
        - Applies **opponent difficulty adjustment** via **opponent-allowed** PTS/REB/AST/3PM (clamped ±15%).
        - Final output rounded to 2 decimals for **PTS, REB, AST, 3PM**.
        """
    )
    st.caption("Tip: Use the Recent Form sections to sanity-check momentum vs projection.")

st.caption("Built with ❤️ for Hot Shot Props · Streamlit + nba_api")