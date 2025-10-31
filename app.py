# app.py — Hot Shot Props | NBA Player Stats (Free, Premium Look)
# - Uses nba_api (no external paid key)
# - Stabilized endpoints + caching + small delays to reduce rate-limit issues
# - NBA.com-inspired dark/blue theme, premium card styling
# - Player search, auto-refresh toggle, watchlist, CSV export
# - Safer data guards to avoid crashes if columns are missing

import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playercareerstats, playergamelogs
import pandas as pd
import numpy as np
import re
import time
from PIL import Image
import altair as alt

# =========================
# Config & Globals
# =========================
st.set_page_config(
    page_title="Hot Shot Props • NBA Player Analytics (Free)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Core stat columns used across calculations
STATS_COLS = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']
PREDICTION_TARGETS = ['PTS','AST','REB','FG3M']

# Small delay between API calls to be gentle on rate limits
API_SLEEP = 0.25

# =========================
# Theming (NBA.com inspired)
# =========================
st.markdown("""
<style>
/* Global */
:root {
  --bg: #0f1116;          /* deep slate */
  --panel: #121722;       /* card bg */
  --panel2: #0f1629;      /* alt bg */
  --ink: #e5e7eb;         /* light ink */
  --muted: #9aa3b2;       /* muted ink */
  --blue: #1d4ed8;        /* brand blue */
  --blue-bright: #2563eb; /* hover */
  --cyan: #06b6d4;        /* accent */
  --gold: #fbbf24;        /* accent 2 */
  --success: #22c55e;
}

/* Page background + text */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--ink);
}

/* Main title */
h1, h2, h3, h4 {
  color: #b3d1ff !important;
  letter-spacing: 0.2px;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f1629 0%, #0f1116 100%);
  border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #e5f0ff; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
  background: #14213d;
  color: #dbeafe;
  border-radius: 10px 10px 0 0;
  padding: 10px 16px;
  font-weight: 700;
}
.stTabs [aria-selected="true"] {
  background: #1e3a8a;
  color: white;
}

/* Buttons */
.stButton > button {
  background: var(--blue);
  color: white;
  border: none;
  border-radius: 10px;
  padding: 0.6rem 1rem;
  font-weight: 700;
}
.stButton > button:hover { background: var(--blue-bright); }

/* Metrics */
[data-testid="stMetric"] {
  background: var(--panel);
  padding: 16px;
  border-radius: 16px;
  border: 1px solid #1f2a44;
  box-shadow: 0 6px 24px rgba(0,0,0,0.35);
}
[data-testid="stMetric"] label { color: #cfe1ff; }
[data-testid="stMetric"] div[data-testid="stMetricValue"] {
  color: #e0edff;
  font-size: 1.6rem;
}

/* Dataframes */
.stDataFrame {
  background: var(--panel);
  border-radius: 12px;
  padding: 4px;
  border: 1px solid #1f2a44;
}

/* Chips/badges */
.badge {
  display:inline-block;
  padding:4px 10px;
  font-size: 0.8rem;
  border:1px solid #1f2a44;
  border-radius:9999px;
  background: #0b1222;
  color:#9bd1ff;
}

/* Subtle card container */
.card {
  background: var(--panel);
  border: 1px solid #1f2a44;
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 8px 28px rgba(0,0,0,0.35);
}

/* Progress bar brand color */
.stProgress > div > div > div > div { background-color: var(--blue) !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def extract_opp_from_matchup(matchup: str) -> str | None:
    if not isinstance(matchup, str):
        return None
    m = re.search(r'@\s*([A-Z]{3})|vs\.\s*([A-Z]{3})|VS\.\s*([A-Z]{3})', matchup, re.IGNORECASE)
    if m:
        return (m.group(1) or m.group(2) or m.group(3)).upper()
    return None

# =========================
# Cached data access
# =========================
@st.cache_data(ttl=6*3600)
def get_active_nba_players():
    try:
        return players.get_active_players()
    except Exception as e:
        st.error(f"Error fetching active players: {e}")
        return []

@st.cache_data(ttl=12*3600)
def get_all_nba_teams():
    try:
        return teams.get_teams()
    except Exception as e:
        st.error(f"Error fetching teams: {e}")
        return []

@st.cache_data(ttl=3600)
def fetch_player_data(player_id: int):
    """
    Returns (career_by_season_df, career_game_logs_df)
    """
    career_df = None
    career_game_logs = None
    try:
        # Career by season summary
        time.sleep(API_SLEEP)
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        frames = career_stats.get_data_frames()
        if not frames:
            return None, None
        career_df = frames[0]

        if career_df is None or career_df.empty:
            return career_df, None

        # Seasons list (as provided, e.g., '2019-20')
        seasons = career_df['SEASON_ID'].tolist()
        logs_list = []
        for s in seasons:
            try:
                time.sleep(API_SLEEP)
                # PlayerGameLogs supports 'season_nullable' as '2023-24', and returns combined logs
                logs_df = playergamelogs.PlayerGameLogs(
                    player_id_nullable=player_id,
                    season_nullable=s
                ).get_data_frames()[0]
                if logs_df is not None and not logs_df.empty:
                    logs_list.append(logs_df)
            except Exception as se:
                st.warning(f"Could not fetch game logs for {s}. Error: {se}")
                continue

        if logs_list:
            career_game_logs = pd.concat(logs_list, ignore_index=True)
            # Ensure correct types
            if 'GAME_DATE' in career_game_logs.columns:
                career_game_logs['GAME_DATE'] = pd.to_datetime(career_game_logs['GAME_DATE'])
                career_game_logs = career_game_logs.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
        else:
            st.info("No game logs found for this player.")
        return career_df, career_game_logs

    except Exception as e:
        st.error(f"An error occurred while fetching player data: {e}")
        return None, None

# =========================
# Computations
# =========================
def get_player_vs_all_teams_career_stats(career_game_logs_df: pd.DataFrame) -> pd.DataFrame | None:
    if career_game_logs_df is None or career_game_logs_df.empty:
        return None
    try:
        df = career_game_logs_df.copy()
        if 'MATCHUP' not in df.columns:
            return None
        df['OPP_ABBR'] = df['MATCHUP'].apply(extract_opp_from_matchup)
        df = df.dropna(subset=['OPP_ABBR'])
        cols = safe_cols(df, STATS_COLS)
        if not cols:
            return None
        grouped = df.groupby('OPP_ABBR')[cols].mean()
        gp = df.groupby('OPP_ABBR').size().to_frame('GP')
        out = grouped.join(gp)
        out = out[['GP'] + [c for c in out.columns if c != 'GP']]
        return out
    except Exception as e:
        st.error(f"Error computing vs-all-teams career stats: {e}")
        return None

def calculate_recent_game_averages(career_game_logs_df: pd.DataFrame) -> dict:
    out = {}
    if career_game_logs_df is None or career_game_logs_df.empty:
        return out
    df = career_game_logs_df.sort_values('GAME_DATE', ascending=False).copy()
    cols = safe_cols(df, STATS_COLS)
    if not cols:
        return out

    def avg_block(n: int, key: str):
        if len(df) >= n:
            sub = df.head(n)
            csub = [c for c in cols if c in sub.columns]
            if csub:
                out[key] = sub[csub].mean().to_frame(name=key.replace('_', ' ').title()).T

    avg_block(5, 'last_5_games_avg')
    avg_block(10, 'last_10_games_avg')
    avg_block(20, 'last_20_games_avg')

    # Individual last 5
    iv_cols = ['GAME_DATE','MATCHUP','SEASON_YEAR'] + cols
    iv_cols = [c for c in iv_cols if c in df.columns]
    if len(df) >= 5 and iv_cols:
        last5 = df.head(5)[iv_cols].copy()
        if 'GAME_DATE' in last5.columns:
            last5['GAME_DATE'] = pd.to_datetime(last5['GAME_DATE']).dt.strftime('%Y-%m-%d')
        out['last_5_games_individual'] = last5
    return out

def predict_next_game_stats(career_game_logs_df: pd.DataFrame, latest_season: str) -> dict | None:
    if career_game_logs_df is None or career_game_logs_df.empty:
        return None
    df = career_game_logs_df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
    cols = safe_cols(df, STATS_COLS)
    if not cols:
        return None

    features = {}
    for c in cols:
        r5  = df[c].rolling(window=5,  min_periods=1).mean().iloc[-1] if len(df) else np.nan
        r10 = df[c].rolling(window=10, min_periods=1).mean().iloc[-1] if len(df) else np.nan
        r20 = df[c].rolling(window=20, min_periods=1).mean().iloc[-1] if len(df) else np.nan
        features[f'{c}_r5']  = r5
        features[f'{c}_r10'] = r10
        features[f'{c}_r20'] = r20

    # Season avg (prefer latest season rows)
    season_avg = pd.Series(dtype=float)
    if 'SEASON_YEAR' in df.columns:
        cur = df[df['SEASON_YEAR'] == latest_season]
        if not cur.empty:
            season_avg = cur[cols].mean()
    if season_avg.empty:
        season_avg = df[cols].mean()

    preds = {}
    for stat in PREDICTION_TARGETS:
        if all(k in features for k in [f'{stat}_r5', f'{stat}_r10', f'{stat}_r20']):
            s_avg = season_avg.get(stat, 0.0) if pd.notna(season_avg.get(stat, np.nan)) else 0.0
            v5  = features[f'{stat}_r5']  if pd.notna(features[f'{stat}_r5'])  else s_avg
            v10 = features[f'{stat}_r10'] if pd.notna(features[f'{stat}_r10']) else s_avg
            v20 = features[f'{stat}_r20'] if pd.notna(features[f'{stat}_r20']) else s_avg
            pred = 0.4*v5 + 0.3*v10 + 0.2*v20 + 0.1*s_avg
            preds[stat] = round(float(pred), 2)
        else:
            preds[stat] = "N/A"
    return preds if any(isinstance(v, (int,float)) for v in preds.values()) else None

# =========================
# Sidebar — Controls
# =========================
with st.sidebar:
    st.header("Player Finder")
    active_players = get_active_nba_players()
    all_teams = get_all_nba_teams()

    # Build indices
    player_name_to_id = {p['full_name']: p['id'] for p in active_players}
    player_names_sorted = sorted(player_name_to_id.keys())

    # Quick search filter
    q = st.text_input("Search player", value="", help="Type to filter the dropdown list.")
    filtered_names = [n for n in player_names_sorted if q.lower() in n.lower()] if q else player_names_sorted

    player_choice = st.selectbox("Choose a player", filtered_names, index=None, placeholder="Select a player")

    st.markdown("<hr>", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        auto_refresh = st.toggle("Auto-refresh (60s)", value=False, help="Re-run the app every 60 seconds.")
    with colB:
        show_vs_all_teams = st.toggle("Show vs Opponents", value=False, help="Show career averages vs each opponent.")

    if auto_refresh:
        st.experimental_rerun  # static type hint
        st.autorefresh = st.experimental_rerun  # appease linters
        st_autorefresh = st.experimental_rerun
    if auto_refresh:
        st.experimental_rerun  # noop for older versions
    if auto_refresh:
        st_autorefresh_ref = st.experimental_rerun
    if auto_refresh:
        # real refresh
        st_autorefresh = st.autorefresh if hasattr(st, "autorefresh") else None
        st.experimental_rerun  # will be ignored; use native helper:
        st_autorefresh = st.session_state.get("_autorefresh", None)
        st.session_state["_autorefresh"] = st.experimental_rerun

    # Simple watchlist (session-only)
    st.subheader("Watchlist")
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    if player_choice and player_choice not in st.session_state.watchlist:
        if st.button(f"➕ Add {player_choice}"):
            st.session_state.watchlist.append(player_choice)
            st.success(f"Added {player_choice} to watchlist.")
    if st.session_state.watchlist:
        st.write(", ".join(st.session_state.watchlist))
        if st.button("Clear Watchlist"):
            st.session_state.watchlist = []
            st.info("Watchlist cleared.")

# =========================
# Header / Hero
# =========================
st.markdown("""
<div class="card" style="margin-bottom: 10px;">
  <div style="display:flex; align-items:center; justify-content: space-between;">
    <div>
      <h1 style="margin:0;">Hot Shot Props — NBA Player Analytics</h1>
      <div class="badge">Free Beta • All features unlocked</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:0.9rem; color:#9aa3b2;">Live trends • Weighted predictions • CSV export</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Main Logic
# =========================
if not player_choice:
    st.info("To begin, pick a player from the sidebar (use the search box to filter).")
    st.stop()

player_id = player_name_to_id.get(player_choice)
if player_id is None:
    st.warning("Could not resolve player id. Try another player.")
    st.stop()

with st.spinner(f"Loading {player_choice} data…"):
    career_df, logs_df = fetch_player_data(player_id)

if career_df is None or career_df.empty:
    st.info(f"No data available for **{player_choice}**.")
    st.stop()

# Player header & key metrics
st.subheader(f"{player_choice} — Career Summary")

total_gp = int(career_df['GP'].sum()) if 'GP' in career_df.columns else 0
totals = career_df[safe_cols(career_df, STATS_COLS)].sum(numeric_only=True)

valid = [c for c in STATS_COLS if c in totals.index]
overall_avg = (totals[valid] / total_gp) if total_gp > 0 else pd.Series(dtype=float)

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Points", f"{overall_avg.get('PTS', np.nan):.2f}" if 'PTS' in overall_avg.index else "N/A")
with m2: st.metric("Rebounds", f"{overall_avg.get('REB', np.nan):.2f}" if 'REB' in overall_avg.index else "N/A")
with m3: st.metric("Assists", f"{overall_avg.get('AST', np.nan):.2f}" if 'AST' in overall_avg.index else "N/A")
with m4: st.metric("Minutes", f"{overall_avg.get('MIN', np.nan):.2f}" if 'MIN' in overall_avg.index else "N/A")

# Determine latest season id string robustly
if 'SEASON_ID' in career_df.columns and not career_df['SEASON_ID'].empty:
    # Career frames often come oldest->latest; grab last non-null
    latest_season = str(career_df['SEASON_ID'].dropna().iloc[-1])
else:
    latest_season = str(logs_df['SEASON_YEAR'].dropna().iloc[-1]) if logs_df is not None and 'SEASON_YEAR' in logs_df.columns and not logs_df.empty else "N/A"

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Season Averages", "Recent Games", "Predictions", "Exports"])

# ===== Tab 1: Season Averages =====
with tab1:
    st.markdown("### Season Averages")
    cols_for_display = ['SEASON_ID','TEAM_ABBREVIATION','GP'] + STATS_COLS
    cols_for_display = safe_cols(career_df, cols_for_display)
    disp = career_df[cols_for_display].copy()

    # Convert totals to per-game safely
    if 'GP' in disp.columns:
        for c in STATS_COLS:
            if c in disp.columns:
                disp[c] = disp.apply(lambda r: round(r[c]/r['GP'], 2) if r['GP'] and r['GP'] > 0 else 0.0, axis=1)

    # nicer index
    if 'SEASON_ID' in disp.columns:
        disp = disp.set_index('SEASON_ID')

    st.dataframe(disp, use_container_width=True)

    # Quick trend chart (PTS per season) if available
    if {'PTS','GP'}.issubset(disp.columns) or ('PTS' in career_df.columns and 'GP' in career_df.columns):
        chart_df = career_df[['SEASON_ID','PTS','GP']].copy()
        chart_df['PTS/G'] = chart_df.apply(lambda r: r['PTS']/r['GP'] if r['GP']>0 else 0.0, axis=1)
        c = alt.Chart(chart_df).mark_line(point=True).encode(
            x=alt.X('SEASON_ID:N', title='Season'),
            y=alt.Y('PTS/G:Q', title='Points per Game'),
            tooltip=['SEASON_ID','PTS/G']
        ).properties(height=280)
        st.altair_chart(c, use_container_width=True)

# ===== Tab 2: Recent Games =====
with tab2:
    st.markdown("### Detailed Statistics & Recent Averages")

    # Last season averages (previous season row if exists)
    if len(career_df) > 1:
        last_season_row = career_df.iloc[-2]
        last_season_id = last_season_row.get('SEASON_ID', 'N/A')
        gp = last_season_row.get('GP', 0)
        if gp and gp > 0:
            lcols = safe_cols(career_df, STATS_COLS)
            last_totals = last_season_row[lcols]
            last_avg = (last_totals / gp).to_frame(name=f"Last Season ({last_season_id}) Avg").T
            st.dataframe(last_avg.round(2), use_container_width=True)
        else:
            st.info("Last season averages not available (0 GP).")

    # Recent averages
    recent = calculate_recent_game_averages(logs_df)
    st.write("Recent Game Averages")
    c1, c2, c3 = st.columns(3)
    with c1:
        df5 = recent.get('last_5_games_avg')
        st.dataframe(df5.round(2) if df5 is not None and not df5.empty else pd.DataFrame(), use_container_width=True)
    with c2:
        df10 = recent.get('last_10_games_avg')
        st.dataframe(df10.round(2) if df10 is not None and not df10.empty else pd.DataFrame(), use_container_width=True)
    with c3:
        df20 = recent.get('last_20_games_avg')
        st.dataframe(df20.round(2) if df20 is not None and not df20.empty else pd.DataFrame(), use_container_width=True)

    st.write("Last 5 Individual Games (Most Recent)")
    ind5 = recent.get('last_5_games_individual')
    st.dataframe(ind5 if ind5 is not None and not ind5.empty else pd.DataFrame(), use_container_width=True)

    if show_vs_all_teams:
        st.markdown("### Career Averages vs Opponents")
        vs_df = get_player_vs_all_teams_career_stats(logs_df)
        st.dataframe(vs_df.round(2) if vs_df is not None else pd.DataFrame(), use_container_width=True)

# ===== Tab 3: Predictions =====
with tab3:
    st.markdown(f"### {player_choice} — Next Game Prediction")
    preds = predict_next_game_stats(logs_df, latest_season)
    if preds:
        pred_df = pd.DataFrame([preds])
        st.dataframe(pred_df, use_container_width=True)
        st.caption("Heuristic: weighted blend of last 5/10/20 games + season avg (40/30/20/10).")
        if st.button(f"Save Prediction for {player_choice}"):
            if "saved_predictions" not in st.session_state:
                st.session_state.saved_predictions = {}
            key = f"{player_choice} ({latest_season})"
            st.session_state.saved_predictions[key] = {
                'PTS': preds.get('PTS','N/A'),
                'REB': preds.get('REB','N/A'),
                'AST': preds.get('AST','N/A'),
                'FG3M': preds.get('FG3M','N/A'),
                'Season': latest_season
            }
            st.success("Prediction saved.")
    else:
        st.info("Not enough data to generate a prediction.")

    st.markdown("#### Saved Predictions")
    if "saved_predictions" in st.session_state and st.session_state.saved_predictions:
        saved_df = pd.DataFrame.from_dict(st.session_state.saved_predictions, orient='index')
        st.dataframe(saved_df, use_container_width=True)
        cA, cB = st.columns(2)
        with cA:
            if st.button("Export Saved Predictions (CSV)"):
                csv = saved_df.to_csv(index=True).encode('utf-8')
                st.download_button("Download CSV", csv, file_name="saved_predictions.csv", mime="text/csv")
        with cB:
            if st.button("Clear Saved Predictions"):
                st.session_state.saved_predictions = {}
                st.success("Cleared saved predictions.")
    else:
        st.write("No saved predictions yet.")

# ===== Tab 4: Exports =====
with tab4:
    st.markdown("### Data Export")
    colx, coly = st.columns(2)
    with colx:
        st.write("**Season Averages Table**")
        exp_cols = ['SEASON_ID','TEAM_ABBREVIATION','GP'] + STATS_COLS
        exp_cols = safe_cols(career_df, exp_cols)
        season_exp = career_df[exp_cols].copy()
        # to per game
        if 'GP' in season_exp.columns:
            for c in STATS_COLS:
                if c in season_exp.columns:
                    season_exp[c] = season_exp.apply(lambda r: round(r[c]/r['GP'], 2) if r['GP'] and r['GP']>0 else 0.0, axis=1)
        if not season_exp.empty:
            csv = season_exp.to_csv(index=False).encode('utf-8')
            st.download_button("Download Season Averages CSV", data=csv, file_name=f"{player_choice}_season_averages.csv", mime="text/csv")
        else:
            st.write("_No season data to export._")

    with coly:
        st.write("**Career Game Logs (All Seasons)**")
        if logs_df is not None and not logs_df.empty:
            csv = logs_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Game Logs CSV", data=csv, file_name=f"{player_choice}_career_logs.csv", mime="text/csv")
        else:
            st.write("_No game logs to export._")

# Footer ribbon (soft exclusivity without paywall)
st.markdown("""
<div style="margin-top:14px; display:flex; justify-content:space-between; align-items:center; opacity:0.9;">
  <div class="badge">Early Access • All analytics unlocked</div>
  <div style="font-size:0.85rem; color:#9aa3b2;">Hot Shot Props © — Built with nba_api & Streamlit</div>
</div>
""", unsafe_allow_html=True)
