# app.py — Hot Shot Props | NBA Player Analytics (with ML Loader)
# - One-page UX, NBA.com-style theme
# - Single combined search: "TEAM_ABBR — Player Name"
# - Headshots, trend charts, 4 metric rows
# - Favorites + Twilio SMS (manual + daily while app is open)
# - Next Game shows 🏠/✈️ Home/Away icon
# - Robust "Last Game Stats"
# - ML INTEGRATED: loads models/model_{PTS,REB,AST,FG3M}.pkl if present; else fallback

import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playercareerstats,
    playergamelogs,
    scoreboardv2,
    commonteamroster,
)
import pandas as pd
import numpy as np
import re, time, datetime as dt, threading, json, os
import altair as alt
import requests
from io import BytesIO

# Optional Twilio (only used if credentials are provided)
try:
    from twilio.rest import Client as TwilioClient
except Exception:
    TwilioClient = None

# ML model loading
try:
    import joblib
except Exception:
    joblib = None

# ---------------------------------
# Page config + constants
# ---------------------------------
st.set_page_config(
    page_title="Hot Shot Props • NBA Player Analytics (Free)",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATS_COLS   = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']
PREDICT_COLS = ['PTS','AST','REB','FG3M']
API_SLEEP    = 0.25
FAV_FILE     = "favorites.json"
MODEL_DIR    = "models"
MODEL_FILES  = {t: os.path.join(MODEL_DIR, f"model_{t}.pkl") for t in PREDICT_COLS}
FEAT_TEMPLATE = lambda tgt: [f'{tgt}_r5', f'{tgt}_r10', f'{tgt}_r20', f'{tgt}_season_mean', 'IS_HOME', 'DAYS_REST']

# ---------------------------------
# Styling
# ---------------------------------
st.markdown("""
<style>
:root { --bg:#0f1116; --panel:#121722; --ink:#e5e7eb; --muted:#9aa3b2; --blue:#1d4ed8; --line:#1f2a44; }
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--ink);}
h1,h2,h3,h4{color:#b3d1ff!important;letter-spacing:.2px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f1629 0%,#0f1116 100%);border-right:1px solid #1f2937;}
.stButton>button{background:var(--blue);color:white;border:none;border-radius:10px;padding:.6rem 1rem;font-weight:700;}
.stButton>button:hover{background:#2563eb;}
[data-testid="stMetric"]{background:var(--panel);padding:16px;border-radius:16px;border:1px solid var(--line);box-shadow:0 8px 28px rgba(0,0,0,.35);}
[data-testid="stMetric"] label{color:#cfe1ff;}
[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#e0edff;font-size:1.5rem;}
.stDataFrame{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:4px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 8px 28px rgba(0,0,0,.35);}
.badge{display:inline-block;padding:4px 10px;font-size:.8rem;border:1px solid var(--line);border-radius:9999px;background:#0b1222;color:#9bd1ff;}
.stProgress>div>div>div>div{background-color:var(--blue)!important;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Helpers
# ---------------------------------
def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def extract_opp_from_matchup(matchup: str) -> str | None:
    if not isinstance(matchup, str):
        return None
    m = re.search(r'@\s*([A-Z]{3})|vs\.\s*([A-Z]{3})|VS\.\s*([A-Z]{3})', matchup, re.IGNORECASE)
    if m:
        return (m.group(1) or m.group(2) or m.group(3)).upper()
    return None

def cdn_headshot(player_id: int, size: str = "1040x760") -> BytesIO | None:
    """NBA media day headshot from official CDN."""
    url = f"https://cdn.nba.com/headshots/nba/latest/{size}/{player_id}.png"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return BytesIO(r.content)
    except Exception:
        pass
    return None

# --- Favorites persistence ---
def load_favorites() -> list[str]:
    try:
        if os.path.exists(FAV_FILE):
            with open(FAV_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []

def save_favorites(favs: list[str]) -> None:
    try:
        with open(FAV_FILE, "w") as f:
            json.dump(favs, f)
    except Exception:
        pass

# ---------------------------------
# Cached data
# ---------------------------------
@st.cache_data(ttl=6*3600)
def get_active_players():
    try:
        return players.get_active_players()
    except Exception as e:
        st.error(f"Error fetching active players: {e}")
        return []

@st.cache_data(ttl=12*3600)
def get_all_teams():
    try:
        return teams.get_teams()
    except Exception as e:
        st.error(f"Error fetching teams: {e}")
        return []

@st.cache_data(ttl=12*3600)
def build_team_player_index() -> dict[str, int]:
    """
    Returns mapping: 'TEAM_ABBR — Player Name' -> PLAYER_ID
    Built from each team's roster (cached), plus extra active players marked 'FA'.
    """
    mapping: dict[str, int] = {}
    all_t = get_all_teams()
    for t in all_t:
        tid  = t.get("id")
        abbr = t.get("abbreviation", "TBD")
        if not tid:
            continue
        try:
            time.sleep(API_SLEEP)
            roster = commonteamroster.CommonTeamRoster(team_id=tid).get_data_frames()[0]
            if roster is None or roster.empty:
                continue
            for _, row in roster.iterrows():
                name = str(row.get("PLAYER", "")).strip()
                pid  = int(row.get("PLAYER_ID")) if pd.notna(row.get("PLAYER_ID")) else None
                if name and pid:
                    mapping[f"{abbr} — {name}"] = pid
        except Exception:
            continue
    # Add any remaining active players not captured by rosters (FA, two-way, etc.)
    act = get_active_players()
    present_names = set(label.split("—",1)[-1].strip() for label in mapping.keys())
    for p in act:
        name = p.get('full_name')
        pid  = p.get('id')
        if name and pid and name not in present_names:
            mapping[f"FA — {name}"] = pid
    return mapping

@st.cache_data(ttl=3600)
def fetch_player(player_id: int):
    """Returns (career_by_season_df, career_logs_df)."""
    try:
        time.sleep(API_SLEEP)
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]
        seasons = career_stats['SEASON_ID'].tolist() if not career_stats.empty else []
        logs_list = []
        for s in seasons:
            try:
                time.sleep(API_SLEEP)
                df = playergamelogs.PlayerGameLogs(
                    player_id_nullable=player_id, season_nullable=s
                ).get_data_frames()[0]
                if df is not None and not df.empty:
                    logs_list.append(df)
            except Exception as se:
                st.warning(f"Game logs failed for {s}: {se}")
        logs = pd.concat(logs_list, ignore_index=True) if logs_list else pd.DataFrame()
        if 'GAME_DATE' in logs.columns:
            logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
            logs = logs.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
        return career_stats, logs
    except Exception as e:
        st.error(f"Failed to fetch player data: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=600)
def next_game_for_team(team_id: int, lookahead_days: int = 10):
    """Scan next days via ScoreboardV2 to find the team's next game."""
    if team_id is None:
        return None
    today = dt.date.today()
    team_map = {t['id']: t for t in get_all_teams()}
    for d in range(lookahead_days):
        day = today + dt.timedelta(days=d)
        try:
            time.sleep(API_SLEEP)
            sb = scoreboardv2.ScoreboardV2(game_date=day.strftime("%m/%d/%Y"))
            frames = sb.get_data_frames()
            game_header = next((f for f in frames if {'HOME_TEAM_ID','VISITOR_TEAM_ID'}.issubset(f.columns)), None)
            if game_header is None or game_header.empty:
                continue
            for _, row in game_header.iterrows():
                home_id = int(row.get('HOME_TEAM_ID', -1))
                away_id = int(row.get('VISITOR_TEAM_ID', -1))
                if team_id in (home_id, away_id):
                    opp_id  = away_id if team_id == home_id else home_id
                    opp_abr = team_map.get(opp_id, {}).get('abbreviation', 'TBD')
                    home_flag = (team_id == home_id)
                    return {'date': day.strftime("%Y-%m-%d"), 'opp_abbr': opp_abr, 'home': home_flag}
        except Exception:
            continue
    return None

# ---------------------------------
# ML: loaders + feature builder + predictor
# ---------------------------------
@st.cache_data(ttl=3600)
def load_models():
    """Load model pkl files if available. Returns dict target->model."""
    models = {}
    if joblib is None:
        return models
    for tgt, path in MODEL_FILES.items():
        try:
            if os.path.exists(path):
                models[tgt] = joblib.load(path)
        except Exception:
            continue
    return models

def _matchup_is_home(matchup: str) -> int:
    # "TEAM vs OPP" => home (1), "TEAM @ OPP" => away (0)
    if not isinstance(matchup, str):
        return 0
    return 1 if (" vs " in matchup) or (" VS " in matchup) else 0

def build_features_for_inference(logs_df: pd.DataFrame, next_game_date: dt.date | None, is_home_next: int) -> pd.DataFrame | None:
    """
    Build a single-row feature frame for *next* game, matching training:
    - rolling means r5/r10/r20 of target stat, shifted(1) so they use only past games
    - season_mean up to previous game
    - IS_HOME (for the next game)
    - DAYS_REST between last game date and next_game_date (fallback 3 if unknown)
    """
    if logs_df is None or logs_df.empty:
        return None

    # Ensure ascending by GAME_DATE
    df = logs_df.copy()
    if 'GAME_DATE' not in df.columns:
        return None
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)

    # Compute DAYS_REST for last row -> next game
    last_date = pd.to_datetime(df['GAME_DATE'].iloc[-1]).date()
    if isinstance(next_game_date, str):
        try:
            next_game_date = dt.datetime.strptime(next_game_date, "%Y-%m-%d").date()
        except Exception:
            next_game_date = None
    if next_game_date is None:
        days_rest = 3  # reasonable default
    else:
        days_rest = max(0, min(7, (next_game_date - last_date).days))

    # Rolling means and season means (shifted by 1 to avoid leakage)
    feat = {}
    for k in [5, 10, 20]:
        for c in STATS_COLS:
            series = df[c].rolling(k, min_periods=1).mean().shift(1)
            feat[f'{c}_r{k}'] = float(series.iloc[-1]) if pd.notna(series.iloc[-1]) else float(df[c].iloc[:-1].mean() if len(df) > 1 else df[c].iloc[-1])

    if 'SEASON_YEAR' in df.columns:
        # expanding().mean() per season, shifted(1)
        for c in STATS_COLS:
            smean = df.groupby('SEASON_YEAR')[c].expanding().mean().shift(1).reset_index(level=0, drop=True)
            val = smean.iloc[-1] if pd.notna(smean.iloc[-1]) else df[c].expanding().mean().shift(1).iloc[-1]
            if pd.isna(val):
                val = df[c].iloc[:-1].mean() if len(df) > 1 else df[c].iloc[-1]
            feat[f'{c}_season_mean'] = float(val)
    else:
        for c in STATS_COLS:
            smean = df[c].expanding().mean().shift(1)
            val = smean.iloc[-1] if pd.notna(smean.iloc[-1]) else df[c].iloc[:-1].mean() if len(df) > 1 else df[c].iloc[-1]
            feat[f'{c}_season_mean'] = float(val)

    # Non-stat features
    feat['IS_HOME']  = int(is_home_next)
    feat['DAYS_REST'] = int(days_rest)

    # Return as single-row DataFrame
    return pd.DataFrame([feat])

def predict_next_ml(logs_df: pd.DataFrame, next_game_date: dt.date | None, is_home_next: int, models: dict[str, object]) -> dict | None:
    """
    Use ML models if available to predict next game for each target.
    Requires the same feature names used in training.
    """
    if not models:
        return None
    feats_df = build_features_for_inference(logs_df, next_game_date, is_home_next)
    if feats_df is None or feats_df.empty:
        return None

    preds = {}
    for tgt, model in models.items():
        feat_cols = FEAT_TEMPLATE(tgt)
        # Ensure all features present
        missing = [c for c in feat_cols if c not in feats_df.columns]
        if missing:
            # cannot predict this target; skip
            continue
        try:
            val = float(model.predict(feats_df[feat_cols])[0])
            preds[tgt] = round(val, 2)
        except Exception:
            continue
    return preds if preds else None

# ---------------------------------
# Fallback (non-ML) prediction
# ---------------------------------
def recent_averages(logs: pd.DataFrame) -> dict:
    out = {}
    if logs is None or logs.empty:
        return out
    df = logs.copy()
    cols = safe_cols(df, STATS_COLS)
    if len(df) >= 5 and cols:
        sub = df.head(5)
        out['Last 5 Avg'] = sub[cols].mean().to_frame(name='Last 5 Avg').T
    return out

def predict_next_fallback(logs: pd.DataFrame, season_label: str) -> dict | None:
    """Simple weighted-averages model as a fallback."""
    if logs is None or logs.empty:
        return None
    df = logs.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
    cols = safe_cols(df, STATS_COLS)
    if not cols:
        return None

    # Season avg (prefer matching season label; fallback to overall)
    season_avg = pd.Series(dtype=float)
    if 'SEASON_YEAR' in df.columns:
        cur = df[df['SEASON_YEAR'] == season_label]
        if not cur.empty:
            season_avg = cur[cols].mean()
    if season_avg.empty:
        season_avg = df[cols].mean()

    feats = {}
    for c in cols:
        feats[f'{c}_r5']  = df[c].rolling(5,  min_periods=1).mean().iloc[-1]
        feats[f'{c}_r10'] = df[c].rolling(10, min_periods=1).mean().iloc[-1]
        feats[f'{c}_r20'] = df[c].rolling(20, min_periods=1).mean().iloc[-1]

    preds = {}
    for c in PREDICT_COLS:
        r5  = feats.get(f'{c}_r5',  np.nan)
        r10 = feats.get(f'{c}_r10', np.nan)
        r20 = feats.get(f'{c}_r20', np.nan)
        s   = season_avg.get(c, 0.0) if pd.notna(season_avg.get(c, np.nan)) else 0.0
        v5, v10, v20 = (r5 if pd.notna(r5) else s), (r10 if pd.notna(r10) else s), (r20 if pd.notna(r20) else s)
        preds[c] = round(float(0.4*v5 + 0.3*v10 + 0.2*v20 + 0.1*s), 2)
    return preds

# ---------------------------------
# SMS helpers
# ---------------------------------
def format_prediction_sms(player_name: str, team_abbr: str, next_game_info: str, preds: dict) -> str:
    parts = [
        f"Hot Shot Props — {player_name} ({team_abbr})",
        f"Next: {next_game_info}",
        f"Pred: PTS {preds.get('PTS','-')}, REB {preds.get('REB','-')}, AST {preds.get('AST','-')}, 3PM {preds.get('FG3M','-')}"
    ]
    return " | ".join(parts)

def send_sms(body: str, to_numbers: list[str], from_number: str, sid: str, token: str) -> dict:
    results = {"sent":0,"errors":[]}
    if TwilioClient is None:
        results["errors"].append("Twilio SDK not installed. Run: pip install twilio")
        return results
    try:
        client = TwilioClient(sid, token)
        for to in to_numbers:
            try:
                client.messages.create(body=body, from_=from_number, to=to)
                results["sent"] += 1
            except Exception as e:
                results["errors"].append(f"{to}: {e}")
    except Exception as e:
        results["errors"].append(str(e))
    return results

# ---------------------------------
# Sidebar — Single Search + Favorites + SMS
# ---------------------------------
with st.sidebar:
    st.header("Select Player")

    # Combined search list: "TEAM_ABBR — Player Name" -> PLAYER_ID
    label_to_pid = build_team_player_index()
    options = sorted(label_to_pid.keys())
    selection = st.selectbox(
        "Search by team or player",
        options,
        index=None,
        placeholder="e.g., LAL — LeBron James  •  BOS — Jayson Tatum  •  Nikola Jokic"
    )

    player_id = None
    player_name = None
    if selection:
        player_id = label_to_pid[selection]
        player_name = selection.split("—", 1)[-1].strip() if "—" in selection else selection.strip()

    all_t = get_all_teams()

    st.markdown("---")
    st.subheader("⭐ Favorites")
    if "favorites" not in st.session_state:
        st.session_state.favorites = load_favorites()

    if player_name and st.button(f"➕ Add {player_name}"):
        if player_name not in st.session_state.favorites:
            st.session_state.favorites.append(player_name)
            save_favorites(st.session_state.favorites)
            st.success(f"Added {player_name} to favorites.")

    if st.session_state.favorites:
        to_remove = st.multiselect("Remove from favorites", st.session_state.favorites)
        if to_remove and st.button("Remove selected"):
            st.session_state.favorites = [x for x in st.session_state.favorites if x not in to_remove]
            save_favorites(st.session_state.favorites)
            st.info("Updated favorites.")
        st.caption("Favorites: " + ", ".join(st.session_state.favorites))
    else:
        st.caption("No favorites yet.")

    st.markdown("---")
    st.subheader("📲 SMS Notifications")
    tw_sid   = st.text_input("Twilio Account SID", type="password")
    tw_token = st.text_input("Twilio Auth Token", type="password")
    tw_from  = st.text_input("From Number (+1xxxxxxxxxx)")
    tw_to    = st.text_input("To Number(s), comma-separated", value="")
    sms_time = st.time_input("Daily send time (server time)", value=dt.time(hour=10, minute=0))
    send_now = st.button("Send SMS Now for Favorites")

# ---------------------------------
# Header card
# ---------------------------------
st.markdown("""
<div class="card" style="margin-bottom: 12px;">
  <div style="display:flex; align-items:center; justify-content: space-between;">
    <div>
      <h1 style="margin:0;">Hot Shot Props — NBA Player Analytics</h1>
      <div class="badge">One-page • Favorites + SMS • ML predictions (if available)</div>
    </div>
    <div style="text-align:right; font-size:.9rem; color:#9aa3b2;">Trends • Weighted fallback • Clean UX</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# Main content
# ---------------------------------
if not player_id:
    st.info("Pick a player from the sidebar to load their dashboard.")
else:
    # Fetch data
    career_df, logs_df = fetch_player(player_id)
    if career_df.empty:
        st.warning("No career data available for this player.")
        st.stop()

    if not player_name:
        for p in get_active_players():
            if p['id'] == player_id:
                player_name = p['full_name']; break

    team_abbr = career_df['TEAM_ABBREVIATION'].iloc[-1] if 'TEAM_ABBREVIATION' in career_df.columns else "TBD"
    latest_season = str(career_df['SEASON_ID'].dropna().iloc[-1]) if 'SEASON_ID' in career_df.columns else "N/A"

    # Last game
    last_game_info = "N/A"
    last_game_stats = None
    if logs_df is not None and not logs_df.empty:
        lg_df = logs_df.head(1).copy()
        lg_cols = safe_cols(lg_df, STATS_COLS)
        if lg_cols:
            last_game_stats = lg_df[lg_cols].iloc[0].to_dict()
        lg_row  = lg_df.iloc[0]
        lg_date = pd.to_datetime(lg_row['GAME_DATE']).strftime("%Y-%m-%d") if 'GAME_DATE' in lg_row else "—"
        lg_opp  = extract_opp_from_matchup(lg_row.get('MATCHUP', '')) or "TBD"
        last_game_info = f"{lg_date} vs {lg_opp}"

    # Next game (with 🏠/✈️)
    team_id_map = {t['abbreviation']: t['id'] for t in all_t}
    team_id = team_id_map.get(team_abbr, None)
    ng = next_game_for_team(team_id) if team_id is not None else None
    if ng:
        icon = "🏠" if ng.get('home') else "✈️"
        next_game_info = f"{icon} {ng['date']} vs {ng['opp_abbr']}"
        is_home_next = 1 if ng.get('home') else 0
        ng_date_for_feat = dt.datetime.strptime(ng['date'], "%Y-%m-%d").date()
    else:
        next_game_info = "TBD"
        is_home_next = 0
        ng_date_for_feat = None

    # Header strip metrics
    c1, c2, c3, c4 = st.columns([1.3, 0.8, 1.2, 1.2])
    with c1: st.metric("Player", player_name)
    with c2: st.metric("Team", team_abbr)
    with c3: st.metric("Most Recent Game", last_game_info)
    with c4: st.metric("Next Game", next_game_info)

    # Headshot + trend charts
    col_img, col_trend = st.columns([0.28, 0.72])
    with col_img:
        img_bytes = cdn_headshot(player_id, "1040x760") or cdn_headshot(player_id, "260x190")
        if img_bytes:
            st.image(img_bytes, use_container_width=True, caption=f"{player_name} — media day headshot")
        else:
            st.info("Headshot not available.")

    with col_trend:
        if logs_df is not None and not logs_df.empty:
            N = min(10, len(logs_df))
            view = logs_df.head(N).copy().iloc[::-1]
            trend_cols = [c for c in ['PTS','REB','AST','FG3M'] if c in view.columns]
            if trend_cols:
                trend_df = view[['GAME_DATE'] + trend_cols].copy()
                trend_df['GAME_DATE'] = pd.to_datetime(trend_df['GAME_DATE'])
                for stat in trend_cols:
                    ch = alt.Chart(trend_df).mark_line(point=True).encode(
                        x=alt.X('GAME_DATE:T', title=''),
                        y=alt.Y(f'{stat}:Q', title=stat),
                        tooltip=[alt.Tooltip('GAME_DATE:T', title='Game'), alt.Tooltip(f'{stat}:Q', title=stat)]
                    ).properties(height=110)
                    st.altair_chart(ch, use_container_width=True)
            else:
                st.info("No trend stats available.")
        else:
            st.info("No recent games to chart.")

    # Metric rows
    def metric_row(title: str, data: dict | pd.Series | pd.DataFrame, fallback: str = "N/A"):
        st.markdown(f"#### {title}")
        if data is None:
            data = {}
        if isinstance(data, pd.DataFrame):
            row = data.iloc[0].to_dict() if not data.empty else {}
        elif isinstance(data, pd.Series):
            row = data.to_dict()
        else:
            row = dict(data)

        def fmt(v):
            return f"{float(v):.2f}" if isinstance(v, (int,float,np.floating)) and pd.notna(v) else fallback

        cols = st.columns(5)
        with cols[0]: st.metric("PTS", fmt(row.get('PTS')))
        with cols[1]: st.metric("REB", fmt(row.get('REB')))
        with cols[2]: st.metric("AST", fmt(row.get('AST')))
        with cols[3]: st.metric("3PM", fmt(row.get('FG3M')))
        with cols[4]: st.metric("MIN", fmt(row.get('MIN')))

    # Row 1: Current season per-game
    season_row = None
    if not career_df.empty:
        cur = career_df.iloc[-1]
        gp  = cur.get('GP', 0)
        if gp and gp > 0:
            season_row = {c: (cur.get(c, 0)/gp) for c in STATS_COLS}
    metric_row("Current Season Averages", season_row)

    # Row 2: Last game
    metric_row("Last Game Stats", last_game_stats)

    # Row 3: Last 5 averages
    ra = recent_averages(logs_df)
    last5 = ra['Last 5 Avg'].iloc[0].to_dict() if 'Last 5 Avg' in ra and not ra['Last 5 Avg'].empty else None
    metric_row("Last 5 Games Averages", last5)

    # Row 4: Predicted next game (ML or fallback)
    ml_models = load_models()
    ml_preds  = predict_next_ml(logs_df, ng_date_for_feat, is_home_next, ml_models) if ml_models else None
    if ml_preds:
        metric_row("Predicted Next Game (ML)", ml_preds)
    else:
        preds = predict_next_fallback(logs_df, latest_season)
        metric_row("Predicted Next Game (Model)", preds)

# ---------------------------------
# SMS actions (manual & daily while open)
# ---------------------------------
def collect_favorite_predictions(fav_names: list[str]) -> list[str]:
    """Build one SMS line per favorite with prediction + next game (ML if available)."""
    if not fav_names:
        return []
    all_teams = get_all_teams()
    abbr_to_id = {t['abbreviation']: t['id'] for t in all_teams}
    # Map names to ids using combined index
    name_to_id = {}
    for label, pid in build_team_player_index().items():
        nm = label.split("—",1)[-1].strip()
        name_to_id[nm] = pid

    models = load_models()
    lines = []
    for name in fav_names:
        pid = name_to_id.get(name)
        if not pid:
            continue
        cdf, ldf = fetch_player(pid)
        if cdf.empty:
            continue
        team_abbr = cdf['TEAM_ABBREVIATION'].iloc[-1] if 'TEAM_ABBREVIATION' in cdf.columns else "TBD"
        latest_season = str(cdf['SEASON_ID'].dropna().iloc[-1]) if 'SEASON_ID' in cdf.columns else "N/A"
        tid = abbr_to_id.get(team_abbr)
        ng  = next_game_for_team(tid) if tid else None
        is_home_next = 1 if (ng and ng.get('home')) else 0
        ng_date_for_feat = dt.datetime.strptime(ng['date'], "%Y-%m-%d").date() if ng else None

        preds_ml = predict_next_ml(ldf, ng_date_for_feat, is_home_next, models) if models else None
        preds = preds_ml if preds_ml else (predict_next_fallback(ldf, latest_season) or {})
        icon  = "🏠" if ng and ng.get('home') else "✈️"
        next_game_info = f"{icon} {ng['date']} vs {ng['opp_abbr']}" if ng else "TBD"
        lines.append(format_prediction_sms(name, team_abbr, next_game_info, preds))
    return lines

# Manual send
if 'send_now_clicked' not in st.session_state:
    st.session_state.send_now_clicked = False

if send_now:
    st.session_state.send_now_clicked = True
    to_numbers = [n.strip() for n in (tw_to or "").split(",") if n.strip()]
    if not (tw_sid and tw_token and tw_from and to_numbers):
        st.error("Enter Twilio SID, token, from number, and at least one destination number.")
    elif not st.session_state.get("favorites"):
        st.error("Add at least one favorite player first.")
    else:
        lines = collect_favorite_predictions(st.session_state.favorites)
        if not lines:
            st.warning("No predictions available to send.")
        else:
            body = "🟦 Hot Shot Props — Daily Predictions\n" + "\n".join(lines)
            result = send_sms(body, to_numbers, tw_from, tw_sid, tw_token)
            if result["sent"] > 0:
                st.success(f"Sent {result['sent']} SMS.")
            if result["errors"]:
                st.warning("Errors: " + "; ".join(result["errors"]))

# Lightweight daily scheduler (runs while the app tab is open)
def _daily_sms_loop(schedule_time: dt.time, sid, token, from_num, to_csv):
    while st.session_state.get("_daily_sms_enabled", False):
        now = dt.datetime.now()
        last_sent_day = st.session_state.get("_last_sms_day")
        if (last_sent_day != now.date()) and (now.time() >= schedule_time):
            favs = st.session_state.get("favorites", [])
            if favs and sid and token and from_num and (to_csv or "").strip():
                to_numbers = [n.strip() for n in to_csv.split(",") if n.strip()]
                lines = collect_favorite_predictions(favs)
                if lines:
                    body = "🟦 Hot Shot Props — Daily Predictions\n" + "\n".join(lines)
                    send_sms(body, to_numbers, from_num, sid, token)
                    st.session_state["_last_sms_day"] = now.date()
        time.sleep(30)

with st.sidebar:
    st.markdown("---")
    st.subheader("⏰ Daily Sender (runs while app is open)")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Enable Daily SMS"):
            if not (tw_sid and tw_token and tw_from and (tw_to or "").strip()):
                st.error("Enter Twilio credentials and phone numbers first.")
            elif not st.session_state.get("favorites"):
                st.error("Add at least one favorite player first.")
            else:
                if not st.session_state.get("_daily_sms_enabled"):
                    st.session_state["_daily_sms_enabled"] = True
                    st.session_state["_last_sms_day"] = None
                    t = threading.Thread(target=_daily_sms_loop, args=(sms_time, tw_sid, tw_token, tw_from, tw_to), daemon=True)
                    t.start()
                st.success("Daily SMS enabled (this tab must remain open).")
    with colB:
        if st.button("Disable Daily SMS"):
            st.session_state["_daily_sms_enabled"] = False
            st.info("Daily SMS disabled.")

# Footer
st.markdown("""
<div style="margin-top:16px; display:flex; justify-content:space-between; align-items:center; opacity:.9;">
  <div class="badge">Favorites • Daily SMS while app is open</div>
  <div style="font-size:.85rem; color:#9aa3b2;">Hot Shot Props © — Built with nba_api & Streamlit</div>
</div>
""", unsafe_allow_html=True)
