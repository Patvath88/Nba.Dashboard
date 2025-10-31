# app.py
# -------------------------------------------------------------
# NBA Dashboard (patched: scoreboard + standings reliability)
# -------------------------------------------------------------
import os
import io
import time
from datetime import datetime, timedelta
import pytz
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
import streamlit as st

# -----------------------
# Page & Theme
# -----------------------
st.set_page_config(
    page_title="NBA Dashboard",
    page_icon="🏀",
    layout="wide"
)

# Minimal NBA.com-inspired dark styling
st.markdown("""
<style>
:root {
  --nba-blue:#17408B;
  --nba-red:#C9082A;
  --ink:#EDEEF1;
}
html, body, [class*="css"]  {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}
.block-container { padding-top: 1.2rem; }
h1,h2,h3 { letter-spacing: .2px; }
table { font-size: .95rem; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
div[data-testid="stMetricDelta"] svg { display: none; }
header[data-testid="stHeader"] { background: linear-gradient(90deg, var(--nba-blue), #0b1e4b); }
section.main > div:first-child { padding-top: 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Sidebar controls
# -----------------------
if "home_reset" not in st.session_state:
    st.session_state.home_reset = False

def _home_reset():
    st.session_state.home_reset = True

with st.sidebar:
    st.markdown("### 🏀 NBA Dashboard")
    st.button("🏠 Home Screen", on_click=_home_reset, use_container_width=True)
    st.caption("Tip: Data is cached for 10 minutes to reduce timeouts.")

# (Optional) A place to hardcode an API key if you later use paid APIs.
# This is NOT used for NBA Stats (which doesn't need a key, just headers).
USER_API_KEY = os.environ.get("USER_API_KEY", "").strip()
if USER_API_KEY:
    st.sidebar.success("Custom API key loaded from environment.")
else:
    st.sidebar.info("No custom API key set (not required for this page).")

# -----------------------
# Networking (headers + retries)  >>> THIS IS THE IMPORTANT FIX <<<
# -----------------------
def nba_headers() -> dict:
    # Imitate a modern browser; NBA blocks default Python UA.
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Connection": "keep-alive",
        # CORS-ish headers that help on some hosts
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }

def session_with_retries(total: int = 4, backoff: float = 0.6) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=total,
        connect=total,
        read=total,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=40)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(nba_headers())
    return s

SESSION = session_with_retries()

# Utilities
ET = pytz.timezone("US/Eastern")

def yesterday_et_mmddyyyy() -> str:
    now_et = datetime.now(ET)
    y = now_et - timedelta(days=1)
    return y.strftime("%m/%d/%Y")

def season_string_from_today() -> str:
    # NBA season designation like "2024-25"
    now = datetime.now(ET)
    year = now.year
    # Season rolls over around Oct
    if now.month >= 10:
        return f"{year}-{str(year+1)[-2:]}"
    else:
        return f"{year-1}-{str(year)[-2:]}"

def current_season_year_start() -> str:
    # Some endpoints want "2024-25", others want "2024-25" + SeasonType
    return season_string_from_today()

# -----------------------
# Data fetchers (cached)
# -----------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_linescore_yesterday() -> pd.DataFrame:
    """
    Uses stats.nba.com ScoreboardV2 to get yesterday results.
    Returns dataframe with: AWAY, PTS_VISITOR, HOME, PTS_HOME, GAME_STATUS_TEXT
    """
    base = "https://stats.nba.com/stats/scoreboardv2"
    params = {
        "GameDate": yesterday_et_mmddyyyy(),  # MM/DD/YYYY
        "LeagueID": "00",
        "DayOffset": "0"
    }
    r = SESSION.get(base, params=params, timeout=15)
    r.raise_for_status()
    js = r.json()

    # "LineScore" usually index 1, but safer to find by name
    line_idx = None
    for i, rs in enumerate(js.get("resultSets", [])):
        if (rs.get("name") or "").lower() == "linescore":
            line_idx = i
            break
    if line_idx is None:
        # fallback to commonly at 1
        line_idx = 1

    headers = js["resultSets"][line_idx]["headers"]
    rows = js["resultSets"][line_idx]["rowSet"]
    df = pd.DataFrame(rows, columns=headers)

    # The linescore is team-rows. Group into games by GAME_ID
    # We'll pivot to away/home concise table
    wanted_cols = [
        "GAME_ID","GAME_STATUS_TEXT","TEAM_ABBREVIATION","PTS","TEAM_ID","TEAM_CITY_NAME"
    ]
    df_small = df[wanted_cols].copy()

    # Identify home/away by joining with GameHeader if available
    # GameHeader contains HOME_TEAM_ID, VISITOR_TEAM_ID
    hdr_idx = None
    for i, rs in enumerate(js.get("resultSets", [])):
        if (rs.get("name") or "").lower() == "gameheader":
            hdr_idx = i
            break
    if hdr_idx is None:
        # Without GameHeader, try infer by points ordering (less reliable)
        # We'll still produce a two-row-per-game join later — but better to require header.
        pass

    home_map = {}
    away_map = {}
    if hdr_idx is not None:
        gh = pd.DataFrame(
            js["resultSets"][hdr_idx]["rowSet"],
            columns=js["resultSets"][hdr_idx]["headers"]
        )
        for _, row in gh.iterrows():
            home_map[row["GAME_ID"]] = row["HOME_TEAM_ID"]
            away_map[row["GAME_ID"]] = row["VISITOR_TEAM_ID"]

    out_rows = []
    for gid, group in df_small.groupby("GAME_ID"):
        # default values
        away_abbr = home_abbr = None
        away_pts = home_pts = None
        status = group["GAME_STATUS_TEXT"].iloc[0] if not group.empty else ""

        if gid in home_map and gid in away_map:
            for _, r in group.iterrows():
                if r["TEAM_ID"] == away_map[gid]:
                    away_abbr = r["TEAM_ABBREVIATION"]
                    away_pts = r["PTS"]
                elif r["TEAM_ID"] == home_map[gid]:
                    home_abbr = r["TEAM_ABBREVIATION"]
                    home_pts = r["PTS"]
        else:
            # Fallback: take first as away, second as home
            group_sorted = group.sort_values("TEAM_ABBREVIATION")
            if len(group_sorted) >= 2:
                g1, g2 = group_sorted.iloc[0], group_sorted.iloc[1]
                away_abbr, away_pts = g1["TEAM_ABBREVIATION"], g1["PTS"]
                home_abbr, home_pts = g2["TEAM_ABBREVIATION"], g2["PTS"]

        out_rows.append({
            "AWAY": away_abbr,
            "PTS_VISITOR": away_pts,
            "HOME": home_abbr,
            "PTS_HOME": home_pts,
            "GAME_STATUS_TEXT": status
        })

    out_df = pd.DataFrame(out_rows)
    # Clean types
    for col in ("PTS_VISITOR","PTS_HOME"):
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce").astype("Int64")
    return out_df

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_standings() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses stats.nba.com LeagueStandingsV3 to get current standings.
    Returns (east_df, west_df) sorted best->worst.
    """
    season = current_season_year_start()  # e.g., "2024-25"
    url = "https://stats.nba.com/stats/leaguestandingsv3"
    params = {
        "LeagueID": "00",
        "Season": season,
        "SeasonType": "Regular Season",
    }
    r = SESSION.get(url, params=params, timeout=15)
    r.raise_for_status()
    js = r.json()

    rs = js["resultSets"][0]
    headers = rs["headers"]
    rows = rs["rowSet"]
    df = pd.DataFrame(rows, columns=headers)

    # Normalize column names that matter
    # Common columns: "TeamName","TeamTriCode","Conference","W","L","W_PCT","ConfRank"
    # Some builds use slightly different labels. Create aliases:
    alias = {
        "TeamName": "TEAM",
        "TeamCity": "CITY",
        "TeamTriCode": "TRI",
        "Conference": "CONF",
        "W": "W",
        "L": "L",
        "W_PCT": "PCT",
        "ConfRank": "RANK",
    }
    # Add missing alias columns if absent
    for need in alias.keys():
        if need not in df.columns:
            # Try similar names
            for c in df.columns:
                if c.lower() == need.lower():
                    alias[need] = c
                    break

    def view_cols(x: pd.DataFrame) -> pd.DataFrame:
        cols = []
        # Conference rank as int
        rank_col = alias["ConfRank"] if "ConfRank" in alias else "ConfRank"
        if rank_col in x.columns:
            x[rank_col] = pd.to_numeric(x[rank_col], errors="coerce")
        # W/L/PCT as numerics
        for c in ("W","L","W_PCT"):
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        # Build display subset
        disp = pd.DataFrame({
            "Rank": x.get(alias.get("ConfRank","ConfRank")),
            "Team": x.get(alias.get("TeamName","TeamName")),
            "W": x.get(alias.get("W","W")),
            "L": x.get(alias.get("L","L")),
            "Win %": x.get(alias.get("W_PCT","W_PCT")),
        })
        disp = disp.sort_values(["Win %","W","Rank"], ascending=[False, False, True], na_position="last")
        # Re-rank after sort
        disp["Rank"] = range(1, len(disp)+1)
        return disp

    east = df[df[alias.get("Conference","Conference")] == "East"].copy()
    west = df[df[alias.get("Conference","Conference")] == "West"].copy()
    east_v = view_cols(east)
    west_v = view_cols(west)
    return east_v, west_v

# -----------------------
# UI Sections
# -----------------------
st.markdown("### Pick a player from the sidebar to load their dashboard.")
st.divider()

# ---- Last Night's Results ----
st.subheader("Last Night’s Results")

games_err = None
games_df = pd.DataFrame()
try:
    games_df = fetch_linescore_yesterday()
    if games_df.empty:
        st.info("No games found for yesterday (preseason/off days are possible).")
    else:
        st.dataframe(
            games_df[["AWAY","PTS_VISITOR","HOME","PTS_HOME","GAME_STATUS_TEXT"]],
            use_container_width=True,
            height=min(56 + 35 * max(1, len(games_df)), 420)
        )
except Exception as e:
    games_err = f"Game stats error: {e}"
    st.error(games_err)

st.divider()

# ---- Standings by Conference ----
st.subheader("Standings by Conference (Best → Worst)")

standings_err = None
east_df = west_df = pd.DataFrame()
try:
    east_df, west_df = fetch_standings()
except Exception as e:
    standings_err = f"Team stats standings: {e}"
    st.error(standings_err)

left, right = st.columns(2)
with left:
    st.markdown("#### Eastern Conference")
    if not east_df.empty:
        st.dataframe(east_df, use_container_width=True, height=min(56 + 35 * max(1, len(east_df)), 540))
    else:
        st.caption("Standings unavailable.")

with right:
    st.markdown("#### Western Conference")
    if not west_df.empty:
        st.dataframe(west_df, use_container_width=True, height=min(56 + 35 * max(1, len(west_df)), 540))
    else:
        st.caption("Standings unavailable.")

st.divider()

# ---- League Leaders (compact; resilient if blocked) ----
st.subheader("League Leaders (Per Game)")

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_leaders(stat_cat: str = "PTS") -> pd.DataFrame:
    """
    Uses stats.nba.com leagueleaders endpoint.
    stat_cat in {"PTS","REB","AST","STL","BLK","TOV","FG3M"}.
    """
    season = current_season_year_start()
    url = "https://stats.nba.com/stats/leagueleaders"
    params = {
        "LeagueID": "00",
        "PerMode": "PerGame",
        "StatCategory": stat_cat,
        "Season": season,
        "SeasonType": "Regular Season",
        "Scope": "S",
        "ActiveFlag": "Y",
    }
    r = SESSION.get(url, params=params, timeout=15)
    r.raise_for_status()
    js = r.json()
    rs = js["resultSet"]
    headers = rs["headers"]
    rows = rs["rowSet"]
    df = pd.DataFrame(rows, columns=headers)
    # Keep the main columns
    keep = {
        "PLAYER_ID":"PLAYER_ID",
        "PLAYER":"PLAYER",
        "TEAM":"TEAM",
        stat_cat: stat_cat
    }
    # Some payloads use "PTS","REB","AST" columns named exactly.
    cols = ["PLAYER_ID","PLAYER","TEAM"]
    if stat_cat in df.columns:
        cols.append(stat_cat)
    else:
        # find case-insensitively
        for c in df.columns:
            if c.lower() == stat_cat.lower():
                cols.append(c)
                break
    out = df[cols].head(10).copy()
    out.rename(columns={cols[-1]: stat_cat}, inplace=True)
    return out

def headshot_url(player_id: int) -> str:
    # A commonly working headshot route
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"

def player_link(player_id: int) -> str:
    return f"https://www.nba.com/player/{player_id}"

leaders_cols = st.columns(3)
cats = [("PTS","Points"), ("REB","Rebounds"), ("AST","Assists")]
for (cat, label), col in zip(cats, leaders_cols):
    with col:
        try:
            df = fetch_leaders(cat)
            if df.empty:
                st.caption(f"{label} leaders unavailable.")
            else:
                # Build simple table with links
                show = df.copy()
                show["Player"] = show.apply(
                    lambda r: f"[{r['PLAYER']}]({player_link(int(r['PLAYER_ID']))})", axis=1
                )
                show = show[["Player","TEAM",cat]]
                st.markdown(f"**Top {label}**")
                st.dataframe(show, use_container_width=True, height=410)
        except Exception as e:
            st.caption(f"{label} leaders unavailable: {e}")

st.divider()
st.caption("Data: stats.nba.com (with browser headers). If you still see timeouts, redeploy or retry in ~30s due to upstream rate limiting.")
