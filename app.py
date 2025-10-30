"""
Streamlit Dashboard for NBA Player Statistics using the BALLDONTLIE API
====================================================================

This application provides a comprehensive look at an NBA player's recent
performance and calculates several key averages.  It supports the
following features:

* **Player search** – quickly look up any active or historical NBA player.
* **Recent performance** – compute averages over the last 5, 10 and 20
  games for core counting stats (points, rebounds, assists, steals,
  blocks, turnovers and minutes).
* **Season and career averages** – pull a player's current and prior
  season numbers and aggregate a career average by combining all
  available season data.
* **Head‑to‑head breakdown** – display the player’s per‑team averages
  so you can see how they perform against each opponent.
* **On‑demand projections** – a simple predictive model uses recent
  trends to forecast what the player might do in the next game.

To run this dashboard you will need a **GOAT tier** API key from
`ballDontLie.io` and a Python environment with the packages listed in
`requirements.txt` installed.  When the app starts it prompts for your
API key; the key is sent on each request via the `Authorization`
header.

**Important:**  The BALLDONTLIE API now organizes endpoints by sport
(e.g. `nba/v1`, `nfl/v1`).  This dashboard targets the NBA endpoints
exclusively.  Calling the deprecated `/v1` routes with an API key
results in HTTP 401 errors.  Make sure your API key is valid and
has NBA access.

Note: the BALLDONTLIE API uses cursor based pagination.  Helper
functions below transparently follow the `next_cursor` field to
retrieve all requested records.
"""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import time


###############################################################################
# Configuration
###############################################################################

# Base URL for the BALLDONTLIE API.
#
# Note: As of late 2024 the BALLDONTLIE API segments sport‑specific
# endpoints under the `/nba/v1` prefix.  Attempting to call the
# deprecated `/v1` paths with an API key will result in HTTP 401
# errors.  Therefore, the base URL below explicitly targets the
# NBA namespace.  If additional sports are needed you can adjust
# this constant (e.g. use `https://api.balldontlie.io/nfl/v1` for NFL).  
API_BASE_URL = "https://api.balldontlie.io/nba/v1"

# Path to persist the API key locally.  When the user enters a key via
# the dashboard it will be saved to this file, and on subsequent runs
# the stored key will be pre‑populated so the user does not need to
# re‑enter it.  The file lives alongside this script.  If you prefer
# another location you can modify the path below.
API_KEY_FILE = Path(__file__).with_name("api_key.txt")

def load_api_key_from_file() -> str:
    """Load a persisted API key from disk.

    Returns
    -------
    str
        The API key if present; otherwise an empty string.
    """
    try:
        if API_KEY_FILE.exists():
            return API_KEY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def save_api_key_to_file(key: str) -> None:
    """Persist the provided API key to disk.

    Parameters
    ----------
    key : str
        The API key to save.  Leading/trailing whitespace will be removed.
    """
    try:
        API_KEY_FILE.write_text(key.strip(), encoding="utf-8")
    except Exception:
        # If the file cannot be written (e.g. due to permissions) we
        # silently ignore the error.  The user will simply have to
        # enter the key again on the next run.
        pass


def build_headers(api_key: str) -> Dict[str, str]:
    """Return headers including the required Authorization token.

    Parameters
    ----------
    api_key : str
        Your BALLDONTLIE API key.

    Returns
    -------
    Dict[str, str]
        A dictionary of HTTP headers.
    """
    return {"Authorization": api_key} if api_key else {}


###############################################################################
# API Helpers
###############################################################################

@lru_cache(maxsize=128)
def search_players(query: str, api_key: str) -> List[Dict[str, Any]]:
    """Search for players by name.

    Uses the `players` endpoint with the `search` parameter to locate
    matching players.  Results are cached to avoid repeated network calls.

    Parameters
    ----------
    query : str
        Partial or full name to search for.
    api_key : str
        API key for authorization.

    Returns
    -------
    List[dict]
        List of players (each a dict) returned by the API.
    """
    url = f"{API_BASE_URL}/players"
    params: Dict[str, Any] = {"search": query, "per_page": 50}
    # Implement simple retry logic to handle transient errors and rate limiting (HTTP 429).
    # We'll attempt up to 3 times, waiting between attempts based on the Retry‑After header
    # or a default pause.  If the request continues to fail, an error is surfaced.
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            resp = requests.get(
                url, headers=build_headers(api_key), params=params, timeout=20
            )
        except Exception as ex:
            st.error(f"Player search failed: {ex}")
            return []
        # Success
        if resp.status_code == 200:
            return resp.json().get("data", [])
        # Rate limit: 429 Too Many Requests
        if resp.status_code == 429:
            # Determine wait time from Retry‑After header, defaulting to 2 seconds
            wait_time = 2
            try:
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    wait_time = int(float(retry_after))
            except Exception:
                pass
            st.warning(
                f"Rate limit reached, pausing for {wait_time} seconds..."
            )
            time.sleep(wait_time)
            continue
        # Other non‑success codes
        st.error(
            f"Failed to search players: {resp.status_code} – {resp.text}"
        )
        return []
    # If all attempts exhausted
    st.error(
        "Failed to search players after multiple attempts due to rate limiting. "
        "Please try again later."
    )
    return []


def fetch_stats(
    player_id: int,
    api_key: str,
    seasons: Optional[Iterable[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Retrieve game stats for a player.

    This helper wraps the `/stats` endpoint and automatically follows
    pagination cursors until either all data is retrieved or
    `max_records` records have been collected.  It supports filtering
    by one or more seasons and/or a date range.

    Parameters
    ----------
    player_id : int
        The player's unique identifier.
    api_key : str
        API key for authorization.
    seasons : iterable of int, optional
        One or more seasons (e.g. 2023 for the 2023‑24 season).  If
        provided, stats will be limited to these seasons.
    start_date : str, optional
        Filter stats occurring on or after this date (YYYY‑MM‑DD).
    end_date : str, optional
        Filter stats occurring on or before this date (YYYY‑MM‑DD).
    max_records : int, optional
        Maximum number of records to retrieve.  If None, all records
        available will be pulled.

    Returns
    -------
    list of dict
        Each element corresponds to a single game’s stat line.
    """
    # Stats endpoint is namespaced under NBA.  The old `/v1/stats`
    # route will return 401s when using an API key, so construct the
    # path relative to the NBA base.
    url = f"{API_BASE_URL}/stats"
    # Build base parameters.  We allow duplicate keys like player_ids[] and
    # seasons[] by assigning list values; requests will encode correctly.
    params: Dict[str, Any] = {"player_ids[]": [player_id], "per_page": 100}
    if seasons:
        params["seasons[]"] = list(seasons)
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    stats: List[Dict[str, Any]] = []
    cursor: Optional[int] = None
    while True:
        if cursor is not None:
            params["cursor"] = cursor
        try:
            resp = requests.get(
                url, headers=build_headers(api_key), params=params, timeout=30
            )
        except Exception as ex:
            st.error(f"Failed to fetch stats: {ex}")
            break
        # Success
        if resp.status_code == 200:
            payload = resp.json()
        elif resp.status_code == 429:
            # Hit rate limit; sleep and retry the same page
            wait_time = 2
            try:
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    wait_time = int(float(retry_after))
            except Exception:
                pass
            st.warning(
                f"Rate limit reached while fetching stats, pausing for {wait_time} seconds..."
            )
            time.sleep(wait_time)
            # Do not advance cursor; simply retry this iteration
            continue
        else:
            if resp.status_code == 401:
                st.error(
                    "Unauthorized – please verify your BALLDONTLIE API key and plan."
                )
            else:
                st.error(
                    f"Failed to fetch stats: {resp.status_code} – {resp.text}"
                )
            break
        data_batch = payload.get("data", [])
        stats.extend(data_batch)
        # Stop if we have gathered enough records
        if max_records is not None and len(stats) >= max_records:
            stats = stats[:max_records]
            break
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return stats


def fetch_season_average(
    player_id: int,
    api_key: str,
    season: int,
    season_type: str = "regular",
    category: str = "general",
    type: str = "base",
) -> Optional[Dict[str, Any]]:
    """Retrieve season average stats for a player.

    The `/season_averages/<category>` endpoint requires a season and
    additional type parameter.  This function returns the first (and
    typically only) record from the API or None if no data is found.

    Parameters
    ----------
    player_id : int
        Player ID to retrieve data for.
    api_key : str
        API key for authorization.
    season : int
        Year of the season (e.g., 2024 for the 2024‑25 season).  Season
        is interpreted relative to the starting calendar year.
    season_type : str, optional
        Type of season: "regular", "playoffs", "ist", or "playin".
    category : str, optional
        Category of averages to request; see API docs.  Defaults to
        "general".
    type : str, optional
        Specific statistics group; defaults to "base".  See API docs
        for valid pairs of category and type.

    Returns
    -------
    dict or None
        A dictionary containing the season average stats or None if
        nothing was returned.
    """
    # Build URL for season averages.  The BALLDONTLIE API exposes two
    # variations: `/season_averages` which returns general base
    # statistics, and `/season_averages/{category}` which supports
    # multiple categories (general, clutch, defense, shooting) and
    # statistics types.  To reduce the likelihood of an unauthorized
    # response we call the basic endpoint first.  If a specific
    # category is requested (not ``general``) we fall back to the
    # category variant.
    headers = build_headers(api_key)
    # Primary call to simple season averages endpoint
    if category == "general":
        url = f"{API_BASE_URL}/season_averages"
        params = {"season": season, "player_id": player_id}
    else:
        url = f"{API_BASE_URL}/season_averages/{category}"
        params = {
            "season": season,
            "season_type": season_type,
            "type": type,
            # API spec expects `player_ids` array for the category
            # endpoint; using player_ids[] ensures proper form encoding.
            "player_ids[]": [player_id],
        }
    # We implement a simple retry loop similar to search_players and fetch_stats
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
        except Exception as ex:
            st.warning(
                f"Season averages unavailable for {season}: {ex}"
            )
            return None
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            return data[0] if data else None
        if resp.status_code == 429:
            wait_time = 2
            try:
                ra = resp.headers.get("Retry-After")
                if ra is not None:
                    wait_time = int(float(ra))
            except Exception:
                pass
            st.warning(
                f"Rate limit reached while fetching season averages, pausing for {wait_time} seconds..."
            )
            time.sleep(wait_time)
            continue
        # other status codes
        st.warning(
            f"Season averages unavailable for {season}: {resp.status_code} – {resp.text}"
        )
        return None
    # If attempts exhausted
    st.warning(
        f"Season averages unavailable for {season} after multiple attempts due to rate limiting."
    )
    return None


###############################################################################
# Data Transformation Utilities
###############################################################################

def stats_to_dataframe(stats: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of raw stat dictionaries into a cleaned DataFrame.

    The BALLDONTLIE stats payload includes nested player, team and game
    objects alongside basic counting stats.  This function flattens
    nested fields, converts minute strings to integers and ensures
    numeric columns are numeric.

    Parameters
    ----------
    stats : list of dict
        Raw stat objects returned from the API.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per game.  Columns include core
        statistics plus metadata such as game date, opponent and
        whether the game was at home or away.
    """
    if not stats:
        return pd.DataFrame()

    # Flatten stats and nested fields
    records = []
    for entry in stats:
        base = {k: entry.get(k) for k in [
            "id",
            "min",
            "fgm",
            "fga",
            "fg_pct",
            "fg3m",
            "fg3a",
            "fg3_pct",
            "ftm",
            "fta",
            "ft_pct",
            "oreb",
            "dreb",
            "reb",
            "ast",
            "stl",
            "blk",
            "turnover",
            "pf",
            "pts",
        ]}
        # Convert minutes (string) to integer total minutes
        minutes_str = base.get("min")
        if minutes_str is not None:
            try:
                base["min"] = int(minutes_str)
            except (ValueError, TypeError):
                base["min"] = 0
        # Flatten nested game and team objects
        game = entry.get("game", {})
        base["game_id"] = game.get("id")
        base["game_date"] = game.get("date")
        base["season"] = game.get("season")
        base["postseason"] = game.get("postseason")
        base["home_team_id"] = game.get("home_team_id")
        base["visitor_team_id"] = game.get("visitor_team_id")
        base["home_team_score"] = game.get("home_team_score")
        base["visitor_team_score"] = game.get("visitor_team_score")
        team = entry.get("team", {})
        base["team_id"] = team.get("id")
        base["team_name"] = team.get("full_name")
        base["team_abbr"] = team.get("abbreviation")
        records.append(base)

    df = pd.DataFrame.from_records(records)
    # Ensure numeric columns are numeric
    numeric_cols = [
        "min",
        "fgm",
        "fga",
        "fg_pct",
        "fg3m",
        "fg3a",
        "fg3_pct",
        "ftm",
        "fta",
        "ft_pct",
        "oreb",
        "dreb",
        "reb",
        "ast",
        "stl",
        "blk",
        "turnover",
        "pf",
        "pts",
        "home_team_score",
        "visitor_team_score",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    # Convert game_date to datetime
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    # Determine opponent team ID
    df["opponent_team_id"] = np.where(
        df["team_id"] == df["home_team_id"], df["visitor_team_id"], df["home_team_id"]
    )
    return df


def compute_averages(df: pd.DataFrame, window: Optional[int] = None) -> pd.Series:
    """Compute mean stats over an optional rolling window.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing one row per game.  Must include the
        columns used for calculations (pts, reb, ast, stl, blk, turnover, min).
    window : int, optional
        Number of most recent games to include.  If None, all rows
        are used.

    Returns
    -------
    pandas.Series
        Series mapping stat names to their mean values.
    """
    if df.empty:
        return pd.Series(dtype=float)
    ordered = df.sort_values("game_date", ascending=False)
    if window:
        ordered = ordered.head(window)
    stats_cols = ["pts", "reb", "ast", "stl", "blk", "turnover", "min"]
    return ordered[stats_cols].mean()


def head_to_head_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate per‑opponent average stats.

    Parameters
    ----------
    df : pandas.DataFrame
        Game log data for the player.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by opponent team ID with mean statistics.
    """
    if df.empty:
        return pd.DataFrame()
    group_cols = ["opponent_team_id"]
    agg_cols = ["pts", "reb", "ast", "stl", "blk", "turnover", "min"]
    h2h = df.groupby(group_cols)[agg_cols].mean().reset_index()
    return h2h.sort_values("pts", ascending=False)


def weighted_prediction(last5: pd.Series, last10: pd.Series, season: pd.Series) -> pd.Series:
    """Compute a simple weighted projection for the next game.

    The weights emphasize recent performance (last 5 games) while
    incorporating broader context from the last 10 games and the full
    season.  Adjust weights here to tune the forecast.

    Parameters
    ----------
    last5 : pandas.Series
        Mean statistics from the player’s last 5 games.
    last10 : pandas.Series
        Mean statistics from the player’s last 10 games.
    season : pandas.Series
        Mean statistics from the current season.

    Returns
    -------
    pandas.Series
        Projected values for the next game.
    """
    # Define weights – heavier weight on the most recent games
    w5, w10, wSeason = 0.5, 0.3, 0.2
    # Align indices and fill missing values
    combined = pd.concat([last5, last10, season], axis=1, keys=["l5", "l10", "season"]).fillna(0)
    projection = w5 * combined["l5"] + w10 * combined["l10"] + wSeason * combined["season"]
    return projection


###############################################################################
# Streamlit UI
###############################################################################

def main() -> None:
    """Run the Streamlit dashboard."""
    # Configure the page with a custom NBA themed colour palette.  These
    # values approximate the official NBA colours (dark navy, royal blue,
    # and red).  The font is left as sans serif for broad compatibility.
    # Configure the page.  We deliberately avoid passing a theme
    # dictionary here since some Streamlit deployments do not yet
    # support the `theme` parameter on set_page_config.  Instead, we
    # apply our colour scheme via injected CSS below.
    st.set_page_config(
        page_title="NBA Player Dashboard – BALLDONTLIE",
        page_icon="🏀",
        layout="wide",
    )

    # Inject custom CSS for finer control over component appearance.
    st.markdown(
        """
        <style>
        /* Ensure the entire application uses our dark theme */
        .stApp {
            background-color: #041E42;
            color: #FFFFFF;
        }
        /* Headings inherit our colour palette */
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF;
            font-weight: 700;
        }
        /* Paragraphs and labels use a lighter grey */
        p, label, span, div[data-testid="stMetricValue"] {
            color: #D0D8E5;
        }
        /* Style metric containers with a blue background and rounded corners */
        div[data-testid="stMetric"] {
            background-color: #17408B !important;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        /* Metric labels should be lighter and uppercase */
        div[data-testid="stMetric"] > label {
            color: #F5F5F5 !important;
            font-size: 0.85rem;
        }
        /* Metric values in bold red */
        div[data-testid="stMetricValue"] {
            color: #C9082A !important;
            font-size: 1.6rem;
            font-weight: 700;
        }
        /* Style text inputs and select boxes */
        input[type="text"], input[type="password"], select {
            background-color: #17408B !important;
            color: #FFFFFF !important;
            border: 1px solid #C9082A !important;
        }
        /* Style DataFrame headers and rows */
        .dataframe tbody tr, .dataframe thead tr {
            background-color: #17408B;
            color: #FFFFFF;
        }
        /* Adjust spinner colour */
        .stSpinner > div > div {
            border-top-color: #C9082A !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title and description
    st.title("NBA Player Dashboard – BALLDONTLIE")
    st.markdown(
        """
        Use this dashboard to explore detailed NBA player statistics and
        generate a projection for their next game. To begin, enter your
        BALLDONTLIE API key (it will be saved locally) and search for a player.
        """
    )

    # API key section in the sidebar
    with st.sidebar:
        st.header("API Key")
        existing_key = load_api_key_from_file()
        key_input = st.text_input(
            "Enter your BALLDONTLIE API key",
            value=existing_key,
            type="password",
            help="Your API key will be saved locally so you don't have to re‑enter it each time."
        )
        # Save button persists the key to disk
        if st.button("Save API Key"):
            save_api_key_to_file(key_input)
            st.success("API key saved successfully.")
        # Use the newly entered key or the stored key
        api_key = (key_input or existing_key).strip()

    # Stop execution if no API key is provided
    if not api_key:
        st.info("Please enter and save your API key in the sidebar to begin.")
        return

    # Validate the API key with a lightweight request to avoid repeated 401s
    # Validate the API key.  We perform a simple HTTP call to verify
    # authentication rather than caching via experimental_singleton
    # because some Streamlit environments may not support it.
    def _validate_key(key: str) -> bool:
        try:
            test_url = f"{API_BASE_URL}/players"
            response = requests.get(
                test_url, headers=build_headers(key), params={"per_page": 1}, timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False

    if not _validate_key(api_key):
        st.error(
            "Your API key did not authenticate successfully. Please verify that it is a valid "
            "BALLDONTLIE key with NBA access and try again."
        )
        return

    # Player search input
    query = st.text_input("Search for a player", placeholder="LeBron James")
    player_options: List[Tuple[str, int]] = []
    players: List[Dict[str, Any]] = []
    if query and len(query) >= 3:
        with st.spinner("Searching players..."):
            players = search_players(query, api_key=api_key)
        player_options = [
            (
                f"{p['first_name']} {p['last_name']} (ID: {p['id']}, {p.get('team', {}).get('full_name', 'No Team')})",
                p["id"],
            )
            for p in players
        ]

    # Select a player if options are available
    selected_player: Optional[int] = None
    if player_options:
        label = [opt[0] for opt in player_options]
        vals = [opt[1] for opt in player_options]
        sel = st.selectbox(
            "Select a player", options=list(range(len(label))), format_func=lambda i: label[i]
        )
        selected_player = vals[sel]

    # Display stats once a player is chosen
    if selected_player is not None:
        # Fetch player object for name display
        player_obj = next((p for p in players if p["id"] == selected_player), None)
        full_name = f"{player_obj['first_name']} {player_obj['last_name']}" if player_obj else "Selected Player"
        st.header(f"Stats for {full_name}")

        # Season selection: from current year down to 1946
        current_year = datetime.now().year
        seasons = list(range(current_year, 1945, -1))
        col1, col2 = st.columns(2)
        with col1:
            current_season = st.selectbox("Current season", options=seasons, index=0)
        with col2:
            previous_season = st.selectbox(
                "Previous season", options=seasons, index=1 if len(seasons) > 1 else 0
            )

        # Fetch stats
        with st.spinner("Fetching stats..."):
            recent_stats = fetch_stats(selected_player, api_key, max_records=25)
            recent_df = stats_to_dataframe(recent_stats)
            season_stats = fetch_stats(selected_player, api_key, seasons=[current_season])
            season_df = stats_to_dataframe(season_stats)
            prev_stats = fetch_stats(selected_player, api_key, seasons=[previous_season])
            prev_df = stats_to_dataframe(prev_stats)

        # Compute averages
        last5_avg = compute_averages(recent_df, window=5)
        last10_avg = compute_averages(recent_df, window=10)
        last20_avg = compute_averages(recent_df, window=20)
        season_avg = compute_averages(season_df)
        prev_season_avg = compute_averages(prev_df)
        career_avg = compute_averages(pd.concat([season_df, prev_df, recent_df]))

        # Summary metrics
        st.subheader("Averages Summary")
        metrics_cols = st.columns(5)
        for idx, (label, series) in enumerate(
            [
                ("Last 5 Games", last5_avg),
                ("Last 10 Games", last10_avg),
                ("Last 20 Games", last20_avg),
                (f"Season {current_season}", season_avg),
                (f"Season {previous_season}", prev_season_avg),
            ]
        ):
            with metrics_cols[idx]:
                st.markdown(f"**{label}**")
                st.metric("Points", f"{series.get('pts', 0):.1f}")
                st.metric("Rebounds", f"{series.get('reb', 0):.1f}")
                st.metric("Assists", f"{series.get('ast', 0):.1f}")

        # Career summary metrics
        st.markdown("**Career Average (computed)**")
        st.metric("Points", f"{career_avg.get('pts', 0):.1f}")
        st.metric("Rebounds", f"{career_avg.get('reb', 0):.1f}")
        st.metric("Assists", f"{career_avg.get('ast', 0):.1f}")

        # Head‑to‑head averages
        st.subheader("Head‑to‑Head Averages (All Opponents)")
        h2h_df = head_to_head_averages(recent_df)
        if not h2h_df.empty:
            team_map = {
                row["team_id"]: row["team_name"]
                for row in recent_df[["team_id", "team_name"]]
                .dropna()
                .drop_duplicates()
                .to_dict("records")
            }

            def lookup_team_name(opponent_id: Any) -> str:
                return team_map.get(opponent_id, str(opponent_id))

            h2h_df["Opponent"] = h2h_df["opponent_team_id"].apply(lookup_team_name)
            h2h_table = h2h_df[["Opponent", "pts", "reb", "ast", "stl", "blk", "turnover", "min"]]
            h2h_table.rename(
                columns={
                    "pts": "Pts",
                    "reb": "Reb",
                    "ast": "Ast",
                    "stl": "Stl",
                    "blk": "Blk",
                    "turnover": "TO",
                    "min": "Min",
                },
                inplace=True,
            )
            st.dataframe(h2h_table.style.format("{:.1f}"), use_container_width=True)
        else:
            st.info("Not enough data for head‑to‑head averages.")

        # Recent game chart
        st.subheader("Last 20 Game Logs")
        if not recent_df.empty:
            chart_df = recent_df.sort_values("game_date")
            chart_df["game"] = chart_df["game_date"].dt.strftime("%Y-%m-%d")
            metrics = ["pts", "reb", "ast"]
            st.line_chart(chart_df.set_index("game")[metrics])
        else:
            st.info("No recent games available for charting.")

        # Projection for next game
        st.subheader("Projected Next Game")
        projection = weighted_prediction(last5_avg, last10_avg, season_avg)
        proj_cols = st.columns(len(projection))
        for i, (stat, value) in enumerate(projection.items()):
            with proj_cols[i]:
                st.metric(stat.upper(), f"{value:.1f}")


if __name__ == "__main__":
    main()
