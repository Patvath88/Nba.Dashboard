import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leaguegamefinder
from sklearn.ensemble import RandomForestRegressor
from apscheduler.schedulers.background import BackgroundScheduler
import time

FAV_PATH = "favorite_players.json"
PROJ_PATH = "saved_projections.csv"

# --------------- UTILITIES ---------------
def load_favorites():
    if os.path.exists(FAV_PATH):
        with open(FAV_PATH, "r") as f:
            return json.load(f)
    return []

def enrich(df):
    if df.empty:
        return df
    df["P+R"] = df["PTS"] + df["REB"]
    df["P+A"] = df["PTS"] + df["AST"]
    df["R+A"] = df["REB"] + df["AST"]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

def get_games(player_id, season="2025-26"):
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        gl = gl.sort_values("GAME_DATE")
        return gl
    except Exception:
        return pd.DataFrame()

def get_next_game_info(player_id):
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season="2025-26").get_data_frames()[0]
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        gl = gl.sort_values("GAME_DATE")
        team_code = gl.iloc[0]["MATCHUP"].split(" ")[0]
        games = leaguegamefinder.LeagueGameFinder(season_nullable="2025-26").get_data_frames()[0]
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
        team_games = games[games["MATCHUP"].str.contains(team_code)]
        future_games = team_games[team_games["GAME_DATE"] > pd.Timestamp.now()]
        if not future_games.empty:
            next_game = future_games.sort_values("GAME_DATE").iloc[0]
            matchup = next_game["MATCHUP"]
            date_str = next_game["GAME_DATE"].strftime("%Y-%m-%d")
            return date_str, matchup
        return "", ""
    except Exception:
        return "", ""

def predict_next(df):
    if df is None or len(df) < 3:
        return 0
    X = np.arange(len(df)).reshape(-1, 1)
    y = df.values
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    return round(float(model.predict([[len(df)]])[0]), 1)

def save_projection(player_name, projections, game_date, opponent):
    df = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "player": player_name,
        "game_date": game_date,
        "opponent": opponent,
        **projections
    }])
    if os.path.exists(PROJ_PATH):
        existing = pd.read_csv(PROJ_PATH)
        duplicate = existing[
            (existing["player"] == player_name) & (existing["game_date"] == game_date)
        ]
        if not duplicate.empty:
            return
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(PROJ_PATH, index=False)

# --------------- MAIN UPDATE LOGIC ---------------
def update_favorites():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating favorite players...")
    favorites = load_favorites()
    if not favorites:
        print("No favorite players found.")
        return
    nba_players = players.get_active_players()
    player_map = {p["full_name"]: p["id"] for p in nba_players}

    for fav in favorites:
        pid = player_map.get(fav)
        if not pid:
            continue
        current = enrich(get_games(pid))
        if current.empty:
            continue
        next_game_date, next_matchup = get_next_game_info(pid)
        if not next_game_date:
            continue
        pred_next = {}
        for stat in ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA","P+R","P+A","R+A"]:
            pred_next[stat] = predict_next(current[stat])
        save_projection(fav, pred_next, next_game_date, next_matchup)
        print(f"Saved projections for {fav} ({next_matchup} on {next_game_date})")
    print("✅ Daily projections updated.\n")

# --------------- SCHEDULER SETUP ---------------
if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_favorites, "cron", hour=9, minute=0)  # run every day 9:00 UTC
    scheduler.start()
    print("🕐 Daily favorite player projection updater running...")
    update_favorites()  # run once on start
    try:
        while True:
            time.sleep(3600)  # keep alive
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
