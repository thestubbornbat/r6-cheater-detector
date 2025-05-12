import sys
import os
import json
import random
import joblib
import numpy as np
import pandas as pd
from time import sleep
from selenium import webdriver
from selenium_stealth import stealth
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

def parse_json(data):
    try:
        len_name_changes = len(data.get("data", {}).get("metadata", {}).get("nameChanges", []))
    except Exception:
        len_name_changes = np.nan

    parsed_data = {
        "platformUserId": data.get("data", {}).get("platformInfo", {}).get("platformUserId", np.nan),
        "isVerified": data.get("data", {}).get("userInfo", {}).get("isVerified", np.nan),
        "pageviews": data.get("data", {}).get("userInfo", {}).get("pageviews", np.nan),
        "xpTier": data.get("data", {}).get("userInfo", {}).get("xpTier", np.nan),
        "isSuspicious": data.get("data", {}).get("userInfo", {}).get("isSuspicious", np.nan),
        "nameChanges": len_name_changes,
        "currentSeason": data.get("data", {}).get("metadata", {}).get("currentSeason", np.nan),
        "clearanceLevel": data.get("data", {}).get("metadata", {}).get("clearanceLevel", np.nan),
        "isOverwolfAppUser": data.get("data", {}).get("metadata", {}).get("isOverwolfAppUser", np.nan),
        "battlepassLevel": data.get("data", {}).get("metadata", {}).get("battlepassLevel", np.nan),
    }

    num_seasons = 37
    season_data = {}

    for season in range(1, num_seasons + 1):
        for gamemode in ["Quick Match", "Ranked", "Standard", "Event", "Arcade"]:
            prefix = f"season_{season}_{gamemode.lower().replace(' ', '')}_"
            season_data.update({
                prefix + "matchesplayed": np.nan,
                prefix + "matcheswon": np.nan,
                prefix + "matcheslost": np.nan,
                prefix + "kills": np.nan,
                prefix + "deaths": np.nan,
                prefix + "kdRatio": np.nan,
                prefix + "mmr": np.nan,
                prefix + "maxMmr": np.nan,
            })

    for segment in data.get("data", {}).get("segments", []):
        if segment.get("type") == "season":
            season_number = segment.get("attributes", {}).get("season")
            gamemode = segment.get("metadata", {}).get("gamemodeName", "Unknown")

            if season_number and season_number <= num_seasons:
                prefix = f"season_{season_number}_{gamemode.lower().replace(' ', '')}_"
                stats = segment.get("stats", {})

                season_data[prefix + "matchesplayed"] = stats.get("matchesPlayed", {}).get("value", np.nan)
                season_data[prefix + "matcheswon"] = stats.get("matchesWon", {}).get("value", np.nan)
                season_data[prefix + "matcheslost"] = stats.get("matchesLost", {}).get("value", np.nan)
                season_data[prefix + "kills"] = stats.get("kills", {}).get("value", np.nan)
                season_data[prefix + "deaths"] = stats.get("deaths", {}).get("value", np.nan)
                season_data[prefix + "kdRatio"] = stats.get("kdRatio", {}).get("value", np.nan)
                season_data[prefix + "mmr"] = stats.get("mmr", {}).get("value", np.nan)
                season_data[prefix + "maxMmr"] = stats.get("maxMmr", {}).get("value", np.nan)

    overview_data = {}
    for segment in data.get("data", {}).get("segments", []):
        if segment.get("type") == "overview":
            for key, value in segment.get("stats", {}).items():
                overview_data[f"overview_{key.lower()}"] = value.get("value", np.nan)

    gamemode_data = {}
    for segment in data.get("data", {}).get("segments", []):
        if segment.get("type") == "gamemode":
            gamemode_name = segment.get("metadata", {}).get("gamemodeName", "Unknown").lower().replace(" ", "")
            for key, value in segment.get("stats", {}).items():
                gamemode_data[f"gamemode_{gamemode_name}_{key.lower()}"] = value.get("value", np.nan)

    parsed_data.update(season_data)
    parsed_data.update(overview_data)
    parsed_data.update(gamemode_data)

    return parsed_data

def scrape_player_data(name):
    try:
        print(f"[INFO] Scraping data for player: {name}...")
        options = webdriver.ChromeOptions()
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        ]
        user_agent = random.choice(user_agents)
        options.add_argument(f"user-agent={user_agent}")
        options.add_argument("--headless")

        driver = webdriver.Chrome(options=options)
        stealth(driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32",
                webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", fix_hairline=True)

        url = f"https://api.tracker.gg/api/v2/r6siege/standard/profile/ubi/{name}"
        driver.get(url)
        sleep(5)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()

        pre_tag = soup.find("pre")
        if not pre_tag:
            return None, "player doesnt exist"

        json_data = json.loads(pre_tag.text)
        print("[INFO] Successfully retrieved and parsed JSON data.")
        return json_data, None

    except Exception as e:
        print(f"[ERROR] Scraping failed: {e}")
        return None, "unable to scrape data"

def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_cheater.py <player_name>")
        sys.exit(1)

    player_name = " ".join(sys.argv[1:])  # Support multi-word names
    print(f"[INFO] Starting detection for: {player_name}")

    json_data, error = scrape_player_data(player_name)

    if json_data is None:
        print(f"[ERROR] {error or 'unable to scrape data'}")
        sys.exit(1)

    print("[INFO] Parsing player data...")
    try:
        parsed = parse_json(json_data)
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        sys.exit(1)

    df = pd.DataFrame([parsed])
    df.drop('platformUserId', axis=1, inplace=True, errors='ignore')

    print("[INFO] Loading feature names and aligning features...")
    feature_names = joblib.load("twenty_feature_names.pkl")

    for col in feature_names:
        if col.endswith("_missing"):
            base_col = col.replace("_missing", "")
            df[col] = df[base_col].isna().astype(int) if base_col in df else 1
        else:
            df[col] = df.get(col, np.nan)

    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_names]

    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    print("[INFO] Loading model and making prediction...")
    model = joblib.load("decision_tree_model.pkl")

    prediction = model.predict(df)[0]

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    if prediction == 0:
        print(f"[RESULT] {player_name} is NOT a cheater. Get better.")
    else:
        print(f"[RESULT] {player_name} is very likely a cheater!! Good sniffing.")

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == "__main__":
    main()
