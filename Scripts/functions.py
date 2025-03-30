from selenium import webdriver
from selenium_stealth import stealth
from selenium.webdriver.common.by import By
import random
from selenium.webdriver.common.keys import Keys
from time import sleep
import time
import os 
import json
from bs4 import BeautifulSoup
import numpy as np



def scrape_account(username, modifier='profile'):

    # Create an instance of ChromeOptions
    options = webdriver.ChromeOptions()

    # User-agent rotation
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    ]

    user_agent = random.choice(user_agents)
    options.add_argument(f"user-agent={user_agent}")

    # Initialize the WebDriver with options
    driver = webdriver.Chrome(options=options)

    # Apply stealth settings to the driver (assuming you have a stealth function)
    stealth(
        driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )


    driver.get(f"https://api.tracker.gg/api/v2/r6siege/standard/{modifier}/ubi/{username}")
        
    # Wait for the page to load completely
    sleep(5)

    # Wait for the page to load completely (if necessary)
    driver.implicitly_wait(10)  # Waits up to 10 seconds for elements to appear

    # Retrieve the page source
    json_content = driver.page_source

    # Close the browser
    driver.quit()

    return json_content



def extract_json(json_content):
        # Parse HTML structure and extract JSON inside <pre> tag
    soup = BeautifulSoup(json_content, "html.parser")
    pre_tag = soup.find("pre")
    if not pre_tag:
        print(f"No JSON data found")


    json_data = json.loads(pre_tag.text)  # Load JSON

    return json_data



def parse_json(data):
    # Extract top-level information 
    ''' IMPORTANT : WILL ONLY WORK IF MODIFIER IS 'profile'. (Need to write another function for the modifier 'matches')
    '''
    parsed_data = {
        "platformUserId": data["data"].get("platformInfo", {}).get("platformUserId", np.nan),
        "isVerified": data["data"].get("userInfo", {}).get("isVerified", np.nan),
        "pageviews": data["data"].get("userInfo", {}).get("pageviews", np.nan),
        "xpTier": data["data"].get("userInfo", {}).get("xpTier", np.nan),
        "isSuspicious": data["data"].get("userInfo", {}).get("isSuspicious", np.nan),
        "nameChanges": len(data["data"].get("metadata", {}).get("nameChanges", [])),
        "currentSeason": data["data"].get("metadata", {}).get("currentSeason", np.nan),
        "clearanceLevel": data["data"].get("metadata", {}).get("clearanceLevel", np.nan),
        "isOverwolfAppUser": data["data"].get("metadata", {}).get("isOverwolfAppUser", np.nan),
        "battlepassLevel": data["data"].get("metadata", {}).get("battlepassLevel", np.nan),
    }
    
    # Initialize dictionary for storing season data
    num_seasons = 37  # Considering 37 seasons
    season_data = {}
    
    for season in range(1, num_seasons + 1):
        for gamemode in ["Quick Match", "Ranked", "Standard", "Event", "Arcade"]:  # You can extend with more gamemodes
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
    
    # Parse segments
    for segment in data["data"].get("segments", []):
        if segment.get("type") == "season":
            season_number = segment.get("attributes", {}).get("season")
            gamemode = segment.get("metadata", {}).get("gamemodeName", "Unknown")
            
            if season_number is not None and season_number <= num_seasons:
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
    
    # Parse overview section
    overview_data = {}
    for segment in data["data"].get("segments", []):
        if segment.get("type") == "overview":
            stats = segment.get("stats", {})
            for key, value in stats.items():
                feature_name = f"overview_{key.lower()}"
                overview_data[feature_name] = value.get("value", np.nan)
    
    # Parse gamemode section
    gamemode_data = {}
    for segment in data["data"].get("segments", []):
        if segment.get("type") == "gamemode":
            gamemode_name = segment.get("metadata", {}).get("gamemodeName", "Unknown").lower().replace(" ", "")
            stats = segment.get("stats", {})
            for key, value in stats.items():
                feature_name = f"gamemode_{gamemode_name}_{key.lower()}"
                gamemode_data[feature_name] = value.get("value", np.nan)
    
    # Merge all extracted data
    parsed_data.update(season_data)
    parsed_data.update(overview_data)
    parsed_data.update(gamemode_data)
    
    return parsed_data


