import os
import json
import random
import time
import requests
from fake_useragent import UserAgent
from playwright.sync_api import sync_playwright

# === CONFIG ===

proxies = [
    "86.38.234.176:6630:xsdfoijl:ngu01gli7ye0",
    "173.211.0.148:6641:xsdfoijl:ngu01gli7ye0",
    "161.123.152.115:6360:xsdfoijl:ngu01gli7ye0",
    "216.10.27.159:6837:xsdfoijl:ngu01gli7ye0",
    "154.36.110.199:6853:xsdfoijl:ngu01gli7ye0",
    "45.151.162.198:6600:xsdfoijl:ngu01gli7ye0",
    "185.199.229.156:7492:xsdfoijl:ngu01gli7ye0",
    "185.199.228.220:7300:xsdfoijl:ngu01gli7ye0",
    "185.199.231.45:8382:xsdfoijl:ngu01gli7ye0",
    "38.153.152.244:9594:xsdfoijl:ngu01gli7ye0",
]

bad_proxies = set()

with open('../cheater_usernames_statscc/usernames.txt', 'r') as f:
    cheater_data = json.load(f)

cheaters = [name for sublist in cheater_data.values() for name in sublist]

path_json_storage = "../cheater_data_statscc"
os.makedirs(path_json_storage, exist_ok=True)
scraped_cheaters = [f.replace(".json", "") for f in os.listdir(path_json_storage) if f.endswith(".json")]

# === HELPERS ===

def get_random_proxy():
    while True:
        proxy = random.choice(proxies)
        if proxy in bad_proxies:
            continue
        ip, port, user, pwd = proxy.split(":")
        proxy_auth = f"http://{user}:{pwd}@{ip}:{port}"
        proxy_dict = {"http": proxy_auth, "https": proxy_auth}
        if test_proxy(proxy_dict):
            return proxy_dict, proxy_auth
        else:
            print(f"❌ Bad proxy: {proxy_auth}")
            bad_proxies.add(proxy)

def test_proxy(proxy_dict):
    try:
        r = requests.get("https://httpbin.org/ip", proxies=proxy_dict, timeout=8)
        return r.status_code == 200
    except:
        return False

def get_random_user_agent():
    try:
        ua = UserAgent()
        ua_str = ua.random
        while "Mobile" in ua_str or "iPhone" in ua_str or "Android" in ua_str:
            ua_str = ua.random
        return ua_str
    except:
        return random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        ])

def get_cf_clearance_cookie(proxy_str, user_agent):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=user_agent,
            proxy={"server": proxy_str}
        )
        page = context.new_page()
        try:
            print(f"🌐 Visiting tracker.gg to get new cf_clearance...")
            page.goto("https://tracker.gg", timeout=60000)
            page.wait_for_timeout(5000)
            cookies = context.cookies()
            for cookie in cookies:
                if cookie["name"] == "cf_clearance":
                    print(f"✅ Got cf_clearance: {cookie['value'][:12]}...")
                    return cookie["value"]
        except Exception as e:
            print(f"[!] Failed to get cf_clearance: {e}")
        finally:
            browser.close()
    return None

# === SCRAPE LOOP ===

for name in cheaters:
    if name in scraped_cheaters:
        continue

    print(f"\n--- Scraping: {name} ---")

    for attempt in range(3):
        proxy, proxy_str = get_random_proxy()
        user_agent = get_random_user_agent()

        print(f"🧭 UA: {user_agent}")
        print(f"🔀 Proxy: {proxy_str}")

        cf_clearance = get_cf_clearance_cookie(proxy_str, user_agent)

        if not cf_clearance:
            print(f"⚠️ Retrying cookie for {name} (attempt {attempt + 1}/3)")
            bad_proxies.add(proxy_str)
            continue

        headers = {
            "User-Agent": user_agent,
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://tracker.gg/",
            "Origin": "https://tracker.gg",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }

        cookies = {
            "cf_clearance": cf_clearance
        }

        url = f"https://api.tracker.gg/api/v2/r6siege/standard/profile/ubi/{name}"

        try:
            response = requests.get(url, headers=headers, cookies=cookies, proxies=proxy, timeout=30)
            response.raise_for_status()
            data = response.json()

            file_path = os.path.join(path_json_storage, f"{name}.json")
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)

            print(f"✅ Saved {name} → {file_path}")
            break

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for {name}: {e}")
            bad_proxies.add(proxy_str)

        except ValueError:
            print(f"❌ Failed to parse JSON for {name}")
            break

        time.sleep(5)

    else:
        print(f"❌ Skipped {name} after 3 failed attempts.")

    delay = random.randint(60, 180)
    print(f"⏳ Waiting {delay}s before next scrape...\n")
    time.sleep(delay)
