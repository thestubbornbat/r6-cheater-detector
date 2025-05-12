# get_cookies.py
import asyncio
from playwright.async_api import async_playwright

async def run():
    proxy_str = "http://xsdfoijl:ngu01gli7ye0@86.38.234.176:6630"  # Correct proxy format
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=user_agent,
            proxy={"server": proxy_str}
        )
        
        page = await context.new_page()
        print(f"🌐 Visiting tracker.gg to get new cf_clearance...")

        await page.goto("https://tracker.gg", timeout=60000)
        await page.wait_for_timeout(5000)

        cookies = await context.cookies()
        print('🍪 Cookies:', cookies)

        await browser.close()

asyncio.run(run())
