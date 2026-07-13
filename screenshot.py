import os
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 3456, "height": 2304})
        
        file_path = "file:///" + os.path.abspath("poster/index.html").replace("\\", "/")
        print(f"Navigating to {file_path}")
        await page.goto(file_path, wait_until="networkidle")
        
        # Wait for Babel to compile React and render the DOM
        print("Waiting for React to render...")
        await page.wait_for_selector(".header", timeout=15000)
        # Add a short delay for any final CSS/images to settle
        await page.wait_for_timeout(2000)
        
        # Take a screenshot
        await page.screenshot(path="poster-preview.png", full_page=True)
        print("Screenshot saved to poster-preview.png")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
