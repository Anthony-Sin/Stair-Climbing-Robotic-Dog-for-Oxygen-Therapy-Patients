import os
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        file_url = f"file:///{os.path.abspath('poster/index.html').replace(os.sep, '/')}"
        await page.goto(file_url)
        await page.wait_for_selector(".pchart2", timeout=15000)
        svgs_html = await page.evaluate('document.querySelector(".pchart2").innerHTML')
        with open('poster/svgs_dump.html', 'w', encoding='utf-8') as f:
            f.write(svgs_html)
        await browser.close()
        print('Dumped SVGs!')

asyncio.run(main())
