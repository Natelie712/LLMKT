import asyncio
import os
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, DefaultMarkdownGenerator
from playwright.async_api import Page, BrowserContext

# Load environment variables
load_dotenv()

LOGIN_URL = "https://courses.lastingerlearning.com/d2l/login"

COURSE_CONFIGS = {
    "algebraic_reasoning": {
        "name": "Course Description Topic",
        "url": "https://courses.lastingerlearning.com/d2l/le/lessons/12304/topics/519732"
    },
}

SELECTED_COURSE = "algebraic_reasoning"

async def main():
    username = os.getenv("COURSE_USERNAME", "")
    password = os.getenv("COURSE_PASSWORD", "")
    
    course_config = COURSE_CONFIGS.get(SELECTED_COURSE)
    if not course_config:
        print(f"❌ Course configuration '{SELECTED_COURSE}' not found!")
        return
    
    course_name = course_config["name"]
    target_url = course_config["url"]

    async def on_page_context_created(page: Page, context: BrowserContext, **kwargs):
        print("🔐 Starting authentication...")
        await page.goto(LOGIN_URL)
        await page.fill("#userName", username)
        await page.fill("#password", password)
        await page.click("button:has-text('Log In')")
        await page.wait_for_selector("h2:has-text('My Courses')", timeout=15000)
        print("✅ Successfully logged in")
        return page

    async def after_goto(page: Page, context: BrowserContext, url: str, response, **kwargs):
        if url == target_url:
            print(f"🕵️‍♂️ Searching for nested iframe in {url}...")
            try:
                outer_frame_locator = page.frame_locator('.d2l-fra-iframe iframe')
                inner_frame_locator = outer_frame_locator.locator('d2l-iframe-wrapper-for-react')
                final_content_url = await inner_frame_locator.get_attribute('src')
                
                if final_content_url:
                    print(f"🎯 Found final content URL: {final_content_url}")
                    # Re-navigate to the final content page ---
                    print(f"↪️ Navigating to the final content page...")
                    await page.goto(final_content_url)
                else:
                    print("❌ Could not find the src attribute of the nested iframe.")
            except Exception as e:
                print(f"❌ Error while trying to find nested iframe: {e}")
        return page

    browser_config = BrowserConfig(headless=False, verbose=True)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Register the hooks
        crawler.crawler_strategy.set_hook("on_page_context_created", on_page_context_created)
        crawler.crawler_strategy.set_hook("after_goto", after_goto)

        print("\n--- Running crawler ---")
        result = await crawler.arun(target_url, config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            process_iframes=True,
            excluded_selector="#readspeaker_button_1",
            markdown_generator=DefaultMarkdownGenerator()
        ))

        if result.success:
            print(f"\n🎉 Successfully crawled '{course_name}'!")
            print("=" * 60)
            print("PAGE CONTENT (MARKDOWN):")
            print("=" * 60)
            print(result.markdown)
            print("=" * 60)
        else:
            print(f"❌ Failed to crawl '{course_name}'")
            print(f"Error: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())