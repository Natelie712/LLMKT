import asyncio
import os
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from playwright.async_api import Page, BrowserContext

# Load environment variables
load_dotenv()

LOGIN_URL = "https://courses.lastingerlearning.com/d2l/login"

# Course Configuration - Change these to navigate to different courses
COURSE_CONFIGS = {
    "algebraic_reasoning": {
        "name": "Algebraic Reasoning",
        "url": "https://courses.lastingerlearning.com/d2l/home/12304"
    },
    # Add more courses here as needed
    # "another_course": {
    #     "name": "Another Course Name",
    #     "url": "https://courses.lastingerlearning.com/d2l/home/XXXXX"
    # }
}

# Select which course to navigate to
SELECTED_COURSE = "algebraic_reasoning"

async def main():
    username = os.getenv("COURSE_USERNAME", "")
    password = os.getenv("COURSE_PASSWORD", "")
    
    # Get course config
    course_config = COURSE_CONFIGS.get(SELECTED_COURSE)
    if not course_config:
        print(f"❌ Course configuration '{SELECTED_COURSE}' not found!")
        return
    
    course_name = course_config["name"]
    course_url = course_config["url"]

    async def on_page_context_created(page: Page, context: BrowserContext, **kwargs):
        print(f"🔐 Starting authentication for '{course_name}'...")
        
        # Step 1: Login first
        await page.goto(LOGIN_URL)
        await page.fill("#userName", username)
        await page.fill("#password", password)
        await page.click("button:has-text('Log In')")
        
        # Wait for successful login (can use any post-login indicator)
        await page.wait_for_selector("h2:has-text('My Courses')", timeout=15000)
        print("✅ Successfully logged in")
        
        # Step 2: Navigate directly to the course URL
        print(f"🎯 Navigating directly to '{course_name}' course...")
        await page.goto(course_url)
        
        # Step 3: Wait for course page to load (fastest, most robust approach)
        print("⏳ Waiting for course page to load...")
        try:
            # Method 1: Verify we're on the expected URL and network is stable
            await page.wait_for_load_state("domcontentloaded", timeout=8000)
            
            # Scalable URL verification - check if we're on the expected course URL
            current_url = page.url
            if course_url in current_url or current_url.startswith(course_url):
                print(f"✅ URL verification passed: {current_url}")
                
                # Wait for network to stabilize (shorter timeout since we have URL confirmation)
                await page.wait_for_load_state("networkidle", timeout=8000)
                print("✅ Course page loaded (URL + network verified)")
            else:
                print(f"⚠️ URL mismatch - Expected: {course_url}, Got: {current_url}")
                # Still wait for network idle but with longer timeout
                await page.wait_for_load_state("networkidle", timeout=10000)
                print("✅ Course page loaded (network verified despite URL mismatch)")
                
        except Exception as e:
            print(f"⚠️ Using minimal fallback load detection: {e}")
            # Minimal fallback - just wait for basic DOM
            await page.wait_for_timeout(3000)
            print("✅ Course page loaded (timeout fallback)")
        
        return page

    # Updated browser configuration to address compatibility issues
    browser_config = BrowserConfig(
        headless=False,
        verbose=True,
        # Use a more recent Chrome user agent to avoid compatibility warnings
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # Additional browser args to suppress compatibility warnings
        browser_type="chromium",  # Explicitly use chromium
        extra_args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-features=VizDisplayCompositor",
            "--disable-extensions",
            "--disable-plugins",
            "--disable-images",  # Speed up loading
            "--disable-javascript-harmony-shipping",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding"
        ]
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Register the hook
        crawler.crawler_strategy.set_hook("on_page_context_created", on_page_context_created)

        # Configure the crawler run
        crawler_run_config = CrawlerRunConfig(
            js_code="window.scrollTo(0, document.body.scrollHeight);",
            wait_for="body",
            cache_mode=CacheMode.BYPASS
        )

        # Use the course URL directly since our hook handles login
        result = await crawler.arun(course_url, config=crawler_run_config)

        if result.success:
            print(f"\n🎉 Successfully crawled '{course_name}' course page!")
            print("=" * 60)
            print(f"COURSE: {course_name}")
            print(f"URL: {course_url}")
            print("=" * 60)
            print("COURSE PAGE MARKDOWN:")
            print("=" * 60)
            print(result.markdown[:2000])  # Show first 2000 chars to avoid overwhelming output
            if len(result.markdown) > 2000:
                print(f"\n... (truncated, total length: {len(result.markdown)} characters)")
            print("=" * 60)
        else:
            print(f"❌ Failed to crawl '{course_name}' course page")
            print(f"Error: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())