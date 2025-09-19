import asyncio
import os
import re
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, DefaultMarkdownGenerator
from playwright.async_api import Page, BrowserContext

load_dotenv()

LOGIN_URL = "https://courses.lastingerlearning.com/d2l/login"
HOMEPAGE_URL = "https://courses.lastingerlearning.com/d2l/home"

async def authenticate_and_login(crawler, session_id):
    """Handle authentication using session management"""
    username = os.getenv("COURSE_USERNAME", "")
    password = os.getenv("COURSE_PASSWORD", "")
    
    print("🔐 Starting authentication...")
    
    # Custom hook for login process
    async def login_hook(page: Page, context: BrowserContext, **kwargs):
        await page.goto(LOGIN_URL)
        await page.fill("#userName", username)
        await page.fill("#password", password)
        await page.click("button:has-text('Log In')")
        await page.wait_for_selector("h2:has-text('My Courses')", timeout=15000)
        print("✅ Successfully logged in")
        return page
    
    # Set the login hook and authenticate
    crawler.crawler_strategy.set_hook("on_page_context_created", login_hook)
    
    config = CrawlerRunConfig(
        session_id=session_id,
        cache_mode=CacheMode.BYPASS,
        wait_until="domcontentloaded",
        page_timeout=20000,
        process_iframes=True,
        excluded_selector="#readspeaker_button_1",
        markdown_generator=DefaultMarkdownGenerator()
    )
    
    print(f"🌐 ARUN CALL: Authenticating at {HOMEPAGE_URL}")
    result = await crawler.arun(HOMEPAGE_URL, config=config)
    
    # Remove the login hook after authentication
    crawler.crawler_strategy.set_hook("on_page_context_created", None)
    print("🔓 Login hook removed")
    
    return result

async def extract_course_info(page: Page):
    """Extract course name and ID from the timeline page"""
    nav_link = page.locator(".d2l-navigation-s-link").first
    
    if await nav_link.count() > 0:
        course_name = await nav_link.get_attribute("title") or "Unknown Course"
        href = await nav_link.get_attribute("href")
        course_id = "unknown"
        if href:
            course_id_match = re.search(r'/(\d+)$', href)
            course_id = course_id_match.group(1) if course_id_match else "unknown"
    else:
        course_name = "Unknown Course"
        course_id = "unknown"
    
    # Clean course name for folder naming
    course_name_clean = re.sub(r'[^\w\s-]', '', course_name).strip()
    course_name_clean = re.sub(r'[-\s]+', '_', course_name_clean)
    
    return {
        'name': course_name_clean,
        'id': course_id,
        'folder_name': f"{course_name_clean}_{course_id}"
    }

async def get_unit_content(page: Page):
    """Extract unit content from the current page using h1.screen-reader-only"""
    iframe = page.frame_locator('.d2l-fra-iframe iframe')
    
    # Check if this is actually a unit by looking for .d2l-heading-1 or .d2l-heading-2
    has_unit_heading = (await iframe.locator(".d2l-heading-1").count() > 0) or (await iframe.locator(".d2l-heading-2").count() > 0)
    
    if not has_unit_heading:
        # This is not a unit, return None so it can be processed as a topic
        return None
    
    # Get name from h1.screen-reader-only (preferred method)
    heading_element = iframe.locator("h1.screen-reader-only").first
    unit_name = await heading_element.text_content() if await heading_element.count() > 0 else "Unknown Unit"
    unit_name = unit_name.strip() if unit_name else "Unknown Unit"
    
    # Determine unit type and extract ID based on URL structure
    url = page.url
    unit_id = "unknown"
    unit_type = 'unit'
    
    # Check for unit: /units/ID
    unit_match = re.search(r'/units/(\d+)', url)
    if unit_match:
        unit_id = unit_match.group(1)
        unit_type = 'unit'
    else:
        # Check for sub-unit: /lessons/ID (when not at course level)
        lesson_match = re.search(r'/lessons/(\d+)/lessons/(\d+)', url)
        if lesson_match:
            unit_id = lesson_match.group(2)  # Second lessons ID is the sub-unit
            unit_type = 'sub-unit'
        else:
            # Fallback: try to extract any ID from URL
            fallback_match = re.search(r'/(\d+)(?:/|$)', url)
            unit_id = fallback_match.group(1) if fallback_match else "unknown"
    
    # Get unit description from .module-inner-container (if it exists)
    unit_description = ""
    container_element = iframe.locator(".module-inner-container").first
    if await container_element.count() > 0:
        # First try to get content from d2l-html-block's html attribute
        html_block = container_element.locator("d2l-html-block").first
        if await html_block.count() > 0:
            html_attr = await html_block.get_attribute("html")
            if html_attr:
                # Decode HTML entities and use that content
                import html
                unit_description = html.unescape(html_attr)
                print(f"  📝 Found content in d2l-html-block html attribute")
        
        # If no d2l-html-block content, try regular inner HTML
        if not unit_description:
            unit_description = await container_element.inner_html()
            if unit_description:
                print(f"  📝 Found content in module-inner-container inner HTML")
    
    if not unit_description:
        print(f"  ⚠️ No description content found for unit: {unit_name}")
    
    return {
        'type': unit_type,
        'title': unit_name,
        'id': unit_id,
        'content': unit_description,
        'url': page.url
    }

async def get_topic_content_html(page: Page):
    """Extract topic content HTML from various structures"""
    iframe = page.frame_locator('.d2l-fra-iframe iframe')
    
    # Get topic name from h1.screen-reader-only (preferred method)
    heading_element = iframe.locator("h1.screen-reader-only").first
    topic_name = await heading_element.text_content() if await heading_element.count() > 0 else "Unknown Topic"
    topic_name = topic_name.strip() if topic_name else "Unknown Topic"
    
    # Extract ID from URL based on structure
    url = page.url
    topic_id = "unknown"
    
    # Check for topic: /topics/ID
    topic_match = re.search(r'/topics/(\d+)', url)
    if topic_match:
        topic_id = topic_match.group(1)
    else:
        # Fallback: try to extract any ID from URL
        fallback_match = re.search(r'/(\d+)(?:/|$)', url)
        topic_id = fallback_match.group(1) if fallback_match else "unknown"
    
    # Wait a bit for dynamic content to load (videos, PDFs, etc.)
    await asyncio.sleep(2)
    
    # Check for video with captions (single iframe structure)
    caption_menu = iframe.locator("#d2l-menu-item").filter(has_text="Captions (.vtt)")
    if await caption_menu.count() > 0:
        # Get the VTT download link
        vtt_link = await caption_menu.get_attribute("href")
        return {
            'type': 'video_with_captions',
            'title': topic_name,
            'id': topic_id,
            'vtt_url': vtt_link,
            'url': page.url
        }
    
    # Check for .jump-to-activity in the outer iframe
    if await iframe.locator(".jump-to-activity").count() > 0:
        container_html = await iframe.locator(".jump-to-activity").first.inner_html()
        if container_html and container_html.strip():
            return {
                'type': 'jump_to_activity',
                'title': topic_name,
                'id': topic_id,
                'content_html': container_html,
                'url': page.url
            }
    
    # Check for d2l-pdf-viewer in the outer iframe (wait for it to load)
    pdf_viewer = iframe.locator("d2l-pdf-viewer").first
    if await pdf_viewer.count() > 0:
        # Wait for PDF to load
        try:
            await pdf_viewer.wait_for(state="visible", timeout=5000)
        except:
            pass  # Continue even if timeout
        
        container_html = await pdf_viewer.inner_html()
        if container_html and container_html.strip():
            return {
                'type': 'pdf_viewer',
                'title': topic_name,
                'id': topic_id,
                'content_html': container_html,
                'url': page.url
            }
    
    # Check if inner iframe exists
    resizing_iframe = iframe.frame_locator(".resizing-iframe iframe")
    has_inner_iframe = await resizing_iframe.locator("html").count() > 0
    
    # For structures with inner iframe, check the remaining content types
    if has_inner_iframe:
        # Wait for inner iframe content to load
        await asyncio.sleep(1)
        
        # 1. Check for .container-fluid (use .first to avoid strict mode violations)
        if await resizing_iframe.locator(".container-fluid").count() > 0:
            container_html = await resizing_iframe.locator(".container-fluid").first.inner_html()
            if container_html and container_html.strip():
                return {
                    'type': 'container_fluid',
                    'title': topic_name,
                    'id': topic_id,
                    'content_html': container_html,
                    'url': page.url
                }
        
        # 2. Check for .d2l_read_element_1 (direct content under this element)
        if await resizing_iframe.locator(".d2l_read_element_1").count() > 0:
            container_html = await resizing_iframe.locator(".d2l_read_element_1").first.inner_html()
            if container_html and container_html.strip():
                return {
                    'type': 'd2l_read_element',
                    'title': topic_name,
                    'id': topic_id,
                    'content_html': container_html,
                    'url': page.url
                }
        
        # Fallback: get entire inner iframe content if it has substantial content
        container_html = await resizing_iframe.locator("html").first.inner_html()
        if container_html and container_html.strip() and len(container_html.strip()) > 100:  # Only if substantial content
            return {
                'type': 'generic_content',
                'title': topic_name,
                'id': topic_id,
                'content_html': container_html,
                'url': page.url
            }
    
    # Check for video elements in outer iframe (for videos without captions)
    video_elements = await iframe.locator("video, iframe[src*='youtube'], iframe[src*='vimeo'], iframe[src*='video']").count()
    if video_elements > 0:
        return {
            'type': 'video_no_captions',
            'title': topic_name,
            'id': topic_id,
            'url': page.url
        }
    
    # If no specific content type detected but we have a title, it might be a video without captions
    if topic_name != "Unknown Topic" and not has_inner_iframe:
        return {
            'type': 'video_no_captions',
            'title': topic_name,
            'id': topic_id,
            'url': page.url
        }
    
    return None

async def process_unit_content(crawler, session_id, unit_info):
    """Process unit HTML content through markdown converter"""
    if unit_info.get('content') and unit_info['content'].strip():
        print(f"🌐 Processing unit content through markdown converter")
        
        raw_html_url = f"raw:{unit_info['content']}"
        
        config = CrawlerRunConfig(
            session_id=session_id,
            cache_mode=CacheMode.BYPASS,
            wait_until="networkidle",
            page_timeout=30000,
            process_iframes=True,
            excluded_selector="#readspeaker_button_1",
            markdown_generator=DefaultMarkdownGenerator()
        )
        
        try:
            result = await crawler.arun(raw_html_url, config=config)
            
            if result.markdown:
                unit_info['content'] = result.markdown
            else:
                # If markdown conversion fails, keep the HTML but add a header
                unit_info['content'] = f"# {unit_info['title']}\n\n{unit_info['content']}"
        except Exception as e:
            print(f"❌ Failed to process unit HTML content: {e}")
            # Keep the HTML but add a header
            unit_info['content'] = f"# {unit_info['title']}\n\n{unit_info['content']}"
    else:
        # No content, create a basic markdown structure
        unit_info['content'] = f"# {unit_info['title']}\n\nNo content available."
    
    return unit_info

async def scrape_topic_content(crawler, session_id, topic_info):
    """Scrape actual topic content based on topic type"""
    print(f"🌐 Processing {topic_info['type']} content: {topic_info['title']}")
    
    if topic_info['type'] == 'video_with_captions':
        # For videos with captions, download the VTT file
        if 'vtt_url' in topic_info and topic_info['vtt_url']:
            try:
                config = CrawlerRunConfig(
                    session_id=session_id,
                    cache_mode=CacheMode.BYPASS,
                    wait_until="networkidle",
                    page_timeout=30000
                )
                
                result = await crawler.arun(topic_info['vtt_url'], config=config)
                topic_info['content'] = result.html if result.html else "VTT content not available"
                topic_info['content_type'] = 'vtt'
            except Exception as e:
                print(f"❌ Failed to download VTT file: {e}")
                topic_info['content'] = "VTT download failed"
                topic_info['content_type'] = 'error'
        else:
            topic_info['content'] = "VTT URL not found"
            topic_info['content_type'] = 'error'
    
    elif topic_info['type'] == 'video_no_captions':
        # For videos without captions, just note that no transcript is available
        topic_info['content'] = f"# {topic_info['title']}\n\nThis is a video content with no available transcript to download."
        topic_info['content_type'] = 'markdown'
    
    elif 'content_html' in topic_info:
        # For content with HTML, process it through the crawler
        raw_html_url = f"raw:{topic_info['content_html']}"
        
        config = CrawlerRunConfig(
            session_id=session_id,
            cache_mode=CacheMode.BYPASS,
            wait_until="networkidle",
            page_timeout=30000,
            process_iframes=True,
            excluded_selector="#readspeaker_button_1",
            markdown_generator=DefaultMarkdownGenerator()
        )
        
        try:
            result = await crawler.arun(raw_html_url, config=config)
            
            if result.markdown:
                topic_info['content'] = result.markdown
                topic_info['content_type'] = 'markdown'
            else:
                topic_info['content'] = f"# {topic_info['title']}\n\nContent extraction failed"
                topic_info['content_type'] = 'markdown'
        except Exception as e:
            print(f"❌ Failed to process HTML content: {e}")
            topic_info['content'] = f"# {topic_info['title']}\n\nContent processing failed: {str(e)}"
            topic_info['content_type'] = 'markdown'
        
        # Clean up raw HTML
        topic_info.pop('content_html', None)
    
    else:
        # Fallback for unknown types
        topic_info['content'] = f"# {topic_info['title']}\n\nUnknown content type: {topic_info['type']}"
        topic_info['content_type'] = 'markdown'
    
    return topic_info

async def click_next_iterator(page: Page):
    """Click the next iterator button and return success status"""
    print("🔍 Waiting for iframe and iterator button to load...")
    
    # Wait for the main iframe and iterator button
    await page.wait_for_selector('.d2l-fra-iframe iframe', timeout=15000)
    await asyncio.sleep(2)
    
    iframe = page.frame_locator('.d2l-fra-iframe iframe')
    await iframe.locator("#iteratorButtonNext").wait_for(state="visible", timeout=15000)
    
    next_button = iframe.locator("#iteratorButtonNext button")
    
    if await next_button.count() == 0:
        print("❌ Next iterator button not found")
        return False
    
    # Check if button is enabled BEFORE trying to click
    is_disabled = await next_button.get_attribute('disabled')
    if is_disabled is not None:  # disabled attribute exists
        print("📄 Reached end of timeline (next button disabled)")
        return False
    
    print("▶️ Clicking next iterator button...")
    current_url = page.url
    await next_button.click()
    
    # Wait for navigation or content change
    try:
        await page.wait_for_url(lambda url: url != current_url, timeout=10000)
        await page.wait_for_load_state("networkidle", timeout=15000)
        print("✅ Successfully navigated to next item")
    except:
        # URL didn't change, wait for content update
        print("⏳ Waiting for content to update...")
        await asyncio.sleep(3)
        await page.wait_for_selector('.d2l-fra-iframe iframe', timeout=10000)
        await asyncio.sleep(2)
        print("✅ Content updated")
    
    return True

async def save_content(content_info, course_info, unit_id=None, output_dir="crawl_output"):
    """Save content to appropriately named file"""
    # Create course folder
    course_folder = os.path.join(output_dir, course_info['folder_name'])
    os.makedirs(course_folder, exist_ok=True)
    
    # Clean title for filename
    title_clean = re.sub(r'[^\w\s-]', '', content_info['title']).strip()
    title_clean = re.sub(r'[-\s]+', '_', title_clean)
    
    # Determine file extension and prefix based on content type
    if content_info['type'] == 'unit':
        type_prefix = "Unit"
        file_extension = ".md"
    elif content_info['type'] == 'sub-unit':
        type_prefix = "SubUnit"
        file_extension = ".md"
    elif content_info['type'] == 'video_with_captions' and content_info.get('content_type') == 'vtt':
        type_prefix = "Topic"
        file_extension = ".vtt"
    else:  # All other topic types
        type_prefix = "Topic"
        file_extension = ".md"
    
    # Include unit ID and content ID in filename for all content types to avoid overwrites
    if unit_id:
        # For topics under units: Topic_Title_UnitID_TopicID.md
        if content_info['type'] in ['unit', 'sub-unit']:
            filename = f"{type_prefix}_{title_clean}_{content_info['id']}{file_extension}"
        else:
            filename = f"{type_prefix}_{title_clean}_{unit_id}_{content_info['id']}{file_extension}"
    else:
        # For standalone content: Type_Title_ID.md
        filename = f"{type_prefix}_{title_clean}_{content_info['id']}{file_extension}"
    filepath = os.path.join(course_folder, filename)
    
    # Write content
    with open(filepath, 'w', encoding='utf-8') as f:
        if content_info.get('content'):
            f.write(content_info['content'])
        else:
            if file_extension == ".vtt":
                f.write(f"NOTE\nVTT content for: {content_info['title']}\nNo content available.")
            else:
                f.write(f"# {content_info['title']}\n\nNo content available.")
    
    print(f"  ✅ Saved: {filename}")
    return filepath

async def scrape_timeline_content(page: Page, crawler, session_id):
    """Main function to scrape all timeline content"""
    print("🎯 Starting timeline content scraping...")
    
    # Wait for timeline content to be fully loaded using dynamic conditions
    print("⏳ Waiting for timeline content to load...")
    
    # Wait for main iframe to be present
    await page.wait_for_selector('.d2l-fra-iframe iframe', timeout=15000)
    
    # Wait for iframe content to be loaded using a JavaScript condition
    await page.wait_for_function("""
        () => {
            const iframe = document.querySelector('.d2l-fra-iframe iframe');
            if (!iframe) return false;
            
            const iframeDoc = iframe.contentDocument;
            if (!iframeDoc) return false;
            
            // Check if either heading or iterator button is present and visible
            const heading = iframeDoc.querySelector('.d2l-heading-1');
            const iterator = iframeDoc.querySelector('#iteratorButtonNext');
            
            return (heading && heading.textContent.trim() !== '') || 
                   (iterator && iterator.offsetParent !== null);
        }
    """, timeout=20000)
    
    print("✅ Timeline content loaded and ready")
    
    # Extract course information
    course_info = await extract_course_info(page)
    print(f"📚 Course: {course_info['name']} (ID: {course_info['id']})")
    
    content_count = 0
    current_unit_id = None  # Track the current unit context
    
    while True:
        print(f"\n--- Processing item {content_count + 1} ---")
        
        # Extract unit context from URL regardless of content type
        url = page.url
        page_unit_id = None
        
        # Check if we're in a unit context: /lessons/COURSE_ID/units/UNIT_ID
        unit_context_match = re.search(r'/lessons/(\d+)/units/(\d+)', url)
        if unit_context_match:
            page_unit_id = unit_context_match.group(2)  # Unit ID
        else:
            # Check if we're in a sub-unit context: /lessons/COURSE_ID/lessons/SUBUNIT_ID  
            subunit_context_match = re.search(r'/lessons/(\d+)/lessons/(\d+)', url)
            if subunit_context_match:
                page_unit_id = subunit_context_match.group(2)  # Sub-unit ID as context
        
        # Update current unit context if we found one
        if page_unit_id:
            current_unit_id = page_unit_id
        
        # Get content from the page
        unit_info = await get_unit_content(page)
        topic_info = await get_topic_content_html(page)
        
        # Determine content type and process accordingly
        if unit_info:
            # This is a unit/sub-unit (has unit heading indicators)
            unit_info = await process_unit_content(crawler, session_id, unit_info)
            print(f"📖 {unit_info['type'].title()}: {unit_info['title']} (ID: {unit_info['id']})")
            await save_content(unit_info, course_info, current_unit_id)
        elif topic_info:
            # This is a topic (no unit heading, but has topic content)
            topic_info = await scrape_topic_content(crawler, session_id, topic_info)
            print(f"📖 Topic ({topic_info['type']}): {topic_info['title']} (ID: {topic_info['id']}, Unit: {current_unit_id})")
            await save_content(topic_info, course_info, current_unit_id)
        else:
            # Fallback case - shouldn't happen often
            print("⚠️ No recognizable content found on this page")
            content_count -= 1  # Don't count this iteration
        
        content_count += 1
        
        # Try to go to next item
        has_next = await click_next_iterator(page)
        if not has_next:
            print(f"\n🎉 Completed scraping! Processed {content_count} items.")
            break
            
        # Small delay between iterations
        await asyncio.sleep(1)
    
    return content_count

async def main():
    session_id = "timeline_session"
    browser_config = BrowserConfig(headless=False, verbose=True)
    
    # Initialize crawler manually
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    try:
        print("\n--- Starting authentication ---")
        await authenticate_and_login(crawler, session_id)
        
        print("\n--- Starting timeline navigation and scraping ---")
        # Now set up the navigation hook for timeline scraping
        async def after_goto(page: Page, context: BrowserContext, url: str, response, **kwargs):
            print(f"📄 Navigated to: {url}")
            
            # Only process when we're on the homepage
            if url != HOMEPAGE_URL:
                return page
                
            print("🎯 Looking for courses to analyze...")
            await page.wait_for_load_state("domcontentloaded")
            await page.wait_for_selector("d2l-enrollment-card", timeout=15000)
            
            course_elements = await page.locator("d2l-enrollment-card").all()
            if not course_elements:
                print("❌ No course elements found")
                return page
                
            print(f"✅ Found {len(course_elements)} course elements")
            
            # Process all courses
            for course_index, element in enumerate(course_elements):
                print(f"\n📚 Processing course {course_index + 1}/{len(course_elements)}...")
                
                # Get course title
                title = "Unknown Course"
                card_element = element.locator("d2l-card").first
                if await card_element.count() > 0:
                    title = await card_element.get_attribute("text") or "Unknown Course"
                
                print(f"  📖 Course title: {title}")
                
                # Click course link
                potential_link = element.locator("d2l-card >> a").first
                if await potential_link.count() == 0:
                    print(f"  ❌ No clickable link found, skipping course")
                    continue
                
                current_url = page.url
                print(f"  🔗 Clicking course link...")
                await potential_link.click()
                
                await page.wait_for_url(lambda url: url != current_url, timeout=15000)
                await page.wait_for_load_state("domcontentloaded")
                print(f"  ✅ Navigated to course: {page.url}")
                
                # Navigate to Timeline
                timeline_element = page.locator("a:has-text('Timeline')").first
                await timeline_element.wait_for(state="visible", timeout=15000)
                
                if await timeline_element.count() > 0:
                    print(f"  ✅ Found Timeline button, clicking...")
                    await timeline_element.click()
                    await page.wait_for_load_state("domcontentloaded")
                    
                    print(f"  ✅ Successfully navigated to Timeline: {page.url}")
                    
                    # Start scraping timeline content
                    content_count = await scrape_timeline_content(page, crawler, session_id)
                    print(f"  🎉 Completed course {course_index + 1}: {content_count} items processed")
                else:
                    print(f"  ❌ No Timeline button found, skipping course")
                
                # Navigate back to homepage for next course (unless this is the last course)
                if course_index < len(course_elements) - 1:
                    print(f"  🏠 Returning to homepage for next course...")
                    await page.goto(HOMEPAGE_URL)
                    await page.wait_for_load_state("domcontentloaded")
                    await page.wait_for_selector("d2l-enrollment-card", timeout=15000)
                    
                    # Re-get course elements since we navigated back
                    course_elements = await page.locator("d2l-enrollment-card").all()
                    print(f"  ✅ Back at homepage, ready for next course")
            
            print(f"\n🎊 All courses completed! Processed {len(course_elements)} courses total.")
            return page

        # Set the navigation hook
        crawler.crawler_strategy.set_hook("after_goto", after_goto) # pyright: ignore[reportAttributeAccessIssue]
        
        # Navigate to homepage to trigger the course discovery and timeline scraping
        config = CrawlerRunConfig(
            session_id=session_id,
            cache_mode=CacheMode.BYPASS,
            wait_until="domcontentloaded",
            page_timeout=20000,
            process_iframes=True,
            excluded_selector="#readspeaker_button_1",
            markdown_generator=DefaultMarkdownGenerator()
        )
        
        print(f"🌐 ARUN CALL: Starting timeline scraping at {HOMEPAGE_URL}")
        await crawler.arun(HOMEPAGE_URL, config=config)
        
        print(f"\n🎉 Completed timeline scraping!")
        
    finally:
        await crawler.close()
        print("✅ Crawler closed")

if __name__ == "__main__":
    asyncio.run(main())