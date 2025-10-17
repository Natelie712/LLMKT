import asyncio
import html
import os
import re
import shutil
import traceback
import urllib.parse
from urllib.parse import urljoin
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, DefaultMarkdownGenerator
from playwright.async_api import Page, BrowserContext, TimeoutError as PlaywrightTimeoutError

load_dotenv()

LOGIN_URL = "https://courses.lastingerlearning.com/d2l/login"
HOMEPAGE_URL = "https://courses.lastingerlearning.com/d2l/home"

async def authenticate_and_login(crawler, session_id):
    """Handle authentication using session management"""
    username = os.getenv("COURSE_USERNAME", "")
    password = os.getenv("COURSE_PASSWORD", "")
    
    print("Starting authentication...")
    
    # Custom hook for login process
    async def login_hook(page: Page, context: BrowserContext, **kwargs):
        await page.goto(LOGIN_URL)
        await page.get_by_label("Username").fill(username)
        await page.get_by_label("Password").fill(password)
        await page.get_by_role("button", name="Log In").click()
        await page.get_by_role("heading", name="My Courses").wait_for(state="visible", timeout=15000)
        print("Successfully logged in")
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
    
    print(f"ARUN CALL: Authenticating at {HOMEPAGE_URL}")
    result = await crawler.arun(HOMEPAGE_URL, config=config)
    
    crawler.crawler_strategy.set_hook("on_page_context_created", None)
    print("Login hook removed")
    
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
    
    has_unit_heading = (await iframe.locator(".d2l-heading-1").count() > 0) or (await iframe.locator(".d2l-heading-2").count() > 0)
    
    if not has_unit_heading:
        return None
    
    heading_element = iframe.locator("h1.screen-reader-only").first
    unit_name = await heading_element.text_content() if await heading_element.count() > 0 else "Unknown Unit"
    unit_name = unit_name.strip() if unit_name else "Unknown Unit"
    
    url = page.url
    unit_id = "unknown"
    unit_type = 'unit'
    unit_match = re.search(r'/units/(\d+)', url)
    if unit_match:
        unit_id = unit_match.group(1)
        unit_type = 'unit'
    else:
        lesson_match = re.search(r'/lessons/(\d+)/lessons/(\d+)', url)
        if lesson_match:
            unit_id = lesson_match.group(2)
            unit_type = 'sub-unit'
        else:
            fallback_match = re.search(r'/(\d+)(?:/|$)', url)
            unit_id = fallback_match.group(1) if fallback_match else "unknown"
    
    unit_description = ""
    container_element = iframe.locator(".module-inner-container").first
    if await container_element.count() > 0:
        html_block = container_element.locator("d2l-html-block").first
        if await html_block.count() > 0:
            html_attr = await html_block.get_attribute("html")
            if html_attr:
                unit_description = html.unescape(html_attr)
                print(f"  Found content in d2l-html-block html attribute")
        
        if not unit_description:
            unit_description = await container_element.inner_html()
            if unit_description:
                print(f"  Found content in module-inner-container inner HTML")
    
    if not unit_description:
        print(f"  No description content found for unit: {unit_name}")
    
    return {
        'type': unit_type,
        'title': unit_name,
        'id': unit_id,
        'content': unit_description,
        'url': page.url
    }

async def get_pdf_download_info(page: Page, topic_info):
    """Extract PDF download information for save_content to handle"""
    print(f"  Preparing PDF download info: {topic_info['title']}")
    
    try:
        iframe = page.frame_locator('.d2l-fra-iframe iframe')
        pdf_viewer = iframe.locator("d2l-pdf-viewer").first
        await pdf_viewer.wait_for(state="visible", timeout=15000)

        hover_target = pdf_viewer.locator("#viewerContainer").first
        if await hover_target.count() == 0:
            hover_target = pdf_viewer.locator("canvas").first
        if await hover_target.count() == 0:
            hover_target = pdf_viewer

        await hover_target.hover()
        await asyncio.sleep(0.1)

        download_button = pdf_viewer.get_by_role("button", name="Download").first
        await download_button.wait_for(state="visible", timeout=5000)
            
        if await download_button.count() > 0:
            print("  Found download button for PDF")
            return {
                'has_download_button': True,
                'page': page,
                'iframe': iframe,
                'pdf_viewer': pdf_viewer,
                'hover_target': hover_target,
                'download_button': download_button
            }
        else:
            raise Exception("No download button found")
            
    except Exception as e:
        print(f"  Error preparing PDF download: {e}")
        return None

async def get_vtt_download_info(page: Page, topic_info):
    """Prepare VTT download details for save_content to handle"""

    print(f"  Preparing VTT download info: {topic_info['title']}")

    try:
        iframe = page.frame_locator('.d2l-fra-iframe iframe')

        # Ensure the media player has rendered before interacting
        await iframe.locator("d2l-content-activity-renderer").first.wait_for(state="visible", timeout=15000)

        settings_button = iframe.get_by_role("button", name="settings")
        await settings_button.wait_for(state="visible", timeout=10000)

        transcript_menu_item = iframe.locator("#transcript-viewer-menu-item")
        transcript_download_button = iframe.locator("#video-transcript-download-button")
        captions_menu_item = iframe.locator('d2l-menu-item[aria-label="Captions (.vtt)"]')

        return {
            'page': page,
            'settings_button': settings_button,
            'transcript_menu_item': transcript_menu_item,
            'transcript_download_button': transcript_download_button,
            'captions_menu_item': captions_menu_item
        }
    except Exception as e:
        print(f"  Error preparing VTT download: {e}")
        return None


async def get_topic_content_html(page: Page):
    """Extract topic content HTML from various structures"""
    iframe = page.frame_locator('.d2l-fra-iframe iframe')
    
    # Get topic name from screen reader heading
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
    
    # Wait for content activity renderer to load before checking for video elements
    content_renderer = iframe.locator("d2l-content-activity-renderer").first
    if await content_renderer.count() > 0:
        print(f"  Found d2l-content-activity-renderer, waiting for content to load...")
        await content_renderer.wait_for(state="visible", timeout=15000)

        # Wait for track elements to attach before checking (video detection method)
        track_locator = iframe.locator("track")
        try:
            await track_locator.first.wait_for(state="attached", timeout=10000)
        except PlaywrightTimeoutError:
            print("  Track element did not appear within timeout window")

        if await track_locator.count() > 0:
            print(f"  Found track element, detecting video with captions")
            return {
                'type': 'video_with_captions',
                'title': topic_name,
                'id': topic_id,
                'url': page.url
            }
        else:
            print(f"  No track element found, detecting video without captions")
            return {
                'type': 'video_no_captions',
                'title': topic_name,
                'id': topic_id,
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
        await pdf_viewer.wait_for(state="visible", timeout=5000)
        return {
            'type': 'pdf',
            'title': topic_name,
            'id': topic_id,
            'url': page.url
        }
    
    # Check if inner iframe exists
    resizing_iframe = iframe.frame_locator(".resizing-iframe iframe")
    has_inner_iframe = await resizing_iframe.locator("html").count() > 0
    
    # For structures with inner iframe, check the remaining content types
    if has_inner_iframe:
        if await resizing_iframe.locator(".container-fluid").count() > 0:
            container_html = await resizing_iframe.locator(".container-fluid").first.inner_html()
            
            # Check for d2l-cplus-accordion elements and extract from data-panels attribute
            accordion = resizing_iframe.locator("d2l-cplus-accordion").first
            if await accordion.count() > 0:
                data_panels = await accordion.get_attribute("data-panels")
                data_instructions = await accordion.get_attribute("data-instructions")
                
                if data_panels:
                    print(f"  Found d2l-cplus-accordion element with data-panels attribute")
                    import json
                    
                    # HTML-unescape the JSON string
                    unescaped_panels = html.unescape(data_panels)
                    
                    try:
                        panels = json.loads(unescaped_panels)
                        
                        # Build accordion HTML with titles and content
                        accordion_html = ""
                        if data_instructions:
                            accordion_html += f"<p><strong>{html.unescape(data_instructions)}</strong></p>\n"
                        
                        for panel in panels:
                            panel_title = panel.get('title', '')
                            panel_content = panel.get('content', '')
                            accordion_html += f"<h3>{panel_title}</h3>\n{panel_content}\n"
                        
                        # Get the outer HTML of the accordion element to replace it in container_html
                        accordion_outer_html = await accordion.evaluate("el => el.outerHTML")
                        
                        # Replace the accordion element with extracted content
                        container_html = container_html.replace(accordion_outer_html, accordion_html)
                        print(f"  Extracted {len(panels)} accordion panels")
                    except json.JSONDecodeError as e:
                        print(f"  Failed to parse accordion data-panels JSON: {e}")
            
            # Check for .accordion class elements and extract from .card-title and .card-body
            # First check if accordion with card structure exists using count() with short timeout
            has_card_titles = await resizing_iframe.locator(".accordion .card-title").count() > 0
            has_card_bodies = await resizing_iframe.locator(".accordion .card-body").count() > 0
            
            if has_card_titles or has_card_bodies:
                print(f"  Found .accordion element with card structure, extracting content")
                bootstrap_accordion = resizing_iframe.locator(".accordion").first
                card_titles = await resizing_iframe.locator(".accordion .card-title").all()
                card_bodies = await resizing_iframe.locator(".accordion .card-body").all()
                
                accordion_html = ""
                
                # Match titles with bodies (they should be in the same order)
                for i in range(max(len(card_titles), len(card_bodies))):
                    if i < len(card_titles):
                        card_title_element = card_titles[i]
                        card_title_text = await card_title_element.text_content()
                        if card_title_text and card_title_text.strip():
                            accordion_html += f"<h3>{card_title_text.strip()}</h3>\n"
                    
                    if i < len(card_bodies):
                        card_body_element = card_bodies[i]
                        card_body_html = await card_body_element.inner_html()
                        accordion_html += f"{card_body_html}\n"
                
                # Only replace if we successfully extracted content
                if accordion_html.strip():
                    # Get the outer HTML of the accordion element to replace it in container_html
                    accordion_outer_html = await bootstrap_accordion.evaluate("el => el.outerHTML")
                    
                    # Replace the accordion element with extracted content
                    container_html = container_html.replace(accordion_outer_html, accordion_html)
                    print(f"  Extracted {len(card_titles)} titles and {len(card_bodies)} card-body sections")
            
            # Check for d2l-cplus-tabs-container elements and extract from data-tabs-data attribute
            tabs_container = resizing_iframe.locator("d2l-cplus-tabs-container").first
            if await tabs_container.count() > 0:
                data_tabs = await tabs_container.get_attribute("data-tabs-data")
                data_instruction = await tabs_container.get_attribute("data-instruction")
                
                if data_tabs:
                    print(f"  Found d2l-cplus-tabs-container element with data-tabs-data attribute")
                    import json
                    
                    # HTML-unescape the JSON string
                    unescaped_tabs = html.unescape(data_tabs)
                    
                    try:
                        tabs = json.loads(unescaped_tabs)
                        
                        # Build tabs HTML with titles and content
                        tabs_html = ""
                        if data_instruction:
                            tabs_html += f"<p><strong>{html.unescape(data_instruction)}</strong></p>\n"
                        
                        for tab in tabs:
                            tab_title = tab.get('title', '')
                            tab_content = tab.get('content', '')
                            tabs_html += f"<h3>{tab_title}</h3>\n{tab_content}\n"
                        
                        # Get the outer HTML of the tabs container element to replace it in container_html
                        tabs_outer_html = await tabs_container.evaluate("el => el.outerHTML")
                        
                        # Replace the tabs container element with extracted content
                        container_html = container_html.replace(tabs_outer_html, tabs_html)
                        print(f"  Extracted {len(tabs)} tab sections")
                    except json.JSONDecodeError as e:
                        print(f"  Failed to parse tabs data-tabs-data JSON: {e}")
            
            if container_html and container_html.strip():
                return {
                    'type': 'container_fluid',
                    'title': topic_name,
                    'id': topic_id,
                    'content_html': container_html,
                    'url': page.url
                }
        
        # Check for .d2l_read_element_1 (direct content under this element)
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
    
    return None

async def process_unit_content(crawler, session_id, unit_info):
    """Process unit HTML content through markdown converter"""
    if unit_info.get('content') and unit_info['content'].strip():
        print(f"Processing unit content through markdown converter")
        
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
            print(f"Failed to process unit HTML content: {e}")
            # Keep the HTML but add a header
            unit_info['content'] = f"# {unit_info['title']}\n\n{unit_info['content']}"
    else:
        # No content, create a basic markdown structure
        unit_info['content'] = f"# {unit_info['title']}\n\nNo content available."
    
    return unit_info

async def scrape_topic_content(crawler, session_id, topic_info):
    """Scrape actual topic content based on topic type"""
    print(f"Processing {topic_info['type']} content: {topic_info['title']}")
    
    if topic_info['type'] == 'video_with_captions':
        # For videos with captions using the new d2l-labs-media-player-video detection, 
        # mark it as needing VTT download
        topic_info['content_type'] = 'vtt'
        topic_info['needs_vtt_download'] = True
        print(f"  Video with captions detected - VTT download will be handled in main loop")
    
    elif topic_info['type'] == 'video_no_captions':
        # For videos without captions, just note that no transcript is available
        topic_info['content'] = f"# {topic_info['title']}\n\nThis is a video content with no available transcript to download."
        topic_info['content_type'] = 'markdown'
    
    elif topic_info['type'] == 'pdf':
        # For PDFs, we need to trigger the download and store the file path
        # The actual download will be handled separately by the main scraping loop
        # For now, just mark it as a PDF content type
        topic_info['content_type'] = 'pdf'
        topic_info['needs_pdf_download'] = True
        print(f"  PDF detected - download will be handled in main loop")
    
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
            print(f"Failed to process HTML content: {e}")
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
    
    iframe = page.frame_locator('.d2l-fra-iframe iframe')
    await iframe.locator("#iteratorButtonNext").wait_for(state="visible", timeout=15000)
    
    next_button = iframe.locator("#iteratorButtonNext button")
    
    if await next_button.count() == 0:
        print("Next iterator button not found")
        return False
    
    is_disabled = await next_button.get_attribute('disabled')
    if is_disabled is not None:
        print("Reached end of timeline (next button disabled)")
        return False
    
    print("Clicking next iterator button...")
    current_url = page.url

    heading_locator = iframe.locator("h1.screen-reader-only").first
    previous_heading = None
    if await heading_locator.count() > 0:
        try:
            previous_heading = (await heading_locator.text_content()) or None
        except PlaywrightTimeoutError:
            previous_heading = None

    await next_button.click()

    # Wait for navigation to complete
    try:
        await page.wait_for_url(lambda url: url != current_url, timeout=10000)
    except PlaywrightTimeoutError:
        print("  URL did not change after iterator click; continuing with iframe checks")

    if previous_heading:
        heading_changed = False
        for _ in range(30):
            try:
                current_heading = await heading_locator.text_content()
            except PlaywrightTimeoutError:
                current_heading = None

            if current_heading and current_heading.strip() != previous_heading.strip():
                heading_changed = True
                break

            await asyncio.sleep(0.5)

        if not heading_changed:
            print("  Heading did not change after iterator click; proceeding regardless")

    print("Successfully navigated to next item")
    
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
    elif content_info['type'] == 'pdf' and content_info.get('content_type') == 'pdf':
        type_prefix = "Topic"
        file_extension = ".pdf"
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
    
    # Handle different file types
    if file_extension == ".pdf":
        # For PDF files, handle the download directly to final location
        if content_info.get('pdf_download_info'):
            download_info = content_info['pdf_download_info']
            page = download_info['page']
            download_button = download_info['download_button']
            pdf_viewer = download_info.get('pdf_viewer')
            hover_target = download_info.get('hover_target')

            print(f"  Downloading PDF directly to: {filename}")

            target = hover_target or pdf_viewer or download_button
            if target:
                try:
                    await target.hover()
                    await asyncio.sleep(0.1)
                except Exception as hover_error:
                    print(f"  Hover preparation skipped: {hover_error}")

            try:
                await download_button.scroll_into_view_if_needed()
            except Exception as scroll_error:
                print(f"  Scroll into view skipped: {scroll_error}")

            await download_button.wait_for(state="visible", timeout=5000)

            async with page.expect_download(timeout=30000) as download_ctx:
                await download_button.evaluate("button => button.click()")

            download = await download_ctx.value
            await download.save_as(filepath)
            print(f"  Saved PDF: {filename}")

            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                raise Exception(f"PDF download produced empty file: {filename}")
        else:
            raise Exception(f"No PDF download info available for: {filename}")
    elif file_extension == ".vtt":
        download_info = content_info.get('vtt_download_info')

        if download_info and download_info.get('page'):
            page = download_info['page']
            settings_button = download_info['settings_button']
            transcript_menu_item = download_info['transcript_menu_item']
            transcript_download_button = download_info['transcript_download_button']
            captions_menu_item = download_info['captions_menu_item']

            try:
                print(f"  Downloading VTT via UI sequence: {filename}")

                await settings_button.click()

                await transcript_menu_item.wait_for(state="attached", timeout=10000)
                await transcript_menu_item.scroll_into_view_if_needed()

                await transcript_menu_item.wait_for(state="visible", timeout=10000)
                await transcript_menu_item.click()

                await transcript_download_button.wait_for(state="attached", timeout=10000)
                await transcript_download_button.scroll_into_view_if_needed()
                await transcript_download_button.wait_for(state="visible", timeout=10000)
                await transcript_download_button.click()

                await captions_menu_item.wait_for(state="attached", timeout=10000)
                await captions_menu_item.scroll_into_view_if_needed()
                await captions_menu_item.wait_for(state="visible", timeout=10000)
                download_promise = page.wait_for_event('download', timeout=30000)
                await asyncio.sleep(2)
                # Use evaluate to trigger native JS click and bypass overlay and Playwright issues
                await captions_menu_item.evaluate('(element) => element.click()')

                download = await download_promise
                await download.save_as(filepath)
                print(f"  Saved VTT: {filename}")
            except Exception as e:
                print(f"  VTT download via UI failed: {e}")
                with open(filepath, 'w', encoding='utf-8') as f:
                    if content_info.get('content'):
                        f.write(content_info['content'])
                    else:
                        f.write(f"NOTE\nVTT content for: {content_info['title']}\nNo content available.")
                print(f"  Saved VTT placeholder: {filename}")
        elif content_info.get('vtt_file_path') and os.path.exists(content_info['vtt_file_path']):
            shutil.copy2(content_info['vtt_file_path'], filepath)
            print(f"  Saved VTT: {filename}")
            
            try:
                os.remove(content_info['vtt_file_path'])
            except Exception as e:
                print(f"  Could not clean up temp file: {e}")
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                if content_info.get('content'):
                    f.write(content_info['content'])
                else:
                    f.write(f"NOTE\nVTT content for: {content_info['title']}\nNo content available.")
            print(f"  Saved VTT placeholder: {filename}")
    else:
        # For text files (markdown, vtt, etc.)
        with open(filepath, 'w', encoding='utf-8') as f:
            if content_info.get('content'):
                f.write(content_info['content'])
            else:
                if file_extension == ".vtt":
                    f.write(f"NOTE\nVTT content for: {content_info['title']}\nNo content available.")
                else:
                    f.write(f"# {content_info['title']}\n\nNo content available.")
        print(f"  Saved: {filename}")
    
    return filepath

async def scrape_timeline_content(page: Page, crawler, session_id):
    """Main function to scrape all timeline content"""
    print("Starting timeline content scraping...")
    
    print("Waiting for timeline content to load...")
    
    # Wait for main iframe to be present
    await page.wait_for_selector('.d2l-fra-iframe iframe', timeout=15000)
    
    iframe = page.frame_locator('.d2l-fra-iframe iframe')
    
    # Wait for iframe content to actually load by waiting for key elements
    await iframe.locator(".d2l-heading-1, .d2l-heading-2, #iteratorButtonNext").first.wait_for(state="visible", timeout=20000)
    
    print("Timeline content loaded and ready")
    
    # Extract course information
    course_info = await extract_course_info(page)
    print(f"Course: {course_info['name']} (ID: {course_info['id']})")
    
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
            print(f"{unit_info['type'].title()}: {unit_info['title']} (ID: {unit_info['id']})")
            await save_content(unit_info, course_info, current_unit_id)
        elif topic_info:
            # This is a topic (no unit heading, but has topic content)
            topic_info = await scrape_topic_content(crawler, session_id, topic_info)
            
            # Handle PDF downloads if needed
            if topic_info.get('needs_pdf_download') and topic_info['type'] == 'pdf':
                print(f"📥 Preparing PDF download: {topic_info['title']}")
                
                # Get PDF download information for save_content to handle
                pdf_download_info = await get_pdf_download_info(page, topic_info)
                
                if pdf_download_info:
                    topic_info['pdf_download_info'] = pdf_download_info
                    print(f"  PDF download info prepared")
                else:
                    raise Exception(f"PDF download preparation failed for {topic_info['title']}")
                    
            # Handle VTT downloads if needed
            if topic_info.get('needs_vtt_download') and topic_info['type'] == 'video_with_captions':
                print(f"📝 Preparing VTT download: {topic_info['title']}")

                vtt_download_info = await get_vtt_download_info(page, topic_info)

                if vtt_download_info:
                    topic_info['vtt_download_info'] = vtt_download_info
                    print("  VTT download info prepared")
                else:
                    print("  VTT download preparation failed")
            
            print(f"Topic ({topic_info['type']}): {topic_info['title']} (ID: {topic_info['id']}, Unit: {current_unit_id})")
            await save_content(topic_info, course_info, current_unit_id)
        else:
            print("No recognizable content found on this page")
            content_count -= 1
        
        content_count += 1
        
        has_next = await click_next_iterator(page)
        if not has_next:
            print(f"\nCompleted scraping! Processed {content_count} items.")
            break
            
        # Small delay between iterations
        await asyncio.sleep(1)
        # await page.wait_for_load_state("networkidle")
    
    return content_count

async def main():
    session_id = "timeline_session"
    
    # Create downloads directory for temporary PDF downloads
    downloads_path = os.path.join(os.getcwd(), "downloads")
    os.makedirs(downloads_path, exist_ok=True)
    
    browser_config = BrowserConfig(
        headless=False, 
        verbose=True,
        accept_downloads=True,
        downloads_path=downloads_path
    )
    
    # Initialize crawler manually
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    try:
        print("\n--- Starting authentication ---")
        await authenticate_and_login(crawler, session_id)
        
        print("\n--- Starting timeline navigation and scraping ---")
        # Now set up the navigation hook for timeline scraping
        async def after_goto(page: Page, context: BrowserContext, url: str, response, **kwargs):
            print(f"Navigated to: {url}")
            
            # Only process when we're on the homepage
            if url != HOMEPAGE_URL:
                return page
                
            print("Looking for courses to analyze...")
            await page.wait_for_load_state("domcontentloaded")
            await page.locator("d2l-enrollment-card").first.wait_for(state="visible", timeout=15000)
            course_elements = await page.locator("d2l-enrollment-card").all()
            
            if not course_elements:
                print("No course elements found")
                return page
                
            print(f"Found {len(course_elements)} course elements")
            
            for course_index, element in enumerate(course_elements):
                print(f"\nProcessing course {course_index + 1}/{len(course_elements)}...")
                
                title = "Unknown Course"
                card_element = element.locator("d2l-card").first
                if await card_element.count() > 0:
                    title = await card_element.get_attribute("text") or "Unknown Course"
                
                print(f"  Course title: {title}")
                
                # Click course link
                potential_link = element.get_by_role("link").first
                if await potential_link.count() == 0:
                    print(f"  No clickable link found, skipping course")
                    continue
                
                current_url = page.url
                print(f"  Clicking course link...")
                await potential_link.click()
                
                await page.wait_for_url(lambda url: url != current_url, timeout=15000)
                await page.wait_for_load_state("domcontentloaded")
                print(f"  Navigated to course: {page.url}")
                
                timeline_element = page.get_by_role("link", name="Timeline").first
                await timeline_element.wait_for(state="visible", timeout=15000)
                
                if await timeline_element.count() > 0:
                    print(f"  Found Timeline button, clicking...")
                    await timeline_element.click()
                    await page.wait_for_load_state("domcontentloaded")
                    
                    print(f"  Successfully navigated to Timeline: {page.url}")
                    
                    # Start scraping timeline content
                    content_count = await scrape_timeline_content(page, crawler, session_id)
                    print(f"  Completed course {course_index + 1}: {content_count} items processed")
                else:
                    print(f"  No Timeline button found, skipping course")
                
                if course_index < len(course_elements) - 1:
                    print(f"  Returning to homepage for next course...")
                    await page.goto(HOMEPAGE_URL)
                    await page.wait_for_load_state("domcontentloaded")
                    await page.locator("d2l-enrollment-card").first.wait_for(state="visible", timeout=15000)
                    course_elements = await page.locator("d2l-enrollment-card").all()
                    print(f"  Back at homepage, ready for next course")
            
            print(f"\nAll courses completed! Processed {len(course_elements)} courses total.")
            return page

        # Set the navigation hook
        crawler.crawler_strategy.set_hook("after_goto", after_goto) # pyright: ignore[reportAttributeAccessIssue]
        
        # Navigate to homepage to trigger the course discovery and timeline scraping
        config = CrawlerRunConfig(
            session_id=session_id,
            cache_mode=CacheMode.BYPASS,
            wait_until="domcontentloaded",
            page_timeout=20000,
            excluded_selector="#readspeaker_button_1",
            markdown_generator=DefaultMarkdownGenerator()
        )
        
        print(f"ARUN CALL: Starting timeline scraping at {HOMEPAGE_URL}")
        await crawler.arun(HOMEPAGE_URL, config=config)
        
        print(f"\nCompleted timeline scraping!")
        
    finally:
        await crawler.close()
        print("Crawler closed")

        # Clean up downloads directory
        if os.path.exists(downloads_path):
            try:
                shutil.rmtree(downloads_path)
                print("Cleaned up downloads directory")
            except Exception as e:
                print(f"Could not clean up downloads directory: {e}")

if __name__ == "__main__":
    asyncio.run(main())