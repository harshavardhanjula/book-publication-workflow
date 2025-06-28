"""Web scraper for extracting book content and taking screenshots."""
import asyncio
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from loguru import logger
from playwright.async_api import async_playwright


class ChapterScraper:
    """Handles web scraping of book chapters and screenshot capture."""

    def __init__(self, output_dir: str = "data"):
        """Initialize the scraper with output directory.
        
        Args:
            output_dir: Directory to save scraped content and screenshots
        """
        self.output_dir = Path(output_dir)
        self.screenshots_dir = self.output_dir / "screenshots"
        self.content_dir = self.output_dir / "content"
        
        # Create necessary directories
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.content_dir.mkdir(parents=True, exist_ok=True)

    async def _take_screenshot(self, page, url: str) -> str:
        """Take a full-page screenshot and save it.
        
        Args:
            page: Playwright page object
            url: URL being scraped (used for naming)
            
        Returns:
            Path to the saved screenshot
        """
        # Generate a filename from the URL
        parsed_url = urlparse(url)
        filename = f"{parsed_url.path.strip('/').replace('/', '_')}.png"
        screenshot_path = self.screenshots_dir / filename
        
        # Take full page screenshot
        await page.screenshot(path=str(screenshot_path), full_page=True)
        logger.info(f"Screenshot saved to {screenshot_path}")
        return str(screenshot_path)

    async def _extract_content(self, html: str) -> Dict[str, str]:
        """Extract content from the HTML using BeautifulSoup.
        
        Args:
            html: HTML content to parse
            
        Returns:
            Dictionary containing extracted content elements
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract the main content - adjust selectors based on the actual page structure
        title = soup.find('h1', {'id': 'firstHeading'})
        content = soup.find('div', {'class': 'mw-parser-output'})
        
        # Clean up the content
        if content:
            # Remove edit links and other unwanted elements
            for element in content.select('.mw-editsection, .mw-empty-elt'):
                element.decompose()
            
            # Get text with proper spacing
            content_text = '\n\n'.join(
                p.get_text().strip() 
                for p in content.find_all(['p', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if p.get_text().strip()
            )
        else:
            content_text = ""
        
        return {
            'title': title.get_text().strip() if title else "",
            'content': content_text,
            'html': str(content) if content else ""
        }

    async def scrape_chapter(self, url: str) -> Dict:
        """Scrape a chapter from the given URL.
        
        Args:
            url: URL of the chapter to scrape
            
        Returns:
            Dictionary containing scraped data and metadata
        """
        logger.info(f"Scraping chapter from {url}")
        
        async with async_playwright() as p:
            # Launch browser in headless mode
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            
            try:
                # Create a new page
                page = await context.new_page()
                
                # Set a reasonable timeout
                page.set_default_timeout(30000)  # 30 seconds
                
                # Navigate to the URL
                response = await page.goto(url, wait_until='networkidle')
                
                if not response or not response.ok:
                    raise Exception(f"Failed to load page: {response.status if response else 'No response'}")
                
                # Wait for the content to be visible
                await page.wait_for_selector('.mw-parser-output', state='visible')
                
                # Get page content
                content = await page.content()
                
                # Extract text content
                extracted = await self._extract_content(content)
                
                # Take screenshot
                screenshot_path = await self._take_screenshot(page, url)
                
                # Save content to file
                filename = urlparse(url).path.strip('/').replace('/', '_')
                content_path = self.content_dir / f"{filename}.txt"
                content_path.write_text(extracted['content'], encoding='utf-8')
                
                logger.info(f"Successfully scraped content from {url}")
                
                return {
                    'url': url,
                    'title': extracted['title'],
                    'content': extracted['content'],
                    'html': extracted['html'],
                    'screenshot_path': screenshot_path,
                    'content_path': str(content_path)
                }
                
            except Exception as e:
                logger.error(f"Error scraping {url}: {str(e)}")
                raise
                
            finally:
                # Clean up
                await context.close()
                await browser.close()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        scraper = ChapterScraper()
        url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
        result = await scraper.scrape_chapter(url)
        print(f"Scraped: {result['title']}")
        print(f"Content length: {len(result['content'])} characters")
        print(f"Screenshot saved to: {result['screenshot_path']}")
    
    asyncio.run(main())
