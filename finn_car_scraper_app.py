import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging based on environment variable
log_level = logging.DEBUG if os.getenv('DEBUG_MODE', 'False').lower() == 'true' else logging.INFO
log_file = os.getenv('LOG_FILE', 'scraper_debug.log')
enable_file_logging = os.getenv('ENABLE_FILE_LOGGING', 'False').lower() == 'true'

# Set up handlers
handlers = [logging.StreamHandler()]
if enable_file_logging:
    handlers.append(logging.FileHandler(log_file))

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

class FinnCarScraper:
    def __init__(self, base_url, attributes_to_extract, progress_callback=None,
                 listing_callback=None, processed_callback=None):
        """
        Initialize the scraper with the base URL and attributes to extract.
        
        Args:
            base_url (str): The Finn.no URL with applied filters
            attributes_to_extract (list): List of attributes to extract
            progress_callback (function): Optional callback for progress updates
            listing_callback (function): Optional callback for page and listing counts
            processed_callback (function): Optional callback when a listing is processed
        """
        self.base_url = base_url
        self.attributes_to_extract = attributes_to_extract
        self.progress_callback = progress_callback
        self.listing_callback = listing_callback
        self.processed_callback = processed_callback
        self.data = []
        
        # Define price-related terms for better detection
        self.price_terms = [
            'pris', 'price', 'totalpris', 'total pris', 'sum', 
            'kjøpesum', 'kjøpspris', 'kostnad', 'cost', 'total'
        ]
        
        # Check if any price attribute is in the attributes to extract
        self.extract_price = any(
            any(price_term in attr.lower() for price_term in self.price_terms)
            for attr in self.attributes_to_extract
        )
        
        # Get configuration from environment variables
        self.request_delay = float(os.getenv('REQUEST_DELAY', '3'))
        self.debug_mode = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
        
        logger.info(f"Initializing scraper with URL: {self.base_url}")
        if self.debug_mode:
            logger.debug(f"Attributes: {self.attributes_to_extract}")
            logger.debug(f"Extract price: {self.extract_price}")
            logger.debug(f"Request delay: {self.request_delay} seconds")
        
        self.setup_driver()

    def setup_driver(self):
        """Set up the Chrome WebDriver with appropriate options."""
        logger.info("Setting up Chrome WebDriver...")
        options = webdriver.ChromeOptions()
        
        # Use environment variables for Chrome options
        headless = os.getenv('HEADLESS_MODE', 'True').lower() == 'true'
        if headless:
            options.add_argument('--headless=new')
            
        # Add other Chrome arguments from environment
        chrome_args = os.getenv('CHROME_ARGS', '')
        if chrome_args:
            for arg in chrome_args.split(','):
                if arg.strip():
                    options.add_argument(arg.strip())
        else:
            # Default arguments if not specified in environment
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-extensions')
            
        options.add_argument('--window-size=1920,1080')
        
        try:
            if self.debug_mode:
                logger.debug("Initializing ChromeDriver...")
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            logger.info("ChromeDriver successfully initialized")
        except Exception as e:
            logger.error(f"Error setting up ChromeDriver: {e}")
            logger.info("Trying alternative setup method...")
            try:
                self.driver = webdriver.Chrome(options=options)
                logger.info("Alternative ChromeDriver setup successful")
            except Exception as e2:
                logger.error(f"Alternative setup also failed: {e2}")
                raise Exception(
                    "Could not initialize Chrome WebDriver. "
                    "Please ensure Chrome is installed and up to date."
                )
        
        self.driver.implicitly_wait(10)

    def get_total_pages(self):
        """Get the total number of pages to scrape."""
        try:
            logger.info("Navigating to base URL")
            self.driver.get(self.base_url)
            time.sleep(self.request_delay)  # Use configured delay
            
            # Log page source for debugging
            if self.debug_mode:
                logger.debug(f"Page title: {self.driver.title}")
                logger.debug("Current URL: " + self.driver.current_url)
            
            # Look for the pagination info
            if self.debug_mode:
                logger.debug("Looking for pagination elements...")
            pagination = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "button[aria-label*='Side']"
            )
            
            if pagination:
                # Get the last page number
                last_page = pagination[-1].get_attribute('aria-label')
                if last_page:
                    # Extract the number from "Side X av Y"
                    total = last_page.split(' av ')[-1]
                    logger.info(f"Total pages detected: {total}")
                    return int(total)
            
            logger.info("No pagination found or only one page detected")
            return 1
        except Exception as e:
            logger.error(f"Error getting total pages: {e}")
            if self.debug_mode:
                logger.debug("Exception details:", exc_info=True)
            return 1

    def extract_price_from_page(self, soup):
        """
        Extract price using multiple methods for better reliability.
        Returns a dictionary of {price_type: price_value} for different price types found.
        """
        prices = {}
        
        # Method 1: Try different direct CSS selectors for price elements
        price_selectors = [
            ('span', {'data-testid': 'object-price'}),
            ('span', {'class': 'price'}),
            ('div', {'class': 'price'}),
            ('span', {'class': 'u-t1'}),
            ('p', {'class': 'u-t1'}),
            ('div', {'class': 'u-mt32'}),
            ('span', {'class': 'amount'})
        ]
        
        for tag, attrs in price_selectors:
            price_elem = soup.find(tag, attrs)
            if price_elem:
                price_text = price_elem.text.strip()
                if price_text and any(s in price_text.lower() for s in ['kr', 'nok', ',-']):
                    if self.debug_mode:
                        logger.debug(f"Main price found: {price_text}")
                    prices['Pris'] = price_text
                    break
        
        # Method 2: Look for price info in key-value lists in the page
        # Find all DL, TABLE, and UL elements that might contain price information
        for container in soup.find_all(['dl', 'table', 'ul', 'div']):
            # For DL elements
            if container.name == 'dl':
                terms = container.find_all('dt')
                values = container.find_all('dd')
                
                for term, value in zip(terms, values):
                    term_text = term.text.strip().lower()
                    
                    # Check if this is a price-related term
                    if any(price_term in term_text for price_term in self.price_terms):
                        value_text = value.text.strip()
                        if value_text and self._looks_like_price(value_text):
                            if 'total' in term_text or 'sum' in term_text:
                                prices['Totalpris'] = value_text
                            else:
                                prices[term.text.strip()] = value_text
            
            # For TABLE elements            
            elif container.name == 'table':
                for row in container.find_all('tr'):
                    cells = row.find_all(['th', 'td'])
                    if len(cells) >= 2:
                        header = cells[0].text.strip().lower()
                        if any(price_term in header for price_term in self.price_terms):
                            value = cells[1].text.strip()
                            if self._looks_like_price(value):
                                if 'total' in header or 'sum' in header:
                                    prices['Totalpris'] = value
                                else:
                                    prices[cells[0].text.strip()] = value
            
            # For general DIV containers
            elif container.name == 'div':
                # Look for headings or strong text followed by price-like text
                headings = container.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b'])
                for heading in headings:
                    heading_text = heading.text.strip().lower()
                    if any(price_term in heading_text for price_term in self.price_terms):
                        # Look at the next element or sibling for the value
                        next_elem = heading.find_next()
                        if next_elem:
                            value = next_elem.text.strip()
                            if self._looks_like_price(value):
                                if 'total' in heading_text or 'sum' in heading_text:
                                    prices['Totalpris'] = value
                                else:
                                    prices[heading.text.strip()] = value
        
        # Method 3: General scan for price patterns in the page
        if not prices:
            # Try to find any text that looks like a price (kr + numbers)
            price_pattern = re.compile(r'([\d\s,.]+)(\s*kr|\s*,-|\s*NOK)', re.IGNORECASE)
            
            # Look in paragraphs and spans first
            for elem in soup.find_all(['p', 'span', 'div']):
                text = elem.text.strip()
                match = price_pattern.search(text)
                if match:
                    price_text = match.group(0)
                    # Try to determine what kind of price this is
                    container_text = elem.parent.text.lower() if elem.parent else text.lower()
                    if any(term in container_text for term in ['total', 'sum', 'omk']):
                        prices['Totalpris'] = price_text
                    elif not prices:  # If we haven't found any price yet, take this one
                        prices['Pris'] = price_text
        
        if self.debug_mode:
            logger.debug(f"All prices found: {prices}")
            
        return prices

    def _looks_like_price(self, text):
        """Check if a text looks like a price (contains numbers and currency symbols)."""
        text = text.lower()
        return ('kr' in text or 'nok' in text or ',-' in text) and any(c.isdigit() for c in text)

    def extract_listing_data(self, listing_url):
        """Extract data from an individual listing page."""
        try:
            if self.debug_mode:
                logger.debug(f"Extracting data from: {listing_url}")
            else:
                logger.info("Scraping listing...")
            
            self.driver.get(listing_url)
            time.sleep(self.request_delay)  # Use configured delay
            
            # Save the page source for debugging
            if self.debug_mode:
                page_id = listing_url.split('/')[-1]
                with open(f"listing_{page_id}.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                logger.debug(f"Saved page source to file: listing_{page_id}.html")
            
            # Using more generic waits and selectors
            try:
                if self.debug_mode:
                    logger.debug("Waiting for page content to load...")
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((
                        By.TAG_NAME, "h1"  # Wait for any h1 tag
                    ))
                )
            except Exception as wait_error:
                if self.debug_mode:
                    logger.warning(f"Wait timeout: {wait_error}")
            
            if self.debug_mode:
                logger.debug("Parsing page with BeautifulSoup")
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            listing_data = {'page': 1}  # Initialize with page 1 by default
            
            # Extract title - more generic approach
            if self.debug_mode:
                logger.debug("Extracting title...")
            title_elem = soup.find('h1')  # Any h1 tag
            if title_elem:
                listing_data['title'] = title_elem.text.strip()
                if self.debug_mode:
                    logger.debug(f"Title found: {listing_data['title']}")
            else:
                listing_data['title'] = 'N/A'
                if self.debug_mode:
                    logger.warning("Title element not found")
            
            # Extract price only if it's requested
            if self.extract_price:
                if self.debug_mode:
                    logger.debug("Extracting price information...")
                
                # Get all prices found on the page
                prices = self.extract_price_from_page(soup)
                
                # Map the extracted prices to the requested attributes
                for attr in self.attributes_to_extract:
                    attr_lower = attr.lower()
                    # Check if this attribute is price-related
                    if any(price_term in attr_lower for price_term in self.price_terms):
                        # First try exact match
                        if attr in prices:
                            listing_data[attr] = prices[attr]
                        # Then try matching based on price type
                        elif 'total' in attr_lower and 'Totalpris' in prices:
                            listing_data[attr] = prices['Totalpris']
                        elif attr_lower == 'pris' and 'Pris' in prices:
                            listing_data[attr] = prices['Pris']
                        # If no specific match, use any price found
                        elif prices:
                            listing_data[attr] = next(iter(prices.values()))
                        else:
                            listing_data[attr] = 'N/A'
                            
                        if self.debug_mode:
                            logger.debug(f"Price for {attr}: {listing_data[attr]}")
            
            # Extract attributes from details section - try multiple selectors
            if self.debug_mode:
                logger.debug("Extracting attributes from details section...")
            details_selectors = [
                ('dl', {'data-testid': 'object-key-details'}),
                ('dl', {'class': 'definition-list'}),
                ('table', {'class': 'table'}),
                ('div', {'class': 'panel'})
            ]
            
            details_section = None
            for tag, attrs in details_selectors:
                details_section = soup.find(tag, attrs)
                if details_section:
                    if self.debug_mode:
                        logger.debug(f"Details found with: {tag}, {attrs}")
                    break
            
            if details_section:
                # Handle DL elements
                if details_section.name == 'dl':
                    terms = details_section.find_all('dt')
                    values = details_section.find_all('dd')
                    
                    # Log all found terms for debugging
                    if self.debug_mode:
                        all_terms = [term.text.strip() for term in terms]
                        logger.debug(f"All terms found: {all_terms}")
                    
                    for term, value in zip(terms, values):
                        key = term.text.strip()
                        val = value.text.strip()
                        
                        if key in self.attributes_to_extract:
                            listing_data[key] = val
                            if self.debug_mode:
                                logger.debug(f"Extracted: {key} = {val}")
                
                # Handle TABLE elements
                elif details_section.name == 'table':
                    rows = details_section.find_all('tr')
                    
                    for row in rows:
                        cells = row.find_all(['th', 'td'])
                        if len(cells) >= 2:
                            key = cells[0].text.strip()
                            val = cells[1].text.strip()
                            
                            if key in self.attributes_to_extract:
                                listing_data[key] = val
                                if self.debug_mode:
                                    logger.debug(f"Extracted: {key} = {val}")
                
                # Handle DIV/panels
                else:
                    # Look for any key-value pairs in div structure
                    key_elems = details_section.find_all(
                        ['dt', 'th', 'strong', 'b']
                    )
                    for key_elem in key_elems:
                        key = key_elem.text.strip()
                        # Find value in the next sibling or parent's next child
                        val_elem = key_elem.find_next(
                            ['dd', 'td', 'span', 'div']
                        )
                        if val_elem:
                            val = val_elem.text.strip()
                            
                            if key in self.attributes_to_extract:
                                listing_data[key] = val
                                if self.debug_mode:
                                    logger.debug(f"Extracted: {key} = {val}")
            else:
                if self.debug_mode:
                    logger.warning("Details section not found")
                
                # Last resort: scan the entire page for attribute names
                if self.debug_mode:
                    logger.debug("Scanning page for attributes...")
                for attr in self.attributes_to_extract:
                    # Skip if we already found this attribute
                    if attr in listing_data:
                        continue
                        
                    # Find any element containing the attribute name
                    attr_elem = soup.find(
                        string=lambda text: text and attr in text
                    )
                    if attr_elem:
                        # Try to find the value in nearby elements
                        parent = attr_elem.parent
                        if parent:
                            # Look for siblings or next elements
                            next_elem = parent.find_next()
                            if next_elem:
                                val = next_elem.text.strip()
                                listing_data[attr] = val
                                if self.debug_mode:
                                    logger.debug(f"Found via scan: {attr} = {val}")
            
            # Add URL to the data
            listing_data['url'] = listing_url
            
            # Call callback with the listing data if provided
            if self.processed_callback:
                self.processed_callback(listing_data)
            
            # Ensure all requested attributes exist in the data
            for attr in self.attributes_to_extract:
                if attr not in listing_data:
                    listing_data[attr] = 'N/A'
            
            return listing_data
            
        except Exception as e:
            logger.error(f"Error extracting data from listing: {e}")
            if self.debug_mode:
                logger.debug("Exception details:", exc_info=True)
            return None

    def scrape_listings(self, limit=5):
        """
        Scrape car listings from the search results across multiple pages.
        
        Args:
            limit (int): Maximum number of listings to scrape
        """
        try:
            # Check if a page parameter already exists in the URL
            import re
            page_in_url = re.search(r'[?&]page=\d+', self.base_url)
            
            # Create base URL without page parameter if it exists
            if page_in_url:
                base_url_without_page = re.sub(r'([?&])page=\d+[&]?', '\\1', self.base_url)
                # Fix trailing '&' or '?' if needed
                if base_url_without_page.endswith('&') or base_url_without_page.endswith('?'):
                    base_url_without_page = base_url_without_page[:-1]
                self.base_url = base_url_without_page
                logger.info(f"Removed existing page parameter. Base URL: {self.base_url}")
            
            # Always start with page 1 to determine total pages
            if '?' in self.base_url:
                page1_url = f"{self.base_url}&page=1"
            else:
                page1_url = f"{self.base_url}?page=1"
                
            logger.info(f"Starting with URL: {page1_url}")
            self.driver.get(page1_url)
            time.sleep(self.request_delay)
            
            # Get total number of pages
            total_pages = self.get_pagination_info()
            logger.info(f"Found {total_pages} pages to scrape")
            
            # Each page typically has up to 50 listings
            listings_per_page = 50
            
            # Calculate how many pages we need to visit to get the requested number of listings
            if limit < float('inf'):
                pages_to_scrape = min(total_pages, (limit + listings_per_page - 1) // listings_per_page)
                logger.info(f"Will scrape up to {limit} listings (about {pages_to_scrape} pages)")
            else:
                pages_to_scrape = total_pages
                logger.info(f"Will scrape all {total_pages} pages")
            
            # Make sure we scrape at least 2 pages if limit requires it and total_pages > 1
            if limit > listings_per_page and total_pages > 1 and pages_to_scrape < 2:
                pages_to_scrape = 2
                logger.info(f"Adjusted to scrape at least {pages_to_scrape} pages for the requested limit")
            
            listings_processed = 0
            
            # Iterate through pages
            for page in range(1, pages_to_scrape + 1):
                # Exit if we've reached the limit
                if listings_processed >= limit:
                    logger.info(f"Reached limit of {limit} listings. Stopping.")
                    break
                
                # Construct URL with page parameter
                if '?' in self.base_url:
                    page_url = f"{self.base_url}&page={page}"
                else:
                    page_url = f"{self.base_url}?page={page}"
                    
                logger.info(f"Navigating to page {page}/{pages_to_scrape} at URL: {page_url}")
                
                # Try loading the page with retries
                max_attempts = 3
                for attempt in range(1, max_attempts + 1):
                    try:
                        # Navigate to the page
                        self.driver.get(page_url)
                        time.sleep(self.request_delay)
                        
                        # Verify we loaded a results page by checking for listings
                        test_elements = self.driver.find_elements(
                            By.CSS_SELECTOR,
                            "a[data-testid='car-ad-link'], article.sf-search-ad, div.ads__unit"
                        )
                        
                        if test_elements:
                            logger.debug(f"Successfully loaded page {page} (attempt {attempt})")
                            break
                        else:
                            logger.warning(f"Page {page} loaded but no listings found (attempt {attempt})")
                            if attempt < max_attempts:
                                logger.info(f"Retrying page {page} in 2 seconds...")
                                time.sleep(2)
                    except Exception as e:
                        logger.warning(f"Error loading page {page} (attempt {attempt}): {e}")
                        if attempt < max_attempts:
                            logger.info(f"Retrying page {page} in 2 seconds...")
                            time.sleep(2)
                        else:
                            logger.error(f"Failed to load page {page} after {max_attempts} attempts")
                            # Skip to next page if this one fails repeatedly
                            continue
                
                # Current page URL for debugging
                logger.debug(f"Current URL after navigation: {self.driver.current_url}")
                
                # Report progress if callback is provided
                if self.progress_callback:
                    progress = min(page / pages_to_scrape * 0.5, 0.5)
                    message = f"Scanning page {page}/{pages_to_scrape}"
                    self.progress_callback(progress, 1.0, message)
                
                # Save debug info if in debug mode
                if self.debug_mode:
                    with open(f"page_{page}_source.html", "w", encoding="utf-8") as f:
                        f.write(self.driver.page_source)
                    logger.debug(f"Saved page {page} source to file")
                
                # Find all listing links on this page
                listing_urls = self.extract_listing_links_from_page()
                logger.info(f"Found {len(listing_urls)} listing URLs on page {page}")
                
                # Check if we found any listings on this page
                if not listing_urls:
                    logger.warning(f"No listings found on page {page}, moving to next page")
                    continue
                
                # Update listing statistics callback if provided
                if self.listing_callback:
                    self.listing_callback(page, pages_to_scrape, len(listing_urls))
                
                # Process listings from this page
                listings_to_process = listing_urls
                
                # Limit the number of listings to process if needed
                remaining = limit - listings_processed
                if remaining < len(listings_to_process):
                    listings_to_process = listings_to_process[:remaining]
                    
                logger.info(f"Processing {len(listings_to_process)} listings from page {page}")
                
                # Process each listing from this page
                for i, url in enumerate(listings_to_process):
                    if self.debug_mode:
                        logger.debug(f"Scraping listing {i+1}/{len(listings_to_process)} from page {page}")
                    
                    # Update progress
                    if self.progress_callback:
                        base_progress = 0.5  # First half was for page scanning
                        listings_progress = ((page - 1) * listings_per_page + i) / (pages_to_scrape * listings_per_page) * 0.5
                        total_progress = base_progress + listings_progress
                        message = f"Scraping listing {i+1}/{len(listings_to_process)} from page {page}"
                        self.progress_callback(total_progress, 1.0, message)
                    
                    # Extract data from this listing
                    listing_data = self.extract_listing_data(url)
                    if listing_data:
                        listing_data['url'] = url
                        # Make sure the page number is set (will override the default)
                        listing_data['page'] = page
                        self.data.append(listing_data)
                        listings_processed += 1
                        logger.info(f"Processed {listings_processed}/{limit} total listings")
                        
                        # Call callback with the listing data if provided
                        if self.processed_callback:
                            self.processed_callback(listing_data)
                    else:
                        if self.debug_mode:
                            logger.warning(f"Failed to extract data from listing {url}")
                    
                    # Optional small delay between listings
                    time.sleep(max(0.5, self.request_delay / 3))
                
                logger.info(f"Completed page {page}/{pages_to_scrape}, {listings_processed}/{limit} listings processed")
                
                # Add a small delay between pages
                if page < pages_to_scrape:
                    logger.debug(f"Waiting {self.request_delay} seconds before loading next page...")
                    time.sleep(self.request_delay)
            
            logger.info(f"All listing processing completed. Collected {listings_processed} listings in total.")
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            if self.debug_mode:
                logger.debug("Exception details:", exc_info=True)
        finally:
            logger.info("Closing WebDriver")
            self.driver.quit()
            
            # Final progress update
            if self.progress_callback:
                self.progress_callback(1.0, 1.0, "Scraping completed")

    def get_pagination_info(self):
        """Extract pagination information from the current page."""
        try:
            # Look for pagination elements
            logger.debug("Looking for pagination elements...")
            
            # Try to find the pagination container first
            pagination_container = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "nav[aria-label='Pagination'], .pagination, .pagination-container"
            )
            
            if pagination_container:
                logger.debug("Found pagination container")
            
            # Method 1: Look for pagination buttons (most common)
            pagination_elements = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "button[aria-label*='Side'], [data-testid*='pagination'], .pagination button, .sf-pagination-item, a[href*='page=']"
            )
            
            if pagination_elements:
                logger.debug(f"Found {len(pagination_elements)} pagination elements")
                
                # Get the last button's aria-label (e.g., "Side 5 av 10")
                last_page_button = pagination_elements[-1]
                aria_label = last_page_button.get_attribute('aria-label')
                logger.debug(f"Last pagination element aria-label: {aria_label}")
                
                if aria_label and " av " in aria_label:
                    # Extract the number after "av"
                    total_pages = int(aria_label.split(" av ")[-1])
                    logger.info(f"Total pages from pagination: {total_pages}")
                    return total_pages
                
                # Check for text content of the last page button
                last_button_text = last_page_button.text.strip()
                logger.debug(f"Last pagination element text: {last_button_text}")
                
                if last_button_text and last_button_text.isdigit():
                    total_pages = int(last_button_text)
                    logger.info(f"Total pages from last pagination button: {total_pages}")
                    return total_pages
            
            # Method 2: Look for "Next" button or link
            next_buttons = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "[aria-label*='Neste'], [aria-label*='Next'], a[href*='page='][rel='next'], .next, .pagination-next"
            )
            
            if next_buttons:
                # If there's a next button, there are at least 2 pages
                logger.info("Found 'Next' button - at least 2 pages")
                return 2
            
            # Method 3: Try to find any text indicating total number of hits
            total_hits_elements = self.driver.find_elements(
                By.CSS_SELECTOR, 
                ".u-strong, .total-count, .search-results-count, .search-hits"
            )
            
            for element in total_hits_elements:
                hits_text = element.text
                logger.debug(f"Found potential hit count: {hits_text}")
                
                # Try to extract a number
                import re
                hits_match = re.search(r'(\d+)', hits_text)
                if hits_match:
                    total_hits = int(hits_match.group(1))
                    
                    # If the number seems to be a count of total listings
                    if total_hits > 20:
                        # Calculate pages (50 results per page)
                        estimated_pages = (total_hits + 49) // 50
                        logger.info(f"Estimated {estimated_pages} pages from {total_hits} hits")
                        return max(1, estimated_pages)
            
            # Method 4: Check the URL if it already indicates we're on page 1
            current_url = self.driver.current_url
            logger.debug(f"Current URL: {current_url}")
            
            if "page=1" in current_url:
                # Try to find links to page 2
                page2_links = self.driver.find_elements(
                    By.CSS_SELECTOR,
                    f"a[href*='page=2'], button[aria-label*='Side 2']"
                )
                
                if page2_links:
                    logger.info("Found link to page 2 - at least 2 pages")
                    return 2
            
            # Method 5: Try to count the number of car listings to see if we need pagination
            listing_count = len(self.driver.find_elements(
                By.CSS_SELECTOR,
                "a[data-testid='car-ad-link'], article.sf-search-ad"
            ))
            
            logger.debug(f"Found {listing_count} listings on current page")
            
            if listing_count >= 45:
                # If we have many listings on one page, assume there's at least one more page
                logger.info(f"Found {listing_count} listings on page, assuming at least 2 pages")
                return 2
            
            # Default: Assume at least 1 page
            logger.info("No pagination indicators found - assuming 1 page")
            return 1
            
        except Exception as e:
            logger.warning(f"Error getting pagination info: {e}")
            if self.debug_mode:
                logger.debug("Exception details:", exc_info=True)
            # Default to 2 pages to be safe
            logger.info("Assuming at least 2 pages due to error")
            return 2
            
    def extract_listing_links_from_page(self):
        """Extract all listing links from the current page using multiple strategies."""
        listing_urls = []
        
        try:
            logger.debug("Looking for car listing links on this page...")
            
            # Strategy 1: Look for standard car ad links
            logger.debug("Trying data-testid='car-ad-link' selector...")
            listing_elements = self.driver.find_elements(
                By.CSS_SELECTOR,
                "a[data-testid='car-ad-link']"
            )
            
            if listing_elements:
                logger.debug(f"Found {len(listing_elements)} elements with data-testid='car-ad-link'")
                for elem in listing_elements:
                    href = elem.get_attribute('href')
                    if href and 'finn.no' in href:
                        listing_urls.append(href)
                
                logger.debug(f"Added {len(listing_urls)} links from car-ad-link selector")
            else:
                logger.debug("No listing elements found with data-testid='car-ad-link'")
            
            # Strategy 2: If no elements found with specific selector or too few links, try article links
            if len(listing_urls) < 20:
                logger.debug("Trying article.sf-search-ad selector...")
                
                # Try links inside articles
                article_links = self.driver.find_elements(
                    By.CSS_SELECTOR,
                    "article.sf-search-ad a, div.ads__unit a"
                )
                
                init_count = len(listing_urls)
                for elem in article_links:
                    href = elem.get_attribute('href')
                    if href and 'finn.no' in href and href not in listing_urls:
                        if '/car/' in href or '/car?' in href or 'mobility' in href:
                            listing_urls.append(href)
                
                logger.debug(f"Added {len(listing_urls) - init_count} links from article selector")
            
            # Strategy 3: Check for other common listing containers 
            if len(listing_urls) < 20:
                logger.debug("Trying other listing container selectors...")
                
                container_selectors = [
                    ".search-results-list a", 
                    ".ads a", 
                    ".finn-listing a", 
                    ".listing a",
                    "[data-automation-id='searchResult'] a",
                    "[data-testid='searchResults'] a"
                ]
                
                init_count = len(listing_urls)
                for selector in container_selectors:
                    container_links = self.driver.find_elements(
                        By.CSS_SELECTOR, selector
                    )
                    
                    for elem in container_links:
                        href = elem.get_attribute('href')
                        if href and 'finn.no' in href and href not in listing_urls:
                            if '/car/' in href or '/car?' in href or 'mobility' in href:
                                listing_urls.append(href)
                
                logger.debug(f"Added {len(listing_urls) - init_count} links from container selectors")
            
            # Strategy 4: Look for any link that seems to be a car listing
            if len(listing_urls) < 20:
                logger.debug("Looking for any car listing links...")
                
                all_links = self.driver.find_elements(By.TAG_NAME, "a")
                logger.debug(f"Checking {len(all_links)} total links on the page")
                
                init_count = len(listing_urls)
                for elem in all_links:
                    href = elem.get_attribute('href')
                    if href and 'finn.no' in href and href not in listing_urls:
                        # Check if the link looks like a car listing
                        if ('/car/' in href or 
                            '/car?' in href or 
                            'mobility' in href or 
                            'bap/forsale' in href):
                            listing_urls.append(href)
                
                logger.debug(f"Added {len(listing_urls) - init_count} links from generic link search")
            
            # Filter to only include actual car listings
            logger.debug(f"Found {len(listing_urls)} total potential car listing links")
            
            # Use a more precise regex to identify car listing URLs
            car_listing_pattern = re.compile(r'finn\.no/.*(?:car/|mobility/|bap/forsale)')
            car_listings = [
                url for url in listing_urls 
                if car_listing_pattern.search(url)
            ]
            
            logger.debug(f"After filtering, found {len(car_listings)} valid car listing links")
            
            # If filtering removed too many, fall back to original list
            if len(car_listings) < 10 and len(listing_urls) >= 20:
                logger.warning("Too many links filtered out, using original list")
                return listing_urls
            
            if not car_listings and not listing_urls:
                # Last resort: save the page for debugging
                if self.debug_mode:
                    with open("failed_page_source.html", "w", encoding="utf-8") as f:
                        f.write(self.driver.page_source)
                    logger.debug("No listings found - saved page source to failed_page_source.html")
                logger.warning("Could not find any car listing links on this page!")
            
            return car_listings if car_listings else listing_urls
            
        except Exception as e:
            logger.error(f"Error extracting listing links: {e}")
            if self.debug_mode:
                logger.debug("Exception details:", exc_info=True)
            return []

    def get_total_pages_from_url(self, url):
        """Get the total number of pages to scrape from a specific URL."""
        try:
            logger.info(f"Navigating to {url}")
            self.driver.get(url)
            time.sleep(self.request_delay)  # Use configured delay
            
            return self.get_pagination_info()
            
        except Exception as e:
            logger.error(f"Error getting total pages: {e}")
            if self.debug_mode:
                logger.debug("Exception details:", exc_info=True)
            return 1

    def clean_value(self, value, key):
        """
        Clean extracted values by removing units and formatting.
        
        Args:
            value (str): The value to clean
            key (str): The attribute key (to determine cleaning method)
            
        Returns:
            str: The cleaned value
        """
        if value == 'N/A':
            return value
            
        # Remove whitespace
        cleaned = value.strip()
        
        # Price cleaning - remove "kr", spaces, and commas
        if any(price_term in key.lower() for price_term in self.price_terms):
            # Remove all non-numeric characters except digits
            cleaned = re.sub(r'[^\d]', '', cleaned)
            # If it's empty after cleaning, return original
            if not cleaned:
                return value
            
        # Distance cleaning - remove "km"
        elif any(dist_term in key.lower() for dist_term in ['kilometer', 'km']):
            cleaned = cleaned.replace('km', '').replace(' ', '')
            
        # Power cleaning - remove "hk" (horsepower)
        elif any(power_term in key.lower() for power_term in ['hk', 'effekt']):
            cleaned = cleaned.replace('hk', '').replace(' ', '')
            
        # Year cleaning - extract just the year if there's more info
        elif any(year_term in key.lower() for year_term in ['år', 'modell']):
            # Extract just the first 4-digit number if it exists
            year_match = re.search(r'\b\d{4}\b', cleaned)
            if year_match:
                cleaned = year_match.group(0)
                
        # Date cleaning - standardize format
        elif 'registrert' in key.lower():
            # Keep the date but standardize format if possible
            date_match = re.search(
                r'\b\d{1,2}[.-/]\d{1,2}[.-/]\d{2,4}\b', cleaned
            )
            if date_match:
                cleaned = date_match.group(0)
        
        return cleaned
        
    def save_to_csv(self, filename='car_listings.csv'):
        """
        Save the scraped data to a CSV file.
        If file exists, adds timestamp to filename to avoid overwriting.
        """
        # Get base name and extension
        base, ext = os.path.splitext(filename)
        
        # Try the original filename first
        final_filename = filename
        counter = 1
        
        # If file exists, add timestamp and counter if needed
        while os.path.exists(final_filename):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_filename = f"{base}_{timestamp}_{counter}{ext}"
            counter += 1
        
        df = pd.DataFrame(self.data)
        
        # Clean all data values before saving
        logger.info("Cleaning data values (removing units like 'kr' and 'km')")
        for col in df.columns:
            if col in self.attributes_to_extract:
                df[col] = df[col].apply(
                    lambda x: self.clean_value(x, col) 
                    if isinstance(x, str) else x
                )
        
        # Log data summary
        logger.info(f"Total listings collected: {len(df)}")
        if not df.empty and self.debug_mode:
            logger.debug(f"Columns in dataset: {df.columns.tolist()}")
            for col in df.columns:
                non_na_count = df[col].count()
                logger.debug(f"Column '{col}': {non_na_count}/{len(df)} values")
        elif not df.empty:
            logger.info(f"Dataset contains {len(df.columns)} columns")
        else:
            logger.warning("No data collected! Empty DataFrame.")
        
        df.to_csv(final_filename, index=False, encoding='utf-8-sig')
        logger.info(f"Data saved to {final_filename}")
        return df


def main():
    # Load values from environment variables
    url = os.getenv(
        'DEFAULT_SEARCH_URL',
        "https://www.finn.no/mobility/search/car?model=1.775.2000447"
        "&price_from=190000&price_to=280000&registration_class=1"
    )
    
    attributes_to_extract = os.getenv(
        'DEFAULT_ATTRIBUTES', 
        "Modellår,Kilometerstand,'1. gang registrert',Totalpris"
    ).split(',')
    
    limit = int(os.getenv('DEFAULT_LIMIT', '60'))  # Default limit changed to 60
    
    logger.info("Starting the FinnCarScraper")
    scraper = FinnCarScraper(url, attributes_to_extract)
    scraper.scrape_listings(limit=limit)
    df = scraper.save_to_csv()
    print("\nScraping completed! Data preview:")
    print(df.head())
    logger.info("Script execution completed")


if __name__ == "__main__":
    main() 