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
from selenium.common.exceptions import NoSuchElementException

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

    def get_total_pages(self, base_url):
        """Get the total number of pages for the search results."""
        try:
            self.logger.info("Determining total number of pages...")
            
            # Load the first page to determine total pages
            self.driver.get(base_url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article.sf-search-ad"))
            )
            time.sleep(2)  # Add a slight delay to ensure page loads completely
            
            # Look for pagination information
            try:
                # Try to find the pagination element that contains the total page count
                pagination_text = self.driver.find_element(
                    By.CSS_SELECTOR, 
                    ".pagination__info"
                ).text
                
                # Extract the total pages from text like "Page 1 of 24"
                match = re.search(r'av (\d+)', pagination_text)
                if match:
                    total_pages = int(match.group(1))
                    self.logger.info(f"Found {total_pages} pages to scrape")
                    return total_pages
                else:
                    # If the format is different, fall back to counting page links
                    page_links = self.driver.find_elements(
                        By.CSS_SELECTOR, 
                        ".pagination__page-link"
                    )
                    if page_links:
                        highest_page = max([
                            int(link.text) for link in page_links 
                            if link.text.isdigit()
                        ] or [1])
                        self.logger.info(f"Found {highest_page} pages to scrape")
                        return highest_page
            except NoSuchElementException:
                # If pagination element not found, check if we have results but only 1 page
                listings = self.driver.find_elements(
                    By.CSS_SELECTOR, "article.sf-search-ad"
                )
                if listings:
                    self.logger.info("Only 1 page of results found")
                    return 1
                else:
                    self.logger.warning("No results found on the page")
                    return 0
                
            # Default to 1 page if we can't determine
            self.logger.info("Could not determine page count, defaulting to 1")
            return 1
            
        except Exception as e:
            self.logger.error(f"Error getting total pages: {str(e)}")
            return 1

    def navigate_to_page(self, base_url, page_num):
        """Navigate to a specific page number."""
        try:
            if page_num <= 1:
                # First page has no page parameter
                page_url = base_url
            else:
                # Check if the base URL already has query parameters
                if '?' in base_url:
                    page_url = f"{base_url}&page={page_num}"
                else:
                    page_url = f"{base_url}?page={page_num}"
                
            self.logger.info(f"Navigating to page {page_num}: {page_url}")
            self.driver.get(page_url)
            
            # Wait for the page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article.sf-search-ad"))
            )
            
            # Explicit sleep to avoid rate limiting
            time.sleep(self.request_delay)
            
            # Verify we're on the correct page
            try:
                pagination_text = self.driver.find_element(
                    By.CSS_SELECTOR, 
                    ".pagination__info"
                ).text
                
                # Log the actual page we're on for debugging
                self.logger.info(f"Pagination text: {pagination_text}")
                
                current_page_match = re.search(r'Side (\d+)', pagination_text)
                if current_page_match:
                    current_page = int(current_page_match.group(1))
                    if current_page != page_num:
                        self.logger.warning(
                            f"Expected to be on page {page_num} but found page {current_page}"
                        )
            except Exception as page_verify_error:
                self.logger.warning(
                    f"Could not verify current page number: {str(page_verify_error)}"
                )
                
            return True
        except Exception as e:
            self.logger.error(f"Error navigating to page {page_num}: {str(e)}")
            return False

    def extract_listings_from_current_page(self):
        """Extract all listing URLs from the current page."""
        try:
            # Wait for listings to appear
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article.sf-search-ad"))
            )
            
            # Find all listing elements
            listing_elements = self.driver.find_elements(
                By.CSS_SELECTOR, "article.sf-search-ad"
            )
            
            # Count listings found
            self.logger.info(f"Found {len(listing_elements)} listings on current page")
            
            # Update total count in real-time
            self.total_listings_found += len(listing_elements)
            if self.listing_callback:
                self.listing_callback(self.current_page, self.total_listings_found)
            
            # Extract URLs
            listing_urls = []
            for element in listing_elements:
                try:
                    # Find the link within the listing
                    link_element = element.find_element(By.CSS_SELECTOR, "a.sf-search-ad__link")
                    url = link_element.get_attribute("href")
                    if url:
                        listing_urls.append(url)
                except Exception as e:
                    self.logger.warning(f"Error extracting URL from listing: {str(e)}")
                    continue
            
            self.logger.info(f"Extracted {len(listing_urls)} URLs from current page")
            return listing_urls
            
        except Exception as e:
            self.logger.error(f"Error extracting listings: {str(e)}")
            return []

    def scrape_listings(self, base_url, limit=None):
        """Scrape all car listings from search results."""
        self.logger.info(f"Starting to scrape listings from {base_url}")
        all_listing_urls = []
        self.total_listings_found = 0
        self.current_page = 0
        
        try:
            # Get total number of pages
            total_pages = self.get_total_pages(base_url)
            if total_pages == 0:
                self.logger.warning("No pages found to scrape")
                return []
                
            # Determine how many pages to scrape
            pages_to_scrape = total_pages
            if limit and limit > 0:
                # Estimate number of pages needed based on ~25 listings per page
                estimated_pages = (limit + 24) // 25
                pages_to_scrape = min(total_pages, estimated_pages)
                self.logger.info(
                    f"Limiting to approximately {limit} listings "
                    f"(~{pages_to_scrape} pages out of {total_pages})"
                )
            
            # Scrape each page
            for page_num in range(1, pages_to_scrape + 1):
                self.current_page = page_num
                self.logger.info(f"Scraping page {page_num}/{pages_to_scrape}")
                
                if page_num == 1:
                    # We're already on the first page from get_total_pages
                    # Just extract the listings
                    page_urls = self.extract_listings_from_current_page()
                else:
                    # Navigate to the next page
                    success = self.navigate_to_page(base_url, page_num)
                    if not success:
                        self.logger.warning(f"Failed to navigate to page {page_num}, stopping pagination")
                        break
                        
                    # Extract listings from this page
                    page_urls = self.extract_listings_from_current_page()
                
                # Add to our collection
                all_listing_urls.extend(page_urls)
                
                # Check if we've reached the limit
                if limit and len(all_listing_urls) >= limit:
                    self.logger.info(f"Reached limit of {limit} listings, stopping pagination")
                    all_listing_urls = all_listing_urls[:limit]
                    break
                
                # Don't try to go to the next page if we're on the last page
                if page_num == total_pages:
                    self.logger.info("Reached the last page")
                    break
            
            self.logger.info(f"Successfully scraped {len(all_listing_urls)} listing URLs")
            return all_listing_urls
            
        except Exception as e:
            self.logger.error(f"Error during pagination: {str(e)}")
            # Return any URLs we managed to collect before the error
            self.logger.info(f"Returning {len(all_listing_urls)} URLs collected before error")
            return all_listing_urls

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
                logger.debug(f"Navigating to listing: {listing_url}")
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
            listing_data = {}
            
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


def run_scraper(url, attributes_to_extract, limit=None):
    """
    Run the Finn.no car scraper to collect data from car listings.
    
    Args:
        url (str): The base URL for Finn.no car search results
        attributes_to_extract (list): List of attributes to extract from each listing
        limit (int, optional): Maximum number of listings to scrape
        
    Returns:
        pd.DataFrame: DataFrame containing the extracted car data
    """
    logger.info("Starting the FinnCarScraper")
    scraper = FinnCarScraper(url, attributes_to_extract)
    scraper.scrape_listings(url, limit=limit)
    df = scraper.save_to_csv()
    print("\nScraping completed! Data preview:")
    print(df.head())
    return df


if __name__ == "__main__":
    main() 