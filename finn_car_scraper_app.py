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

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more details
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinnCarScraper:
    def __init__(self, base_url, attributes_to_extract, progress_callback=None):
        """
        Initialize the scraper with the base URL and attributes to extract.
        
        Args:
            base_url (str): The Finn.no URL with applied filters
            attributes_to_extract (list): List of attributes to extract
            progress_callback (function): Optional callback for progress updates
        """
        self.base_url = base_url  # Use the provided URL, not a hardcoded one
        self.attributes_to_extract = attributes_to_extract
        self.progress_callback = progress_callback
        self.data = []
        logger.info(f"Initializing scraper with URL: {self.base_url}")
        logger.info(f"Attributes to extract: {self.attributes_to_extract}")
        self.setup_driver()

    def setup_driver(self):
        """Set up the Chrome WebDriver with appropriate options."""
        logger.info("Setting up Chrome WebDriver...")
        options = webdriver.ChromeOptions()
        options.add_argument('--headless=new')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-extensions')
        options.add_argument('--enable-unsafe-swiftshader')
        options.add_argument('--window-size=1920,1080')  # Set window size
        
        try:
            logger.debug("Attempting to initialize ChromeDriver with WebDriverManager...")
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
            logger.info(f"Navigating to base URL: {self.base_url}")
            self.driver.get(self.base_url)
            time.sleep(5)  # Increased wait time
            
            # Log page source for debugging
            logger.debug(f"Page title: {self.driver.title}")
            logger.debug("Current URL: " + self.driver.current_url)
            
            # Look for the pagination info
            logger.info("Looking for pagination elements...")
            pagination = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "button[aria-label*='Side']"
            )
            logger.debug(f"Found {len(pagination)} pagination elements")
            
            if pagination:
                # Get the last page number
                last_page = pagination[-1].get_attribute('aria-label')
                logger.debug(f"Last pagination button aria-label: {last_page}")
                if last_page:
                    # Extract the number from "Side X av Y"
                    total = last_page.split(' av ')[-1]
                    logger.info(f"Total pages detected: {total}")
                    return int(total)
            
            logger.info("No pagination found or only one page detected")
            return 1
        except Exception as e:
            logger.error(f"Error getting total pages: {e}")
            logger.debug("Exception details:", exc_info=True)
            return 1

    def extract_listing_data(self, listing_url):
        """Extract data from an individual listing page."""
        try:
            logger.info(f"Navigating to listing: {listing_url}")
            self.driver.get(listing_url)
            time.sleep(3)  # Increased wait time
            
            # Save the page source for debugging
            page_id = listing_url.split('/')[-1]
            with open(f"listing_{page_id}.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            logger.debug(f"Saved page source to listing_{page_id}.html")
            
            # Using more generic waits and selectors
            try:
                logger.debug("Waiting for page content to load...")
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((
                        By.TAG_NAME, "h1"  # Wait for any h1 tag
                    ))
                )
            except Exception as wait_error:
                logger.warning(f"Wait timeout, but continuing: {wait_error}")
            
            logger.debug("Parsing page with BeautifulSoup")
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            listing_data = {}
            
            # Extract title - more generic approach
            logger.debug("Extracting title...")
            title_elem = soup.find('h1')  # Any h1 tag
            if title_elem:
                listing_data['title'] = title_elem.text.strip()
                logger.debug(f"Title found: {listing_data['title']}")
            else:
                listing_data['title'] = 'N/A'
                logger.warning("Title element not found")
            
            # Extract price - try different selectors
            logger.debug("Extracting price...")
            # Try different possible price selectors
            price_selectors = [
                ('span', {'data-testid': 'object-price'}),
                ('div', {'class': 'price'}),
                ('span', {'class': 'u-t1'}),
                ('p', {'class': 'u-t1'})
            ]
            
            price_elem = None
            for tag, attrs in price_selectors:
                price_elem = soup.find(tag, attrs)
                if price_elem:
                    logger.debug(f"Price found with selector: {tag}, {attrs}")
                    break
                    
            price_text = price_elem.text.strip() if price_elem else 'N/A'
            listing_data['price'] = price_text
            if price_elem:
                logger.debug(f"Price found: {listing_data['price']}")
            else:
                logger.warning("Price element not found with any selector")
            
            # Extract attributes from details section - try multiple selectors
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
                    logger.debug(f"Details section found with selector: {tag}, {attrs}")
                    break
            
            if details_section:
                # Handle DL elements
                if details_section.name == 'dl':
                    terms = details_section.find_all('dt')
                    values = details_section.find_all('dd')
                    logger.debug(
                        f"Found {len(terms)} term elements and {len(values)} value elements"
                    )
                    
                    # Log all found terms for debugging
                    all_terms = [term.text.strip() for term in terms]
                    logger.debug(f"All terms found: {all_terms}")
                    
                    for term, value in zip(terms, values):
                        key = term.text.strip()
                        val = value.text.strip()
                        logger.debug(f"Found attribute: {key} = {val}")
                        
                        if key in self.attributes_to_extract:
                            listing_data[key] = val
                            logger.info(f"Extracted attribute: {key} = {val}")
                
                # Handle TABLE elements
                elif details_section.name == 'table':
                    rows = details_section.find_all('tr')
                    logger.debug(f"Found {len(rows)} table rows")
                    
                    for row in rows:
                        cells = row.find_all(['th', 'td'])
                        if len(cells) >= 2:
                            key = cells[0].text.strip()
                            val = cells[1].text.strip()
                            logger.debug(f"Found attribute: {key} = {val}")
                            
                            if key in self.attributes_to_extract:
                                listing_data[key] = val
                                logger.info(f"Extracted attribute: {key} = {val}")
                
                # Handle DIV/panels
                else:
                    # Look for any key-value pairs in div structure
                    key_elems = details_section.find_all(
                        ['dt', 'th', 'strong', 'b']
                    )
                    for key_elem in key_elems:
                        key = key_elem.text.strip()
                        # Try to find value in the next sibling or parent's next child
                        val_elem = key_elem.find_next(
                            ['dd', 'td', 'span', 'div']
                        )
                        if val_elem:
                            val = val_elem.text.strip()
                            logger.debug(f"Found attribute: {key} = {val}")
                            
                            if key in self.attributes_to_extract:
                                listing_data[key] = val
                                logger.info(f"Extracted attribute: {key} = {val}")
            else:
                logger.warning("Details section not found with any selector")
                
                # Last resort: scan the entire page for attribute names
                logger.debug("Attempting to scan page for attributes...")
                for attr in self.attributes_to_extract:
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
                                logger.info(
                                    f"Found attribute via text scan: {attr} = {val}"
                                )
            
            # Add URL to the data
            listing_data['url'] = listing_url
            
            logger.info(f"Extracted data: {listing_data}")
            return listing_data
            
        except Exception as e:
            logger.error(f"Error extracting data from listing {listing_url}: {e}")
            logger.debug("Exception details:", exc_info=True)
            return None

    def scrape_listings(self, limit=5):
        """
        Scrape car listings from the search results.
        
        Args:
            limit (int): Maximum number of listings to scrape (for testing)
        """
        try:
            total_pages = self.get_total_pages()
            logger.info(f"Found {total_pages} pages to scrape")
            logger.info(f"Limiting to {limit} listings for testing phase")
            
            listings_processed = 0
            
            for page in range(1, total_pages + 1):
                if listings_processed >= limit:
                    logger.info(f"Reached limit of {limit} listings. Stopping.")
                    break
                    
                page_url = f"{self.base_url}&page={page}"
                logger.info(f"Scraping page {page}/{total_pages}: {page_url}")
                self.driver.get(page_url)
                time.sleep(3)  # Increased wait time
                
                # Report progress if callback is provided
                if self.progress_callback:
                    progress = min(listings_processed / limit, 0.5)
                    message = f"Scanning page {page}/{total_pages}"
                    self.progress_callback(progress, 1.0, message)
                
                # Find all listing links
                logger.debug("Looking for car ad links...")
                listing_elements = self.driver.find_elements(
                    By.CSS_SELECTOR,
                    "a[data-testid='car-ad-link']"
                )
                logger.info(
                    f"Found {len(listing_elements)} listing elements on page {page}"
                )
                
                # If no elements found, try alternative selectors
                if not listing_elements:
                    logger.debug(
                        "No car-ad-link elements found, trying alternative selectors..."
                    )
                    # Try to find any link inside the results that might be a car listing
                    listing_elements = self.driver.find_elements(
                        By.CSS_SELECTOR,
                        "article.sf-search-ad a"  # Alternative selector
                    )
                    logger.debug(
                        f"Found {len(listing_elements)} listings with alternative selector"
                    )
                
                # If still no elements, try one more selector
                if not listing_elements:
                    logger.debug("Trying one more selector...")
                    listing_elements = self.driver.find_elements(
                        By.CSS_SELECTOR,
                        "a.sf-search-ad-link"  # Another alternative
                    )
                    logger.debug(
                        f"Found {len(listing_elements)} listings with second alternative"
                    )
                
                # One more try
                if not listing_elements:
                    logger.debug("Last attempt to find links...")
                    # Save page source for debugging
                    with open(
                        f"page_{page}_source.html", "w", encoding="utf-8"
                    ) as f:
                        f.write(self.driver.page_source)
                    logger.debug(
                        f"Saved page source to page_{page}_source.html"
                    )
                    
                    # Get all links on the page
                    listing_elements = self.driver.find_elements(
                        By.TAG_NAME, "a"
                    )
                    logger.debug(
                        f"Found {len(listing_elements)} total links on page"
                    )
                
                listing_urls = [
                    elem.get_attribute('href')
                    for elem in listing_elements
                    if elem.get_attribute('href') and 'finn.no' in elem.get_attribute('href')
                ]
                logger.info(f"Extracted {len(listing_urls)} listing URLs")
                
                for i, url in enumerate(listing_urls):
                    if listings_processed >= limit:
                        logger.info(f"Reached limit of {limit} listings. Stopping.")
                        break
                        
                    logger.info(
                        f"Scraping listing {listings_processed+1}/{limit}: {url}"
                    )
                    
                    # Update progress
                    if self.progress_callback:
                        base_progress = 0.5  # First half was for page scanning
                        listing_progress = (listings_processed / limit) * 0.5
                        total_progress = base_progress + listing_progress
                        message = f"Scraping listing {listings_processed+1}/{limit}"
                        self.progress_callback(total_progress, 1.0, message)
                    
                    listing_data = self.extract_listing_data(url)
                    if listing_data:
                        listing_data['url'] = url
                        self.data.append(listing_data)
                        listings_processed += 1
                        logger.info(
                            f"Added listing to dataset, current count: "
                            f"{listings_processed}/{limit}"
                        )
                    else:
                        logger.warning(f"Failed to extract data from {url}")
                
                logger.info(f"Completed page {page}/{total_pages}")
                
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            logger.debug("Exception details:", exc_info=True)
        finally:
            logger.info("Closing WebDriver")
            self.driver.quit()
            
            # Final progress update
            if self.progress_callback:
                self.progress_callback(1.0, 1.0, "Scraping completed")

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
        if any(price_term in key.lower() for price_term in ['pris', 'verdi']):
            cleaned = cleaned.replace('kr', '').replace(' ', '')
            cleaned = cleaned.replace(',', '').replace('.-', '').replace('.', '')
            
        # Distance cleaning - remove "km"
        elif any(dist_term in key.lower() for dist_term in ['kilometer', 'km']):
            cleaned = cleaned.replace('km', '').replace(' ', '')
            
        # Power cleaning - remove "hk" (horsepower)
        elif any(power_term in key.lower() for power_term in ['hk', 'effekt']):
            cleaned = cleaned.replace('hk', '').replace(' ', '')
            
        # Year cleaning - extract just the year if there's more info
        elif any(year_term in key.lower() for year_term in ['år', 'modell']):
            # Extract just the first 4-digit number if it exists
            import re
            year_match = re.search(r'\b\d{4}\b', cleaned)
            if year_match:
                cleaned = year_match.group(0)
                
        # Date cleaning - standardize format
        elif 'registrert' in key.lower():
            # Keep the date but standardize format if possible
            import re
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
            if col in self.attributes_to_extract or col == 'price':
                df[col] = df[col].apply(
                    lambda x: self.clean_value(x, col) if isinstance(x, str) else x
                )
        
        # Log data summary
        logger.info(f"Total listings collected: {len(df)}")
        if not df.empty:
            logger.info(f"Columns in dataset: {df.columns.tolist()}")
            for col in df.columns:
                non_na_count = df[col].count()
                logger.info(
                    f"Column '{col}': {non_na_count}/{len(df)} non-NA values"
                )
        else:
            logger.warning("No data collected! Empty DataFrame.")
        
        df.to_csv(final_filename, index=False, encoding='utf-8-sig')
        logger.info(f"Data saved to {final_filename}")
        return df

def main():
    # Example usage - Using a Volvo URL
    base_url = (
        "https://www.finn.no/mobility/search/car?model=1.775.2000447"
        "&price_from=190000&price_to=280000&registration_class=1"
    )  # Volvo search
    
    attributes_to_extract = [
        "Modellår",
        "Kilometerstand",
        "1. gang registrert",
        "Totalpris"
    ]
    
    logger.info("Starting the FinnCarScraper")
    scraper = FinnCarScraper(base_url, attributes_to_extract)
    scraper.scrape_listings()
    df = scraper.save_to_csv()
    print("\nScraping completed! Data preview:")
    print(df.head())
    logger.info("Script execution completed")

if __name__ == "__main__":
    main() 