import requests
from bs4 import BeautifulSoup
import re
import argparse
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FinnPageCounter')

def add_page_param(url, page_num):
    """Add or update the page parameter in a URL."""
    if 'page=' in url:
        # Replace existing page parameter
        return re.sub(r'page=\d+', f'page={page_num}', url)
    elif '?' in url:
        # URL already has parameters, add page parameter
        return f"{url}&page={page_num}"
    else:
        # URL has no parameters yet
        return f"{url}?page={page_num}"

def is_empty_result_page(html_content):
    """
    Check if a page contains the empty results message.
    
    Args:
        html_content (str): HTML content of the page
        
    Returns:
        bool: True if the page is empty, False otherwise
    """
    empty_indicators = [
        "Ingen treff",
        "No results",
        "Her var det tomt, gitt!",
        "Vi finner ikke produktet du leter etter"
    ]
    
    for indicator in empty_indicators:
        if indicator in html_content:
            return True
    
    return False

def get_total_pages(url, delay=1.0, max_pages=100):
    """
    Determine the total number of pages for a given Finn.no search.
    
    Args:
        url (str): The base Finn.no search URL
        delay (float): Delay between requests in seconds
        max_pages (int): Maximum number of pages to check before stopping
        
    Returns:
        int: Total number of pages found
    """
    logger.info(f"Starting page count for: {url}")
    
    # Headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    # First check if the initial URL works
    try:
        logger.info("Checking page 1...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        if is_empty_result_page(response.text):
            logger.info("No results found on page 1")
            return 0
            
        # Try to extract total number from page if available
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the total count or page indicators
        result_count_elem = soup.select_one("div.u-pl16.u-pr16 > span")
        if result_count_elem:
            count_text = result_count_elem.text.strip()
            logger.info(f"Found count text: {count_text}")
            
            # Try to extract total pages from the count information
            if "treff" in count_text:
                try:
                    # Extract the number of results
                    num_results = int(re.search(r'(\d+)', count_text).group(1))
                    # Finn typically shows 25 results per page
                    results_per_page = 25
                    estimated_pages = (num_results + results_per_page - 1) // results_per_page
                    logger.info(f"Estimated {estimated_pages} pages based on {num_results} results")
                    
                    # We'll still verify by checking pages
                except (AttributeError, ValueError):
                    logger.warning("Could not parse result count")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error accessing page 1: {e}")
        return 0
        
    # Binary search approach to find the last valid page
    left = 1
    right = max_pages
    last_valid_page = 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if mid == 1:
            # We already checked page 1
            left = mid + 1
            continue
            
        page_url = add_page_param(url, mid)
        logger.info(f"Checking page {mid}...")
        
        try:
            # Add delay to avoid overloading the server
            time.sleep(delay)
            
            response = requests.get(page_url, headers=headers)
            
            if response.status_code == 200 and not is_empty_result_page(response.text):
                # This page exists and has results
                last_valid_page = mid
                left = mid + 1
            else:
                # This page doesn't exist or has no results
                right = mid - 1
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error accessing page {mid}: {e}")
            right = mid - 1
    
    logger.info(f"Total pages found: {last_valid_page}")
    return last_valid_page

def sequential_page_check(url, delay=1.0, max_pages=100):
    """
    Check pages sequentially until finding a page with no results.
    
    Args:
        url (str): The base Finn.no search URL
        delay (float): Delay between requests in seconds
        max_pages (int): Maximum number of pages to check before stopping
        
    Returns:
        int: Total number of pages found
    """
    logger.info(f"Starting sequential page check for: {url}")
    
    # Headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    # First check page 1
    try:
        logger.info("Checking page 1...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        if is_empty_result_page(response.text):
            logger.info("No results found on page 1")
            return 0
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error accessing page 1: {e}")
        return 0
    
    # Check subsequent pages
    page = 2
    while page <= max_pages:
        page_url = add_page_param(url, page)
        logger.info(f"Checking page {page}...")
        
        try:
            # Add delay to avoid overloading the server
            time.sleep(delay)
            
            response = requests.get(page_url, headers=headers)
            
            if response.status_code != 200 or is_empty_result_page(response.text):
                # This page doesn't exist or has no results
                logger.info(f"No results or error on page {page}")
                break
                
            logger.info(f"Valid results found on page {page}")
            page += 1
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error accessing page {page}: {e}")
            break
    
    total_pages = page - 1
    logger.info(f"Total pages found: {total_pages}")
    return total_pages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count pages in Finn.no listings")
    parser.add_argument("url", help="Finn.no search URL")
    parser.add_argument("--method", choices=["binary", "sequential"], default="binary",
                       help="Method to use for counting pages (binary or sequential)")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between requests in seconds")
    parser.add_argument("--max-pages", type=int, default=100,
                       help="Maximum number of pages to check")
    
    args = parser.parse_args()
    
    logger.info(f"URL: {args.url}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Delay: {args.delay}s")
    logger.info(f"Max pages: {args.max_pages}")
    
    if args.method == "binary":
        total_pages = get_total_pages(args.url, args.delay, args.max_pages)
    else:
        total_pages = sequential_page_check(args.url, args.delay, args.max_pages)
    
    print(f"\nRESULT: Found {total_pages} pages of results")
    
    if total_pages > 0:
        print("\nExample URLs:")
        print(f"Page 1: {args.url}")
        print(f"Page 2: {add_page_param(args.url, 2)}")
        if total_pages > 2:
            print(f"Last page: {add_page_param(args.url, total_pages)}") 