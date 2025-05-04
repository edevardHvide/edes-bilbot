import streamlit as st
import json
from finn_car_scraper_app import FinnCarScraper
import os
import tempfile
from dotenv import load_dotenv
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import FuncFormatter
from ai_analyzer import CarAIAnalyzer
import openai
import numpy as np
from scipy import stats
from page_counter import get_total_pages
import logging
from logging.handlers import RotatingFileHandler


# Set up logging
# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")

# Configure logger to write to both file and console
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# Create handlers
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
console_handler = logging.StreamHandler()

# Create formatters and add to handlers
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
file_formatter = logging.Formatter(log_format)
console_formatter = logging.Formatter(log_format)
file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Application started")

# Load environment variables
load_dotenv()

# Get configuration from environment variables
DEFAULT_SEARCH_URL = os.getenv(
    'DEFAULT_SEARCH_URL', 
    'https://www.finn.no/mobility/search/car?registration_class=1'
)
DEFAULT_LIMIT = int(os.getenv('DEFAULT_LIMIT', '5'))

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY', '')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT', '')

# For backward compatibility
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Set up OpenAI client if API key is available
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Function to format numbers with Norwegian style (spaces as thousands separator)
def format_with_spaces(num):
    """Format a number with spaces as thousands separator"""
    return f"{int(num):,}".replace(',', ' ')

# Easter egg comments for interesting cars
EASTER_EGGS = [
    "Daamn this one looks nice! 🔥",
    "Wow, what a find! 👀",
    "This one's a steal at that price! 💰",
    "Feit kjerre brosjan ✨",
    "Dream car alert! 🚨",
    "Someone's going to be lucky with this one! 🍀",
    "Bro den bilen er dum da! 🏎️",
    "That's one sweet ride! 🍭",
    "Would definitely turn heads! 👌",
    "This one has your name on it! 📝"
]

# Loading messages to display while waiting
LOADING_MESSAGES = [
    "Starting up the engines...",
    "Connecting to Finn.no...",
    "Warming up the scraper...",
    "Scanning for hidden gems...",
    "Analyzing the marketplace...",
    "Checking listings one by one...",
    "Almost ready to find you some cars..."
]

st.set_page_config(
    page_title="Ede's bilskrapebot",
    page_icon="🚗",
    layout="wide"
)

# Main heading
st.title("Welcome to Ede's bilskrapebot")

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    
    # Base URL input
    base_url = st.text_input(
        "Enter Finn.no search URL",
        value=DEFAULT_SEARCH_URL,
        placeholder="https://www.finn.no/mobility/search/car?...",
        help="Copy and paste a full URL from finn.no after applying filters"
    )
    
    # Attributes to extract
    st.subheader("Attributes to Extract")
    st.caption("Select the car attributes you want to extract")
    
    common_attributes = [
        "Modellår",
        "Totalpris",
        "Kilometerstand", 
        "1. gang registrert",
        "Girkasse",
        "Drivstoff",
        "Motor",
        "Effekt",
        "Hjuldrift"
    ]
    
    selected_attributes = []
    
    # Create checkboxes for common attributes
    # Special handling for "1. gang registrert" to display correctly
    for attr in common_attributes:
        # For "1. gang registrert", we need to handle it specially
        if attr == "1. gang registrert":
            display_name = "1\\. gang registrert"  # Escape the period for display
            if st.checkbox(display_name, value=True, key="first_reg_checkbox"):
                selected_attributes.append(attr)  # Still use original name for data extraction
        else:
            if st.checkbox(attr, value=True):
                selected_attributes.append(attr)
    
    # Custom attribute input
    custom_attr = st.text_input(
        "Add custom attribute", 
        placeholder="e.g., Toppfart"
    )
    if custom_attr:
        selected_attributes.append(custom_attr)
    
    # Scraping mode: limited or all cars
    scrape_mode = st.radio(
        "Scraping Mode",
        ["Limited number of cars", "All cars matching criteria"],
        index=0
    )
    
    # Limit number of listings if in limited mode
    if scrape_mode == "Limited number of cars":
        limit = st.number_input(
            "Number of listings to scrape", 
            min_value=1, 
            max_value=500, 
            value=DEFAULT_LIMIT
        )
    else:
        # When scraping all, we'll use infinity as the limit
        limit = float('inf')
        st.info("⚠️ Note: Scraping all cars may take a long time depending on the search criteria")
    
    # Run button
    run_scraper = st.button("Run Scraper", type="primary")
    
    # Add option to load test data
    col1, col2 = st.columns(2)
    with col1:
        load_test_data = st.button("🧪 Load Test Data", help="Load sample data for testing without running the scraper")
    with col2:
        save_as_test = st.checkbox("📥 Save results as test data", value=False, 
                                help="Save the scraped results as test data for future use")

# Function to load test data
def load_sample_data():
    """Load saved test data if available"""
    test_data_path = "test_data.pkl"
    if os.path.exists(test_data_path):
        try:
            return pd.read_pickle(test_data_path)
        except Exception as e:
            st.error(f"Error loading test data: {str(e)}")
    return None

# Function to save test data
def save_sample_data(df):
    """Save dataframe as test data"""
    test_data_path = "test_data.pkl"
    try:
        df.to_pickle(test_data_path)
        return True
    except Exception as e:
        st.error(f"Error saving test data: {str(e)}")
        return False

# Main content area
if load_test_data:
    # Load saved test data instead of running the scraper
    df = load_sample_data()
    
    if df is not None and not df.empty:
        st.success("✅ Test data loaded successfully!")
        
        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Data Table", "JSONderulo", "Analytics", "AI Analysis", "Download"]
        )
        
        # Convert data to dictionary for JSON serialization
        result_data = df.to_dict(orient='records')
        
        with tab1:
            st.dataframe(df, use_container_width=True)
            
        with tab2:
            st.json(result_data)
            
        with tab3:
            # Analytics tab content
            if "Totalpris" in df.columns and "Kilometerstand" in df.columns:
                # Existing analytics code
                st.subheader("Price vs. Mileage Analysis")
                # ... rest of the analytics code remains the same
                
        with tab4:
            # AI Analysis tab content
            analyzer = CarAIAnalyzer()
            # ... rest of the AI Analysis code remains the same
            
        with tab5:
            # Download tab content
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="finn_car_data.csv",
                mime="text/csv"
            )
            
            json_str = json.dumps(
                result_data, 
                ensure_ascii=False, 
                indent=2
            )
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="finn_car_data.json",
                mime="application/json"
            )
    else:
        st.error("❌ No test data found. Please run the scraper at least once and save the results.")
elif run_scraper:
    if not base_url:
        st.error("Please enter a valid Finn.no URL")
    elif not selected_attributes:
        st.error("Please select at least one attribute to extract")
    else:
        try:
            # Set up the scraping status displays
            progress_bar = st.progress(0)
            status_container = st.container()
            
            with status_container:
                status_text = st.empty()
                listing_info = st.empty()
                
                # Create a two-column layout for status information
                col1, col2 = st.columns(2)
                
                with col1:
                    total_listings_found = st.empty()
                    current_page = st.empty()
                
                with col2:
                    processed_count = st.empty()
                    easter_egg = st.empty()
            
            # Display dynamic loading messages while the system initializes
            for i, message in enumerate(LOADING_MESSAGES):
                status_text.info(f"⏳ {message}")
                # Short delay to create a typing effect
                time.sleep(0.5)
                # Update progress based on loading message step
                progress_value = min(0.3, (i + 1) / len(LOADING_MESSAGES))
                progress_bar.progress(progress_value)
            
            # Store interesting stats during scraping
            scraping_stats = {
                "total_listings_found": 0,
                "current_page": 0,
                "total_pages": 0,
                "processed": 0,
                "interesting_cars": 0,
                "last_update": time.time()
            }
            
            def progress_callback(progress, total, message):
                """Update UI with progress information"""
                # Update progress bar
                progress_bar.progress(progress)
                
                # Only update text if significant time has passed
                current_time = time.time()
                if current_time - scraping_stats["last_update"] > 0.3:
                    status_text.info(f"🔍 {message}")
                    scraping_stats["last_update"] = current_time
                
                # Show easter egg comment occasionally (about 20% of listings)
                if scraping_stats.get("new_listing") and random.random() < 0.1:
                    scraping_stats["interesting_cars"] += 1
                    easter_egg.markdown(f"✨ **{random.choice(EASTER_EGGS)}**")
                    
                # Reset new listing flag
                scraping_stats["new_listing"] = False
            
            # Custom callback to update listing status
            def listing_callback(page, total_pages, listings_found):
                scraping_stats["current_page"] = page
                scraping_stats["total_pages"] = total_pages
                scraping_stats["total_listings_found"] = listings_found
            
            def processed_callback(listing_data):
                scraping_stats["processed"] += 1
                scraping_stats["new_listing"] = True
                
                # Show listing title in the info area
                if listing_data and 'title' in listing_data:
                    # Format the page number for display
                    page_info = f"Page {listing_data.get('page', 1)}" if 'page' in listing_data else ""
                    listing_info.text(f"🔎 Processing: {listing_data['title']} - {page_info} - URL: {listing_data['url']}")
            
            # Initialize the scraper
            status_text.info("🚀 Initializing scraper...")
            scraper = FinnCarScraper(
                base_url, 
                selected_attributes, 
                progress_callback=progress_callback,
                listing_callback=listing_callback,
                processed_callback=processed_callback
            )
            
            # Run the scraper with the specified limit
            status_text.info(
                f"🚀 Starting to scrape {'all' if limit == float('inf') else limit} cars..."
            )
            
            # Check number of pages before scraping (informational only)
            try:
                # Set a short delay to avoid rate limiting
                page_delay = 0.5
                
                # Show a message about checking pages
                pages_info = st.empty()
                pages_info.info("📄 Checking number of pages in search results...")
                
                # Get the number of pages without blocking the interface too long
                num_pages = get_total_pages(base_url, delay=page_delay, max_pages=20)
                
                # Update the status with page information
                if num_pages > 0:
                    pages_info.info(f"📄 Found {num_pages} pages of search results (approximately {num_pages * 25} listings)")
                else:
                    pages_info.warning("⚠️ Could not determine the number of pages in search results")
            except Exception as e:
                # Just log the error but continue with scraping
                logger.warning(f"Could not determine number of pages: {str(e)}")
            
            # Implement batched processing to improve performance
            batch_size = 20  # Process 20 listings at a time
            
            # Initialize the result dataframe
            result_df = pd.DataFrame()
            
            # Calculate number of batches
            total_to_scrape = min(limit, float('inf'))
            if total_to_scrape == float('inf'):
                num_batches = "multiple"
                status_text.info(f"🔄 Processing in batches of {batch_size} listings")
            else:
                num_batches = (total_to_scrape + batch_size - 1) // batch_size
                status_text.info(f"🔄 Processing {total_to_scrape} listings in {num_batches} batches")
            
            # Setup temp directory for batch results
            batch_dir = os.path.join(tempfile.gettempdir(), "finn_batches")
            os.makedirs(batch_dir, exist_ok=True)
            
            remaining = total_to_scrape
            batch_num = 1
            page_offset = 0  # Start from the first page
            
            # Process in batches
            while remaining > 0:
                if remaining == float('inf'):
                    current_batch = batch_size
                else:
                    current_batch = min(batch_size, remaining)
                
                # Update status
                status_text.info(f"🔄 Processing batch {batch_num}: {current_batch} listings (starting from page {page_offset+1})...")
                
                # Clear the scraper's data array to avoid memory buildup
                if hasattr(scraper, 'data'):
                    scraper.data = []
                
                # Process this batch
                try:
                    # Use the page_offset to start from where we left off
                    scraper.scrape_listings(limit=current_batch, page_offset=page_offset)
                    
                    # Save batch results
                    if scraper.data:
                        batch_df = pd.DataFrame(scraper.data)
                        batch_file = os.path.join(batch_dir, f"batch_{batch_num}.csv")
                        batch_df.to_csv(batch_file, index=False)
                        
                        # Append to results
                        if result_df.empty:
                            result_df = batch_df
                        else:
                            result_df = pd.concat([result_df, batch_df], ignore_index=True)
                        
                        status_text.success(f"✅ Batch {batch_num} completed: Added {len(batch_df)} listings")
                        
                        # Calculate the next page offset based on where we stopped
                        # Each page has approximately 50 listings
                        listings_processed = len(batch_df)
                        pages_processed = (listings_processed + 49) // 50
                        page_offset += pages_processed
                    else:
                        status_text.warning(f"⚠️ Batch {batch_num} yielded no results")
                        # If no results, still increment page offset to try next page
                        page_offset += 1
                except Exception as batch_e:
                    status_text.error(f"❌ Error in batch {batch_num}: {str(batch_e)}")
                    # On error, try the next page
                    page_offset += 1
                
                # Update remaining count
                if remaining != float('inf'):
                    remaining -= current_batch
                
                # Increment batch counter
                batch_num += 1
                
                # If infinite, stop after a reasonable number of batches
                if remaining == float('inf') and batch_num > 10:
                    status_text.info("🛑 Reached maximum number of batches for unlimited scraping")
                    break
                
                # Add a delay between batches to let the system recover
                if remaining > 0:
                    status_text.info("⏳ Pausing between batches to optimize performance...")
                    time.sleep(3)  # Pause between batches
                    
                    # Close and reinitialize the driver to free up memory
                    if batch_num % 3 == 0:  # Every 3 batches
                        status_text.info("🔄 Refreshing browser session to optimize memory usage...")
                        try:
                            scraper.driver.quit()
                            time.sleep(1)
                            scraper.setup_driver()
                            time.sleep(1)
                        except Exception as driver_e:
                            status_text.warning(f"⚠️ Error refreshing driver: {str(driver_e)}")
            
            # Save combined results
            if not result_df.empty:
                # Save as temporary CSV
                temp_dir = tempfile.gettempdir()
                temp_csv = os.path.join(temp_dir, "finn_car_data.csv")
                result_df.to_csv(temp_csv, index=False)
                df = result_df
                
                status_text.success(f"✅ All batches processed! Total listings: {len(df)}")
            else:
                status_text.error("❌ No data was collected from any batch")
                df = pd.DataFrame()
                
            # Clean up batch files
            try:
                for file in os.listdir(batch_dir):
                    if file.startswith("batch_") and file.endswith(".csv"):
                        os.remove(os.path.join(batch_dir, file))
            except Exception as cleanup_e:
                st.warning(f"Error cleaning up batch files: {str(cleanup_e)}")
            
            # Save as test data if requested
            if save_as_test and not df.empty:
                if save_sample_data(df):
                    st.success("✅ Results saved as test data for future use!")
            
            # Display results
            status_text.success("✅ Scraping completed!")
            progress_bar.progress(1.0)
            listing_info.empty()  # Clear the listing info
            
            # Show final stats
            st.success(f"""
            ### Scraping Results
            - Total listings found: {scraping_stats['total_listings_found']}
            - Pages scanned: {scraping_stats['total_pages']}
            - Listings processed: {scraping_stats['processed']}
            """)
            
            # Convert data to dictionary for JSON serialization
            result_data = df.to_dict(orient='records')
            
            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["Data Table", "JSONderulo", "Analytics", "AI Analysis", "Download"]
            )
            
            with tab1:
                st.dataframe(df, use_container_width=True)
                
            with tab2:
                st.json(result_data)
                
            with tab3:
                if "Totalpris" in df.columns and "Kilometerstand" in df.columns:
                    st.subheader("Price vs. Mileage Analysis")
                    
                    # Prepare data for plotting
                    try:
                        # First check if we have data
                        if df.empty:
                            st.warning("No data available to create a plot.")
                            
                        else:
                            # Log data to help debug
                            st.write(f"Total rows: {len(df)}")
                            
                            # Try different approaches to convert the data to numeric
                            # First attempt - basic conversion with debug info
                            try:
                           
                                # Clean data - remove non-numeric characters
                                df['Totalpris_clean'] = df['Totalpris'].astype(str).str.replace(r'[^\d]', '', regex=True)
                                df['Kilometerstand_clean'] = df['Kilometerstand'].astype(str).str.replace(r'[^\d]', '', regex=True)
                                
                                # Convert to numeric, coerce errors to NaN
                                df['Totalpris_num'] = pd.to_numeric(df['Totalpris_clean'], errors='coerce')
                                df['Kilometerstand_num'] = pd.to_numeric(df['Kilometerstand_clean'], errors='coerce')
                                
                               
                                # Drop rows with NaN values
                                plot_df = df.dropna(subset=['Totalpris_num', 'Kilometerstand_num'])
                                st.write(f"Rows after dropping NaN: {len(plot_df)}")
                                
                                # Need at least 2 points for a meaningful plot
                                if len(plot_df) >= 2:
                                    # Create a simple scatter plot
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    # Basic scatter plot without year data for now
                                    scatter = ax.scatter(
                                        plot_df['Kilometerstand_num'], 
                                        plot_df['Totalpris_num'],
                                        alpha=0.7,
                                        s=80,
                                        color='royalblue'
                                    )
                                    
                                    # Add a linear regression line
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                                        plot_df['Kilometerstand_num'], 
                                        plot_df['Totalpris_num']
                                    )
                                    
                                    # Create the regression line
                                    x_values = plot_df['Kilometerstand_num']
                                    x_range = np.linspace(x_values.min(), x_values.max(), 100)
                                    y_regression = intercept + slope * x_range
                                    
                                    # Plot the regression line
                                    ax.plot(
                                        x_range, 
                                        y_regression, 
                                        color='red', 
                                        linestyle='--', 
                                        linewidth=2,
                                        label=f'y = {intercept:.0f} + {slope:.2f}x (r² = {r_value**2:.2f})'
                                    )
                                    
                                    # Calculate the average cost per kilometer
                                    avg_cost_per_km = -slope  # Negative because the slope is typically negative
                                    
                                    # Add a legend
                                    ax.legend(loc='upper right')
                                    
                                    # Add row indices as annotations for hovering
                                    # Reset index to get sequential numbers if there were any dropped rows
                                    plot_df = plot_df.reset_index()
                                    
                                    # Add text annotations showing row numbers
                                    for i, row in plot_df.iterrows():
                                        ax.annotate(f"#{i}", 
                                                   (row['Kilometerstand_num'], row['Totalpris_num']),
                                                   xytext=(5, 5),
                                                   textcoords='offset points',
                                                   fontsize=8,
                                                   alpha=0.7)
                                    
                                    # Set labels and title
                                    ax.set_xlabel('Mileage (km)', fontsize=12)
                                    ax.set_ylabel('Price (NOK)', fontsize=12)
                                    ax.set_title(
                                        'Car Price vs. Mileage', 
                                        fontsize=14, 
                                        fontweight='bold'
                                    )
                                    
                                    # Add grid for better readability
                                    ax.grid(True, linestyle='--', alpha=0.7)
                                    
                                    # Format y-axis with thousands separator
                                    def thousands_formatter(x, pos):
                                        return format_with_spaces(x)
                                        
                                    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
                                    
                                    # Improve layout
                                    plt.tight_layout()
                                    
                                    # Display the plot
                                    st.pyplot(fig)
                                    
                                    # Add summary statistics
                                    st.subheader("Summary Statistics")
                                    
                                    # Price statistics
                                    st.write("**Price (NOK)**")
                                    price_min = int(plot_df['Totalpris_num'].min())
                                    price_max = int(plot_df['Totalpris_num'].max())
                                    price_avg = int(plot_df['Totalpris_num'].mean())
                                    
                                    st.write(f"- Min: {format_with_spaces(price_min)}")
                                    st.write(f"- Max: {format_with_spaces(price_max)}")
                                    st.write(f"- Average: {format_with_spaces(price_avg)}")
                                    
                                    # Mileage statistics
                                    st.write("**Mileage (km)**")
                                    km_min = int(plot_df['Kilometerstand_num'].min())
                                    km_max = int(plot_df['Kilometerstand_num'].max())
                                    km_avg = int(plot_df['Kilometerstand_num'].mean())
                                    
                                    st.write(f"- Min: {format_with_spaces(km_min)}")
                                    st.write(f"- Max: {format_with_spaces(km_max)}")
                                    st.write(f"- Average: {format_with_spaces(km_avg)}")
                                    
                                    # Add explanation box
                                    st.info("""
                                    ### What This Plot Shows
                                    
                                    This scatter plot displays the relationship between car mileage and price.
                                    
                                    - Each blue dot represents a car listing
                                    - Cars with similar mileage but different prices may indicate quality differences
                                    - Generally, cars with higher mileage tend to be less expensive
                                    - Outliers (dots far from the cluster) may be worth investigating
                                    
                                    Use this visualization to spot potential good deals or overpriced listings.
                                    """)
                                    
                                    # Display outliers if present
                                    price_mean = plot_df['Totalpris_num'].mean()
                                    price_std = plot_df['Totalpris_num'].std()
                                    km_mean = plot_df['Kilometerstand_num'].mean()
                                    km_std = plot_df['Kilometerstand_num'].std()
                                    
                                    # Find potential outliers (2 standard deviations from mean)
                                    outliers = plot_df[
                                        (plot_df['Totalpris_num'] > price_mean + 2*price_std) |
                                        (plot_df['Totalpris_num'] < price_mean - 2*price_std) |
                                        (plot_df['Kilometerstand_num'] > km_mean + 2*km_std) |
                                        (plot_df['Kilometerstand_num'] < km_mean - 2*km_std)
                                    ]
                                    
                                    if not outliers.empty:
                                        st.subheader("Potential Outliers")
                                        st.write("These listings stand out from the general trend:")
                                        outlier_df = outliers[['title', 'Totalpris', 'Kilometerstand', 'url']]
                                        st.dataframe(outlier_df)
                                    
                                    # Add car depreciation analysis if model year data is available
                                    if "Modellår" in df.columns:
                                        st.markdown("---")
                                        st.subheader("Car Depreciation Analysis")
                                        st.write("Analyze how car prices depreciate over model years for similar car models.")
                                        
                                        # Create clean model year column
                                        try:
                                            # Clean Modellår data and convert to numeric
                                            df['Modellår_clean'] = df['Modellår'].astype(str).str.extract(r'(\d{4})')
                                            df['Modellår_num'] = pd.to_numeric(df['Modellår_clean'], errors='coerce')
                                            
                                            # Create a column for the car model (remove year and other details from title)
                                            # Extract just the make and model from the title (remove year and trim details)
                                            df['car_model'] = df['title'].str.split(' ').apply(
                                                lambda x: ' '.join(x[:2]) if len(x) > 1 else x[0] if len(x) > 0 else ''
                                            )
                                            
                                            # Get car models with multiple model years
                                            model_counts = df.groupby('car_model')['Modellår_num'].nunique()
                                            models_with_multiple_years = model_counts[model_counts >= 2].index.tolist()
                                            
                                            if models_with_multiple_years:
                                                # Let user select a car model to analyze
                                                selected_model = st.selectbox(
                                                    "Select a car model to analyze depreciation:",
                                                    options=models_with_multiple_years,
                                                    key="depreciation_model_selector"
                                                )
                                                
                                                # Filter data for selected model
                                                model_data = df[df['car_model'] == selected_model].copy()
                                                
                                                # Group by model year and calculate average price
                                                if not model_data.empty:
                                                    # Create a clean dataset for the analysis
                                                    model_analysis = model_data.dropna(subset=['Modellår_num', 'Totalpris_num'])
                                                    
                                                    if len(model_analysis) >= 2:
                                                        # Group by year and calculate statistics
                                                        year_stats = model_analysis.groupby('Modellår_num').agg({
                                                            'Totalpris_num': ['mean', 'median', 'count']
                                                        }).reset_index()
                                                        
                                                        # Flatten the multi-index columns
                                                        year_stats.columns = ['model_year', 'avg_price', 'median_price', 'count']
                                                        
                                                        # Display the results in a table
                                                        st.write(f"Depreciation analysis for {selected_model}:")
                                                        
                                                        # Format the prices for display
                                                        year_stats['avg_price_formatted'] = year_stats['avg_price'].apply(
                                                            lambda x: format_with_spaces(int(x))
                                                        )
                                                        year_stats['median_price_formatted'] = year_stats['median_price'].apply(
                                                            lambda x: format_with_spaces(int(x))
                                                        )
                                                        
                                                        # Create display table
                                                        display_cols = ['model_year', 'avg_price_formatted', 'median_price_formatted', 'count']
                                                        display_df = year_stats[display_cols].copy()
                                                        display_df.columns = ['Model Year', 'Average Price (NOK)', 'Median Price (NOK)', 'Number of Cars']
                                                        st.dataframe(display_df, use_container_width=True)
                                                        
                                                        # Calculate year-over-year depreciation
                                                        if len(year_stats) > 1:
                                                            # Sort by year
                                                            year_stats = year_stats.sort_values('model_year')
                                                            
                                                            # Calculate depreciation
                                                            year_stats['prev_year_price'] = year_stats['avg_price'].shift(-1)
                                                            year_stats['depreciation_amount'] = year_stats['avg_price'] - year_stats['prev_year_price']
                                                            year_stats['depreciation_pct'] = (year_stats['depreciation_amount'] / year_stats['avg_price']) * 100
                                                            
                                                            # Filter out rows with NaN depreciation (the newest year)
                                                            depreciation_data = year_stats.dropna(subset=['depreciation_amount'])
                                                            
                                                            if not depreciation_data.empty:
                                                                # Plot depreciation trend
                                                                fig, ax = plt.subplots(figsize=(10, 6))
                                                                
                                                                # Plot bars for depreciation percentage
                                                                bars = ax.bar(
                                                                    depreciation_data['model_year'].astype(str), 
                                                                    depreciation_data['depreciation_pct'],
                                                                    color='skyblue',
                                                                    alpha=0.7
                                                                )
                                                                
                                                                # Add value labels on the bars
                                                                for bar in bars:
                                                                    height = bar.get_height()
                                                                    ax.text(
                                                                        bar.get_x() + bar.get_width() / 2,
                                                                        height + 0.5,
                                                                        f"{height:.1f}%",
                                                                        ha='center',
                                                                        fontsize=9
                                                                    )
                                                                
                                                                # Add line for average depreciation
                                                                avg_depreciation = depreciation_data['depreciation_pct'].mean()
                                                                ax.axhline(
                                                                    avg_depreciation, 
                                                                    color='red', 
                                                                    linestyle='--', 
                                                                    label=f'Avg: {avg_depreciation:.1f}%'
                                                                )
                                                                
                                                                # Add labels and title
                                                                ax.set_xlabel('Model Year', fontsize=12)
                                                                ax.set_ylabel('Depreciation (%)', fontsize=12)
                                                                ax.set_title(
                                                                    f'Year-over-Year Depreciation: {selected_model}', 
                                                                    fontsize=14, 
                                                                    fontweight='bold'
                                                                )
                                                                
                                                                # Add grid and legend
                                                                ax.grid(True, linestyle='--', alpha=0.7)
                                                                ax.legend()
                                                                
                                                                # Display the plot
                                                                plt.tight_layout()
                                                                st.pyplot(fig)
                                                                
                                                                # Add summary statistics
                                                                st.write("### Depreciation Summary")
                                                                st.write(f"- Average annual depreciation: **{avg_depreciation:.1f}%**")
                                                                st.write(f"- Total depreciation from {int(year_stats['model_year'].max())} to {int(year_stats['model_year'].min())}: " +
                                                                        f"**{((year_stats['avg_price'].iloc[-1] - year_stats['avg_price'].iloc[0]) / year_stats['avg_price'].iloc[-1] * 100):.1f}%**")
                                                                
                                                                # Add explanation box
                                                                st.info("""
                                                                ### Understanding Car Depreciation
                                                                
                                                                This analysis shows how much value cars lose each year (depreciation):
                                                                
                                                                - Bars represent the percentage drop in average price from one model year to the next
                                                                - Higher percentages indicate steeper depreciation
                                                                - The red line shows the average depreciation across all years
                                                                - Newer cars typically depreciate faster than older ones
                                                                
                                                                This information can help you decide the optimal time to buy or sell.
                                                                """)
                                                            
                                                                # Add a scatterplot showing all cars with trend line
                                                                st.subheader("Price by Model Year")
                                                                st.write("This visualization shows all cars of this model and how their prices relate to model year.")
                                                                
                                                                # Create scatter plot for all cars of the selected model
                                                                fig2, ax2 = plt.subplots(figsize=(10, 6))
                                                                
                                                                # Scatter plot of all cars
                                                                scatter = ax2.scatter(
                                                                    model_analysis['Modellår_num'], 
                                                                    model_analysis['Totalpris_num'],
                                                                    alpha=0.7,
                                                                    s=80,
                                                                    c=model_analysis['Kilometerstand_num'],
                                                                    cmap='viridis'
                                                                )
                                                                
                                                                # Add colorbar to show mileage scale
                                                                cbar = plt.colorbar(scatter)
                                                                cbar.set_label('Mileage (km)')
                                                                
                                                                # Add a trend line
                                                                if len(model_analysis) > 1:
                                                                    # Calculate trend line using linear regression
                                                                    z = np.polyfit(model_analysis['Modellår_num'], model_analysis['Totalpris_num'], 1)
                                                                    p = np.poly1d(z)
                                                                    
                                                                    # Generate x values for the line
                                                                    x_range = np.linspace(
                                                                        model_analysis['Modellår_num'].min(), 
                                                                        model_analysis['Modellår_num'].max(), 
                                                                        100
                                                                    )
                                                                    
                                                                    # Plot the trend line
                                                                    ax2.plot(
                                                                        x_range, 
                                                                        p(x_range), 
                                                                        'r--',
                                                                        label=f'Trend Line: {z[0]:.0f} NOK/year'
                                                                    )
                                                                    
                                                                    # Calculate and display R-squared value
                                                                    corr_matrix = np.corrcoef(model_analysis['Modellår_num'], model_analysis['Totalpris_num'])
                                                                    corr = corr_matrix[0, 1]
                                                                    r_squared = corr**2
                                                                    ax2.text(
                                                                        0.05, 0.95, 
                                                                        f'R² = {r_squared:.2f}', 
                                                                        transform=ax2.transAxes,
                                                                        fontsize=10,
                                                                        verticalalignment='top'
                                                                    )
                                                                
                                                                # Add average price points by year
                                                                ax2.scatter(
                                                                    year_stats['model_year'],
                                                                    year_stats['avg_price'],
                                                                    color='red',
                                                                    s=120,
                                                                    marker='X',
                                                                    label='Average Price by Year'
                                                                )
                                                                
                                                                # Connect average price points with line
                                                                ax2.plot(
                                                                    year_stats['model_year'],
                                                                    year_stats['avg_price'],
                                                                    'r-',
                                                                    alpha=0.5
                                                                )
                                                                
                                                                # Set labels and title
                                                                ax2.set_xlabel('Model Year', fontsize=12)
                                                                ax2.set_ylabel('Price (NOK)', fontsize=12)
                                                                ax2.set_title(
                                                                    f'Price by Model Year: {selected_model}', 
                                                                    fontsize=14, 
                                                                    fontweight='bold'
                                                                )
                                                                
                                                                # Format y-axis with thousands separator
                                                                ax2.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
                                                                
                                                                # Add grid and legend
                                                                ax2.grid(True, linestyle='--', alpha=0.7)
                                                                ax2.legend()
                                                                
                                                                # Add annotations to identify potential outliers
                                                                for i, row in model_analysis.iterrows():
                                                                    # Check if car is a potential outlier
                                                                    # Compute z-score based on year group
                                                                    year_group = model_analysis[model_analysis['Modellår_num'] == row['Modellår_num']]
                                                                    if len(year_group) > 1:
                                                                        year_mean = year_group['Totalpris_num'].mean()
                                                                        year_std = year_group['Totalpris_num'].std()
                                                                        if year_std > 0:
                                                                            z_score = abs((row['Totalpris_num'] - year_mean) / year_std)
                                                                            if z_score > 1.5:  # Potentially interesting car
                                                                                ax2.annotate(
                                                                                    f"#{i}",
                                                                                    (row['Modellår_num'], row['Totalpris_num']),
                                                                                    xytext=(5, 5),
                                                                                    textcoords='offset points',
                                                                                    fontsize=8,
                                                                                    fontweight='bold',
                                                                                    color='darkred'
                                                                                )
                                                                
                                                                # Adjust plot for better display
                                                                plt.tight_layout()
                                                                st.pyplot(fig2)
                                                                
                                                                # Add explanation box for this visualization
                                                                st.info("""
                                                                ### How to Interpret the Price Chart
                                                                
                                                                This chart shows the relationship between model year and price:
                                                                
                                                                - Each dot is a car, colored by mileage (darker = higher mileage)
                                                                - Red X markers show the average price for each model year
                                                                - The dotted red line shows the overall price trend
                                                                - R² value shows how well model year predicts price (higher is better)
                                                                - Numbered points indicate potential outliers (good deals or overpriced)
                                                                
                                                                Look for cars below the trend line for potentially better deals.
                                                                """)
                                                        else:
                                                            st.info("Not enough years with price data to calculate depreciation.")
                                                    else:
                                                        st.info(f"Insufficient data for {selected_model}. Need at least 2 cars with different model years and valid prices.")
                                                else:
                                                    st.info(f"No data available for {selected_model} with valid model years and prices.")
                                            else:
                                                st.info("No car models found with multiple model years. Try scraping more cars for a better analysis.")
                                        except Exception as depreciation_error:
                                            st.error(f"Error analyzing depreciation: {str(depreciation_error)}")
                                            st.exception(depreciation_error)
                                    else:
                                        st.info("Model year data ('Modellår') is required for depreciation analysis. Please include it in the attributes to extract.")
                                    
                                    # Calculate price per kilometer and find best deals
                                    st.subheader("Best Bang for Buck")
                                    
                                    # Calculate the price per kilometer ratio
                                    plot_df['price_per_km'] = plot_df['Totalpris_num'] / plot_df['Kilometerstand_num']
                                    
                                    # Add a normalized score that considers both price and mileage
                                    # Lower is better for both price and mileage
                                    plot_df['price_normalized'] = (plot_df['Totalpris_num'] - plot_df['Totalpris_num'].min()) / (plot_df['Totalpris_num'].max() - plot_df['Totalpris_num'].min())
                                    plot_df['mileage_normalized'] = (plot_df['Kilometerstand_num'] - plot_df['Kilometerstand_num'].min()) / (plot_df['Kilometerstand_num'].max() - plot_df['Kilometerstand_num'].min())
                                    
                                    # BfB score: lower is better (weighted average of normalized price and mileage)
                                    # Weight price more heavily (70%) than mileage (30%)
                                    plot_df['BfB_score'] = 0.7 * plot_df['price_normalized'] + 0.3 * plot_df['mileage_normalized']
                                    
                                    # Rank the cars by BfB score
                                    plot_df['BfB_rank'] = plot_df['BfB_score'].rank()
                                    
                                    # Add BfB column to main dataframe
                                    # Create a mapping from index to rank
                                    bfb_rank_mapping = plot_df.set_index('index')['BfB_rank'].to_dict()
                                    
                                    # Add BfB column to the main dataframe
                                    df['BfB'] = df.index.map(lambda idx: f"#{int(bfb_rank_mapping.get(idx, 0))}" if idx in bfb_rank_mapping else "N/A")
                                    
                                    # Sort by price per kilometer (ascending)
                                    best_value_cars = plot_df.sort_values('BfB_score').head(5)
                                    
                                    if not best_value_cars.empty:
                                        st.write("Cars with the lowest price per kilometer:")
                                        
                                        # Create a more user-friendly display
                                        for i, row in best_value_cars.iterrows():
                                            col1, col2 = st.columns([3, 1])
                                            
                                            with col1:
                                                st.write(f"**#{i}: {row['title']}**")
                                                st.write(f"Price: {format_with_spaces(int(row['Totalpris_num']))} NOK | " +
                                                         f"Mileage: {format_with_spaces(int(row['Kilometerstand_num']))} km")
                                                if 'url' in row:
                                                    st.write(f"[View on Finn.no]({row['url']})")
                                            
                                            with col2:
                                                # Display price per km ratio in a metric
                                                price_per_km = row['price_per_km']
                                                st.metric(
                                                    "NOK/km", 
                                                    f"{price_per_km:.1f}"
                                                )
                                            
                                            st.write("---")
                                    else:
                                        st.write("Could not calculate best value cars")
                                else:
                                    st.error("Not enough valid numeric data for plotting. Try selecting different attributes.")
                                    
                                    # Show the actual data for debugging
                                    with st.expander("Show data for debugging"):
                                        st.write("Original data sample:")
                                        st.dataframe(df[["Totalpris", "Kilometerstand"]].head(10))
                                        
                                        st.write("Converted data:")
                                        st.dataframe(df[["Totalpris_clean", "Kilometerstand_clean", 
                                                         "Totalpris_num", "Kilometerstand_num"]].head(10))
                                    
                            except Exception as convert_error:
                                st.error(f"Error converting data: {str(convert_error)}")
                                st.exception(convert_error)
                                
                    except Exception as plot_error:
                        st.error(f"Error creating plot: {str(plot_error)}")
                        st.exception(plot_error)
                else:
                    st.warning(
                        "Both 'Totalpris' and 'Kilometerstand' are required for the analysis."
                    )
                    st.info(
                        "Please select both attributes in the configuration."
                    )
                
            with tab4:
                # Initialize the AI analyzer
                analyzer = CarAIAnalyzer(
                    logger=logger  # Pass the logger to the analyzer
                )
                
                # Create a container to hold the entire analysis section
                analysis_container = st.container()
                
                with analysis_container:
                    st.subheader("AI Analysis of Car Listings")
                    st.write("Get AI insights about whether each car is a good deal.")
                    
                    # Add debug expander for troubleshooting
                    with st.expander("🔧 Debug Information", expanded=False):
                        st.write("### Session State")
                        st.json({
                            "analysis_requested": st.session_state.get("analysis_requested", False),
                            "analysis_result": "Available" if st.session_state.get("analysis_result") else "None",
                            "analyzed_car": "Available" if st.session_state.get("analyzed_car") else "None",
                            "clicks": st.session_state.get("clicks", {})
                        })
                        
                        st.write("### Recent Logs")
                        # Try to read the most recent logs
                        try:
                            log_file = os.path.join("logs", "app.log")
                            if os.path.exists(log_file):
                                with open(log_file, "r") as f:
                                    # Get last 20 lines
                                    lines = f.readlines()[-20:]
                                    st.code("".join(lines), language="text")
                            else:
                                st.info("No log file found.")
                                
                            ai_log_file = os.path.join("logs", "ai_analyzer.log")
                            if os.path.exists(ai_log_file):
                                st.write("### AI Analyzer Logs")
                                with open(ai_log_file, "r") as f:
                                    # Get last 20 lines
                                    lines = f.readlines()[-20:]
                                    st.code("".join(lines), language="text")
                        except Exception as e:
                            st.error(f"Error reading logs: {str(e)}")
                    
                    # Initialize session state variables if they don't exist
                    if 'analysis_result' not in st.session_state:
                        st.session_state.analysis_result = None
                    if 'analyzed_car' not in st.session_state:
                        st.session_state.analyzed_car = None
                    if 'analysis_requested' not in st.session_state:
                        st.session_state.analysis_requested = False
                    if 'selected_car' not in st.session_state:
                        st.session_state.selected_car = None
                    # Initialize clicks dictionary for button state tracking
                    if 'clicks' not in st.session_state:
                        st.session_state.clicks = {}
                    
                    # Button callback functions
                    def start_analysis():
                        """Callback to initiate analysis"""
                        if 'car_selector' in st.session_state:
                            car_title = st.session_state.car_selector
                            logger.info(f"Starting analysis for {car_title}")
                            st.session_state.analysis_requested = True
                            st.session_state.selected_car = car_title
                    
                    def reset_analysis():
                        """Reset the analysis state and redirect to main tab"""
                        logger.info("Resetting analysis state")
                        # Reset only the analysis state, not the entire session
                        st.session_state.analysis_result = None
                        st.session_state.analyzed_car = None
                        st.session_state.analysis_requested = False
                        # Keep selected_car to allow the user to select it again
                        # Reset click counter for any buttons that might use it
                        if "clicks" in st.session_state:
                            st.session_state.clicks = 0
                        # No redirection to keep the user in the same tab
                    
                    def clear_analysis_data(reset_to_tab=None):
                        """Callback to clear analysis data and optionally redirect to another tab"""
                        logger.info("Clearing analysis data")
                        st.session_state.analysis_result = None
                        st.session_state.analyzed_car = None
                        st.session_state.analysis_requested = False
                        st.session_state.selected_car = None
                        st.session_state.clicks = {}
                        
                        # Rather than using experimental_rerun, simply set a URL parameter
                        # Let Streamlit's natural rerun behavior handle the rest
                        if reset_to_tab:
                            # Just set the query parameter without forcing a rerun
                            st.query_params["tab"] = reset_to_tab
                    
                    # Function to handle analysis 
                    def run_analysis(car_data):
                        try:
                            logger.info(f"Starting analysis for car: {car_data.get('title', 'unknown')}")
                            st.session_state.analyzed_car = car_data  
                            # Return the analysis result
                            result = analyzer.analyze_car(car_data)
                            logger.info(f"Analysis completed: {result[:50]}...")
                            return result
                        except Exception as e:
                            logger.error(f"Error in run_analysis: {str(e)}", exc_info=True)
                            return f"Failed to analyze: {str(e)}"
                    
                    if analyzer.is_available:
                        # Simple dropdown to select a car
                        if not df.empty:
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                # Persist the selected car between reruns
                                selected_index = 0
                                if st.session_state.selected_car in df['title'].tolist():
                                    selected_index = df['title'].tolist().index(st.session_state.selected_car)
                                
                                car_to_analyze = st.selectbox(
                                    "Select a car to analyze:",
                                    options=df['title'].tolist(),
                                    index=selected_index,
                                    key="car_selector"
                                )
                                
                                # Add automatic analysis trigger when car is selected
                                if st.session_state.get("car_selector") != st.session_state.get("selected_car"):
                                    logger.info(f"Car selection changed to {st.session_state.car_selector}")
                                    # Only trigger analysis if explicitly changing selection
                                    if st.session_state.get("selected_car") is not None:  
                                        st.session_state.analysis_requested = True
                                        st.session_state.selected_car = st.session_state.car_selector
                                        st.session_state.analysis_result = None  # Clear previous result
                            
                            with col2:
                                # Analyze button that sets analysis_requested to True
                                if st.button(
                                    "Analyze This Car", 
                                    type="primary", 
                                    key="analyze_button",
                                    use_container_width=True
                                ):
                                    logger.info(f"Analyze button clicked for {st.session_state.car_selector}")
                                    st.session_state.analysis_requested = True
                                    st.session_state.selected_car = st.session_state.car_selector
                                    st.session_state.analysis_result = None  # Clear previous result
                            
                            # Only run the analysis if requested and not already analyzed
                            if st.session_state.analysis_requested:
                                # Check if we need to analyze or already have analysis for this car
                                car_data = df[df['title'] == st.session_state.selected_car].iloc[0].to_dict()
                                
                                # Only analyze if we don't have a result yet or we're analyzing a different car
                                if (st.session_state.analysis_result is None or 
                                    st.session_state.analyzed_car is None or
                                    st.session_state.analyzed_car.get('title', '') != st.session_state.selected_car):
                                
                                    logger.info(f"Processing analysis for {st.session_state.selected_car}")
                                    
                                    # Create a progress bar
                                    progress_placeholder = st.empty()
                                    progress_bar = progress_placeholder.progress(0)
                                    
                                    progress_status = st.empty()
                                    progress_status.info("Starting analysis...")
                                    logger.info("Progress indicators initialized")
                                    
                                    try:
                                        # Update progress in stages
                                        progress_bar.progress(10)
                                        progress_status.info("Sending your request to AI system...")
                                        logger.info("Progress 10% - Sending request")
                                        time.sleep(0.5)
                                        
                                        progress_bar.progress(30)
                                        progress_status.info("Analyzing car details with Azure OpenAI...")
                                        logger.info("Progress 30% - Request sent, waiting for response")
                                        
                                        # Run the analysis and store the result
                                        logger.info("Calling run_analysis function")
                                        analysis_result = run_analysis(car_data)
                                        st.session_state.analysis_result = analysis_result
                                        logger.info("Analysis result stored in session state")
                                        
                                        progress_bar.progress(90)
                                        progress_status.info("Formatting results...")
                                        logger.info("Progress 90% - Formatting results")
                                        time.sleep(0.5)
                                        
                                        progress_bar.progress(100)
                                        progress_status.success("Analysis complete!")
                                        logger.info("Progress 100% - Analysis complete")
                                        
                                        # Remove progress indicators after completion
                                        time.sleep(0.5)
                                        progress_placeholder.empty()
                                        progress_status.empty()
                                        logger.info("Progress indicators cleared")
                                        
                                    except Exception as e:
                                        logger.error(f"Error during analysis UI update: {str(e)}", exc_info=True)
                                        progress_bar.progress(100)
                                        progress_status.error(f"Error during analysis: {str(e)}")
                                        st.session_state.analysis_result = f"Failed to analyze: {str(e)}"
                              
                                # Always display results when analysis is requested and we have data
                                if st.session_state.analyzed_car is not None:
                                    logger.info("Displaying analysis results from session state")
                                    # Create a visual separator
                                    st.markdown("---")
                                    
                                    # Display results in tabs instead of columns for better layout
                                    detail_tab, analysis_tab = st.tabs(["Car Details", "AI Analysis"])
                                    
                                    with detail_tab:
                                        car_data = st.session_state.analyzed_car
                                        
                                        # Show both image and details in a more organized layout
                                        car_col1, car_col2 = st.columns([1, 1])
                                        
                                        with car_col1:
                                            # Show car image if available
                                            if 'image_url' in car_data and car_data['image_url']:
                                                st.image(car_data['image_url'], width=300)
                                        
                                        with car_col2:
                                            # Only show most important details
                                            important_fields = ['title', 'Totalpris', 'Kilometerstand', 'Modellår']
                                            for field in important_fields:
                                                if field in car_data:
                                                    label = "Model" if field == 'title' else field
                                                    st.write(f"**{label}:** {car_data[field]}")
                                            
                                            # Add a few more fields if available
                                            optional_fields = ['Girkasse', 'Drivstoff', 'Motor']
                                            for field in optional_fields:
                                                if field in car_data and car_data[field]:
                                                    st.write(f"**{field}:** {car_data[field]}")
                                            
                                            if 'url' in car_data:
                                                st.write(f"[View on Finn.no]({car_data['url']})")
                                    
                                    with analysis_tab:
                                        # Show analysis result or loading state
                                        if st.session_state.analysis_result is not None:
                                            analysis = st.session_state.analysis_result
                                            
                                            # Handle error cases
                                            if (analysis.startswith("Error") or 
                                                analysis.startswith("API request") or 
                                                analysis.startswith("Failed")):
                                                logger.error(f"Analysis error detected: {analysis}")
                                                # Show detailed error information
                                                st.error(analysis)
                                                
                                                # Provide troubleshooting info
                                                with st.expander("⚠️ Troubleshooting Tips", expanded=True):
                                                    st.write("""
                                                    ### Possible reasons for failure:
                                                    - Azure OpenAI service might be experiencing high load
                                                    - Network connectivity issues
                                                    - API key or configuration issues
                                                    - Timeout during the API request
                                                    
                                                    Check the logs in the debug panel for more details.
                                                    """)
                                                    
                                                    # Add API configuration info without revealing key
                                                    st.write("### API Configuration Status")
                                                    endpoint_available = bool(os.getenv("AZURE_OPENAI_ENDPOINT", ""))
                                                    api_key_available = bool(os.getenv("AZURE_OPENAI_API_KEY", ""))
                                                    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
                                                    
                                                    st.json({
                                                        "endpoint_configured": endpoint_available,
                                                        "api_key_configured": api_key_available,
                                                        "deployment_name": deployment_name
                                                    })
                                                
                                                # Add button to retry that resets the state without forced rerun
                                                if st.button("Try Again", key="retry_analysis"):
                                                    logger.info("Try Again button clicked")
                                                    st.session_state.analysis_requested = False
                                                    st.session_state.analysis_result = None
                                                    st.session_state.analyzed_car = None
                                                
                                                # Add a button to clear data and run new analysis
                                                clear_col1, clear_col2 = st.columns([1, 1])
                                                with clear_col1:
                                                    if st.button("Clear Analysis Data", key="clear_analysis_data"):
                                                        logger.info("Clear Analysis Data button clicked")
                                                        st.session_state.analysis_result = None
                                                        st.session_state.analyzed_car = None
                                                        st.session_state.analysis_requested = False
                                                        st.session_state.selected_car = None
                                                        st.session_state.clicks = {}
                                                with clear_col2:
                                                    # Use a simpler approach without forced reruns
                                                    if st.button("🔄 Run New Analysis", key="run_new_search"):
                                                        logger.info("Run New Search button clicked")
                                                        # Reset all session state for the analysis tab
                                                        reset_analysis()  # Call the reset_analysis function
                                                        # Additional reset for the selected car that reset_analysis preserves
                                                        st.session_state.selected_car = None
                                                        # Simply set the tab parameter
                                                        st.query_params["tab"] = "scraper"
                                            else:
                                                logger.info("Formatting successful analysis")
                                                # Format analysis for better display
                                                if "•" in analysis or "-" in analysis or "*" in analysis:
                                                    # It's already in bullet points, just display it
                                                    st.markdown(analysis)
                                                else:
                                                    # Try to format it as bullet points if it's not already
                                                    points = analysis.split("\n")
                                                    formatted_points = []
                                                    for point in points:
                                                        if point.strip():
                                                            formatted_points.append(f"• {point.strip()}")
                                                    
                                                    if formatted_points:
                                                        st.markdown("\n".join(formatted_points))
                                                    else:
                                                        # Fallback if we couldn't parse into points
                                                        st.write(analysis)
                                                
                                                # Add a helpful conclusion
                                                st.markdown("---")
                                                st.caption("This analysis is generated by AI and should be used as a guideline only.")
                                                
                                                # Add a button to reset the analysis state for a new analysis
                                                if st.button("Analyze Another Car", key="rerun_analysis"):
                                                    logger.info("Analyze Another Car button clicked")
                                                    st.session_state.analysis_requested = False
                                                    st.session_state.analysis_result = None
                                                    st.session_state.analyzed_car = None
                                                    st.session_state.selected_car = None
                                        else:
                                            # If we have requested analysis but don't have a result yet
                                            st.info("Analysis in progress... please wait.")
                        else:
                            st.warning("No car data available. Please run the scraper first.")
                    else:
                        st.warning(
                            "Azure OpenAI API configuration is missing. Please check your environment variables."
                        )
                        st.info(
                            "Create a .env file in the project root directory with these variables:"
                        )
                        
                        # Example .env file contents
                        st.code("""# .env file example
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
AZURE_OPENAI_API_VERSION=2025-02-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini""", language="text")
            
            with tab5:
                st.download_button(
                    label="Download CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name="finn_car_data.csv",
                    mime="text/csv"
                )
                
                json_str = json.dumps(
                    result_data, 
                    ensure_ascii=False, 
                    indent=2
                )
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="finn_car_data.json",
                    mime="application/json"
                )
                
        except Exception as e:
            st.error(f"An error occurred during scraping: {str(e)}")
            st.exception(e)
else:
    # Only show documentation when scraper is not running
    with st.expander("Documentation", expanded=True):
        st.markdown("""
        ## How to use this amazing tool
        
        1. Enter a Finn.no search URL with your desired filters
        2. Select the attributes you want to extract from each car listing
        3. Choose between scraping a limited number or all matching cars
        4. Click "Run Scraper" to start the process
        5. View results as a table or JSON, and download as CSV or JSON
        
        ## Tips for Better Results
        
        - Apply specific filters on Finn.no before copying the URL
        - The scraper works best with common car attributes
        - For custom attributes, use the exact name as it appears on Finn.no
        - Scraping many listings may take time and might be rate-limited
        - The tool automatically handles pagination to find all matching cars
        
        ## Want to contribute?

        Reach out to edehvide and discuss your opportunities in this project.
        Check out his amazing github repo:
        https://github.com/edevardHvide/edes-bilbot  

        """)

# Footer
st.markdown("---")
st.caption("Edes bilskrapebot | Use responsibly in accordance with terms") 