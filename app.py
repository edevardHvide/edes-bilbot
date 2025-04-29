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

# Load environment variables
load_dotenv()

# Get configuration from environment variables
DEFAULT_SEARCH_URL = os.getenv(
    'DEFAULT_SEARCH_URL', 
    'https://www.finn.no/mobility/search/car?registration_class=1'
)
DEFAULT_LIMIT = int(os.getenv('DEFAULT_LIMIT', '5'))

# Function to format numbers with Norwegian style (spaces as thousands separator)
def format_with_spaces(num):
    """Format a number with spaces as thousands separator"""
    return f"{int(num):,}".replace(',', ' ')

# Easter egg comments for interesting cars
EASTER_EGGS = [
    "Daamn this one looks nice! 🔥",
    "Wow, what a find! 👀",
    "This one's a steal at that price! 💰",
    "Classic beauty right here! ✨",
    "Dream car alert! 🚨",
    "Someone's going to be lucky with this one! 🍀",
    "Perfect weekend cruiser! 🏎️",
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
    page_title="Edes bilskrapebot",
    page_icon="🚗",
    layout="wide"
)

# Main heading
st.title("Welcome to Edes bilskrapebot")

# Sidebar for inputs
with st.sidebar:
    st.header("Scraper Configuration")
    
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
    for attr in common_attributes:
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

# Main content area
if run_scraper:
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
                
                # Update stats
                current_page.text(
                    f"📄 Page: {scraping_stats['current_page']}/{scraping_stats['total_pages']}"
                )
                total_listings_found.text(
                    f"🚗 Found: {scraping_stats['total_listings_found']} cars"
                )
                processed_count.text(
                    f"✅ Processed: {scraping_stats['processed']} cars"
                )
                
                # Show easter egg comment occasionally (about 20% of listings)
                if scraping_stats.get("new_listing") and random.random() < 0.2:
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
                    listing_info.text(f"🔎 Processing: {listing_data['title']}")
            
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
            scraper.scrape_listings(limit=limit)
            
            # Save as temporary CSV
            temp_dir = tempfile.gettempdir()
            temp_csv = os.path.join(temp_dir, "finn_car_data.csv")
            df = scraper.save_to_csv(temp_csv)
            
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
            - Interesting cars found: {scraping_stats['interesting_cars']}
            """)
            
            # Convert data to dictionary for JSON serialization
            result_data = df.to_dict(orient='records')
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Data Table", "JSONderulo", "Analytics", "Download"]
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
                                # Show original data format for debugging
                                st.write("Sample data before conversion:")
                                st.write(df[["Totalpris", "Kilometerstand"]].head())
                                
                                # Clean data - remove non-numeric characters
                                df['Totalpris_clean'] = df['Totalpris'].astype(str).str.replace(r'[^\d]', '', regex=True)
                                df['Kilometerstand_clean'] = df['Kilometerstand'].astype(str).str.replace(r'[^\d]', '', regex=True)
                                
                                # Convert to numeric, coerce errors to NaN
                                df['Totalpris_num'] = pd.to_numeric(df['Totalpris_clean'], errors='coerce')
                                df['Kilometerstand_num'] = pd.to_numeric(df['Kilometerstand_clean'], errors='coerce')
                                
                                # Show conversion results for debugging
                                st.write("Conversion results:")
                                st.write(f"Valid price values: {df['Totalpris_num'].notna().sum()} of {len(df)}")
                                st.write(f"Valid mileage values: {df['Kilometerstand_num'].notna().sum()} of {len(df)}")
                                
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
                                    
                                    # Calculate price per kilometer and find best deals
                                    st.subheader("Best Bang for Buck")
                                    
                                    # Calculate the price per kilometer ratio
                                    plot_df['price_per_km'] = plot_df['Totalpris_num'] / plot_df['Kilometerstand_num']
                                    
                                    # Sort by price per kilometer (ascending)
                                    best_value_cars = plot_df.sort_values('price_per_km').head(5)
                                    
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

# API Documentation
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