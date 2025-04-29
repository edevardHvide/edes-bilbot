import streamlit as st
import json
from finn_car_scraper_app import FinnCarScraper
import os
import tempfile

st.set_page_config(
    page_title="Finn Car Scraper API",
    page_icon="🚗",
    layout="wide"
)

# Main heading
st.title("Finn Car Scraper API")

# Sidebar for inputs
with st.sidebar:
    st.header("Scraper Configuration")
    
    # Base URL input
    base_url = st.text_input(
        "Enter Finn.no search URL",
        placeholder="https://www.finn.no/mobility/search/car?...",
        help="Copy and paste a full URL from finn.no after applying filters"
    )
    
    # Attributes to extract
    st.subheader("Attributes to Extract")
    st.caption("Select the car attributes you want to extract")
    
    common_attributes = [
        "Modellår",
        "Kilometerstand", 
        "1. gang registrert",
        "Totalpris", 
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
    
    # Limit number of listings
    limit = st.number_input(
        "Number of listings to scrape", 
        min_value=1, 
        max_value=100, 
        value=5
    )
    
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
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Initializing scraper...")
            scraper = FinnCarScraper(
                base_url, 
                selected_attributes, 
                progress_callback=lambda c, t, m: (
                    progress_bar.progress(c/t), 
                    status_text.text(m)
                )
            )
            
            # Run the scraper
            status_text.text("Starting scraper...")
            scraper.scrape_listings(limit=limit)
            
            # Save as temporary CSV
            temp_dir = tempfile.gettempdir()
            temp_csv = os.path.join(temp_dir, "finn_car_data.csv")
            df = scraper.save_to_csv(temp_csv)
            
            # Display results
            status_text.text("Scraping completed!")
            progress_bar.progress(1.0)
            
            # Convert data to dictionary for JSON serialization
            result_data = df.to_dict(orient='records')
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Data Table", "JSON", "Download"])
            
            with tab1:
                st.dataframe(df, use_container_width=True)
                
            with tab2:
                st.json(result_data)
                
            with tab3:
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
with st.expander("API Documentation", expanded=False):
    st.markdown("""
    ## How to Use This API
    
    1. Enter a Finn.no search URL with your desired filters
    2. Select the attributes you want to extract from each car listing
    3. Set the number of listings to scrape (more = longer wait)
    4. Click "Run Scraper" to start the process
    5. View results as a table or JSON, and download as CSV or JSON
    
    ## Tips for Better Results
    
    - Apply specific filters on Finn.no before copying the URL
    - The scraper works best with common car attributes
    - For custom attributes, use the exact name as it appears on Finn.no
    - Scraping many listings may take time and might be rate-limited
    
    ## Deployment
    
    This app can be deployed on Render.com as a web service.
    """)

# Footer
st.markdown("---")
st.caption("Finn Car Scraper API | Use responsibly and in accordance with terms") 