"""
AI Analyzer for Finn Car Scraper

This module provides AI-powered analysis of car listings using OpenAI.
"""
import streamlit as st
import os
import openai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Set up OpenAI client if API key is available
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def render_ai_analysis_tab(df):
    """Render the AI Analysis tab with car analysis functionality"""
    st.subheader("AI Analysis of Car Listings")
    
    # Check if data is available first
    if df is None or df.empty:
        st.warning("No car data available. Please run the scraper first.")
        return
    
    # Check if API key is configured
    if not OPENAI_API_KEY:
        st.warning(
            "OpenAI API key not found. Set the OPENAI_API_KEY environment "
            "variable to enable AI analysis."
        )
        st.info(
            "Create a .env file in the project root directory and add: "
            "OPENAI_API_KEY=your_api_key_here"
        )
        
        # Example .env file contents
        st.code("""# .env file
DEFAULT_SEARCH_URL=https://www.finn.no/mobility/search/car?registration_class=1
DEFAULT_LIMIT=5
OPENAI_API_KEY=your_openai_api_key_here""", language="text")
        return
    
    # Simple instruction
    st.write("Get AI insights about whether a car is a good deal.")
    
    # Initialize session state for caching analysis results
    if 'car_analysis' not in st.session_state:
        st.session_state.car_analysis = {}
    
    # Show a 'clear cache' option to reset cached analyses
    if st.session_state.car_analysis and st.button("Clear Cached Analyses"):
        st.session_state.car_analysis = {}
        st.success("Analysis cache cleared")
        time.sleep(1)  # Give the success message time to show
        st.experimental_rerun()
    
    try:
        # Select cars to analyze - handle if title column is missing
        if 'title' not in df.columns:
            st.error("The 'title' column is missing from the data.")
            return
            
        car_options = df['title'].tolist()
        if not car_options:
            st.error("No car titles found in the data.")
            return
            
        selected_car = st.selectbox(
            "Select a car to analyze",
            options=car_options
        )
        
        # Skip API call entirely if already analyzed
        already_analyzed = selected_car in st.session_state.car_analysis
        cached_status = " (cached)" if already_analyzed else ""
        
        # Analysis button
        if st.button(f"Analyze Car{cached_status}", type="primary"):
            if not selected_car:
                st.error("Please select a car to analyze.")
                return
            
            analysis_container = st.container()
            
            # If we've already analyzed this car, load from cache
            if already_analyzed:
                analysis = st.session_state.car_analysis[selected_car]
                
                with analysis_container:
                    st.success("Loaded from cache")
                    display_analysis_results(analysis)
                    
            # Otherwise perform the analysis
            else:
                with st.spinner("Analyzing car (this may take 5-10 seconds)..."):
                    # Get start time for performance monitoring
                    start_time = time.time()
                    
                    try:
                        # Get the selected car data
                        car_row = df[df['title'] == selected_car]
                        if car_row.empty:
                            st.error("Could not find the selected car.")
                            return
                            
                        car_data = car_row.iloc[0].to_dict()
                        
                        # Prepare a minimal set of car details for faster API response
                        car_info = {"Title": car_data.get('title', 'N/A')}
                        
                        # Only include the most important fields for analysis
                        important_fields = [
                            'Totalpris', 'Modellår', 'Kilometerstand', 
                            'Girkasse', 'Drivstoff'
                        ]
                        
                        for field in important_fields:
                            if field in car_data and car_data[field]:
                                car_info[field] = car_data[field]
                        
                        # Create a simpler prompt for faster processing
                        prompt = "Analyze this car listing briefly:\n\n"
                        for key, value in car_info.items():
                            prompt += f"{key}: {value}\n"
                        
                        prompt += "\nGood deal? Why/why not? Keep under 75 words."
                        
                        # Call OpenAI API with optimized parameters
                        response = openai.chat.completions.create(
                            model="gpt-3.5-turbo",  # Faster model
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.5,  # More deterministic
                            max_tokens=120,   # Limit token usage
                            request_timeout=30  # Prevent hanging on slow connections
                        )
                        
                        # Process elapsed time
                        elapsed_time = time.time() - start_time
                        
                        # Extract analysis
                        analysis = {
                            "text": response.choices[0].message.content,
                            "car_info": car_info,
                            "url": car_data.get('url', None),
                            "elapsed_time": elapsed_time
                        }
                        
                        # Save in session state
                        st.session_state.car_analysis[selected_car] = analysis
                        
                        with analysis_container:
                            display_analysis_results(analysis)
                            st.caption(f"Analysis completed in {elapsed_time:.1f} seconds")
                            
                    except Exception as e:
                        st.error(f"Error analyzing car: {str(e)}")
                        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

def display_analysis_results(analysis):
    """Display analysis results in a consistent format"""
    # Display results in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Car Details")
        for key, value in analysis["car_info"].items():
            st.write(f"**{key}:** {value}")
        
        if analysis.get("url"):
            st.write(f"[View on Finn.no]({analysis['url']})")
    
    with col2:
        st.subheader("AI Analysis")
        st.write(analysis["text"]) 