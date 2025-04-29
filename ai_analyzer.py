"""
AI Analyzer for Finn Car Scraper

This module provides AI-powered analysis of car listings using OpenAI.
"""
import streamlit as st
import os
import openai
from dotenv import load_dotenv

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
    
    # Check if API key is available
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
    
    # Check if data is available
    if df is None or df.empty:
        st.warning(
            "No car data available. Please run the scraper first to collect car listings."
        )
        return
    
    # Simple instruction
    st.write("Analyze cars to get AI insights about whether they're good deals.")
    
    # Select cars to analyze
    car_options = df['title'].tolist()
    selected_car = st.selectbox(
        "Select a car to analyze",
        options=car_options
    )
    
    # Analyze button
    if st.button("Analyze Car", type="primary"):
        if not selected_car:
            st.error("Please select a car to analyze.")
            return
            
        with st.spinner("Analyzing car..."):
            try:
                # Get the selected car data
                car_data = df[df['title'] == selected_car].iloc[0].to_dict()
                
                # Get essential car details
                car_info = {
                    "Title": car_data.get('title', 'N/A'),
                    "Price": car_data.get('Totalpris', 'N/A'),
                    "Year": car_data.get('Modellår', 'N/A'),
                    "Mileage": car_data.get('Kilometerstand', 'N/A')
                }
                
                # Add optional fields if they exist
                optional_fields = [
                    'Girkasse', 'Drivstoff', 'Motor', 'Effekt', 'Hjuldrift'
                ]
                for field in optional_fields:
                    if field in car_data and car_data[field]:
                        car_info[field] = car_data[field]
                
                # Create prompt
                prompt = f"""Analyze this car listing:

Title: {car_info['Title']}
Price: {car_info['Price']}
Year: {car_info['Year']}
Mileage: {car_info['Mileage']}
"""
                # Add optional fields to prompt
                for key, value in car_info.items():
                    if key not in ['Title', 'Price', 'Year', 'Mileage']:
                        prompt += f"{key}: {value}\n"
                
                prompt += "\nIs this a good deal? Why or why not? Keep it under 100 words."
                
                # Call OpenAI API
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a car expert analyzing used car listings."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                
                # Extract and display analysis
                analysis = response.choices[0].message.content
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Car Details")
                    for key, value in car_info.items():
                        st.write(f"**{key}:** {value}")
                    
                    if 'url' in car_data:
                        st.write(f"[View on Finn.no]({car_data['url']})")
                
                with col2:
                    st.subheader("AI Analysis")
                    st.write(analysis)
                    
            except Exception as e:
                st.error(f"Error analyzing car: {str(e)}")
                st.info("Please check your OpenAI API key and try again.") 