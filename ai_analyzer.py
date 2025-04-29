import openai
import os
from dotenv import load_dotenv

class CarAIAnalyzer:
    """
    Simple class to handle AI analysis of car listings using OpenAI
    """
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        self.api_key = os.getenv('OPENAI_API_KEY', '')
        self.is_available = bool(self.api_key)
        
        # Set up OpenAI client if API key is available
        if self.is_available:
            openai.api_key = self.api_key
    
    def analyze_car(self, car_data):
        """
        Analyze a single car listing and return the AI's assessment
        
        Args:
            car_data (dict): Dictionary containing car listing data
            
        Returns:
            str: AI analysis of the car or error message
        """
        if not self.is_available:
            return "OpenAI API key not found. Cannot perform analysis."
        
        try:
            # Extract essential car info
            car_info = {
                "Title": car_data.get('title', 'N/A'),
                "Price": car_data.get('Totalpris', 'N/A'),
                "Mileage": car_data.get('Kilometerstand', 'N/A'),
                "Year": car_data.get('Modellår', 'N/A')
            }
            
            # Add optional fields if they exist
            optional_fields = [
                'Girkasse', 'Drivstoff', 'Motor', 'Effekt', 'Hjuldrift'
            ]
            for field in optional_fields:
                if field in car_data and car_data[field]:
                    car_info[field] = car_data[field]
            
            # Create prompt
            prompt_parts = ["Analyze this car listing:"]
            for key, value in car_info.items():
                prompt_parts.append(f"{key}: {value}")
            
            prompt_parts.append(
                "\nIs this a good deal? Why or why not? Keep it under 100 words."
            )
            prompt_text = "\n".join(prompt_parts)
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a car expert analyzing used car listings."
                    },
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            # Extract analysis
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error analyzing car: {str(e)}" 